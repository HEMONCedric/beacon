import json
import threading
import unittest
from types import SimpleNamespace

import httpx
import numpy as np
import SimpleITK as sitk
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.testclient import TestClient

from napari_nninteractive_beacon.native_grid import (
    CAPABILITIES_PATH, MAX_VOXELS, NATIVE_PATH, PROTOCOL_VERSION,
    grid_metadata, resample_native, same_grid,
)
from napari_nninteractive_beacon.native_upload import upload_prepared
from napari_nninteractive_beacon.server_native import install_native_routes
from napari_nninteractive_beacon.volume_geometry import prepare_layer, prepare_volume
from nnInteractive.inference.remote.serialization import pack_array, unpack_array


class Transport:
    _set_image_timeout = 30
    original_image_shape = None

    def __init__(self, http):
        self.http = http
        self.native_posts = 0
        self.legacy_posts = 0
        self.tamper = None

    def _get_json(self, path):
        response = self.http.get(path)
        response.raise_for_status()
        return response.json()

    def _post_binary(self, path, meta, body, timeout=None):
        self.native_posts += 1
        response = self.http.post(path, headers={"X-Meta": json.dumps(meta)}, content=body)
        response.raise_for_status()
        if self.tamper:
            info = response.json()
            self.tamper(info)
            return SimpleNamespace(json=lambda: info)
        return response

    def set_image(self, data, properties):
        self.legacy_posts += 1
        self.legacy_image = data


def volume(dtype=np.int16, direction=None):
    data = np.arange(5 * 7 * 9, dtype=dtype).reshape(5, 7, 9) - 100
    image = sitk.GetImageFromArray(data)
    image.SetSpacing((0.8, 1.4, 3.3))
    image.SetOrigin((123.25, -53.5, -1572.5))
    if direction is not None:
        image.SetDirection(direction)
    return image, prepare_volume(image, "test")


class NativeUploadTests(unittest.TestCase):
    def setUp(self):
        self.entry = SimpleNamespace(session=SimpleNamespace(), lock=threading.Lock(), marks=0)
        def mark():
            self.entry.marks += 1
        self.entry.mark_active = mark
        def set_image(data, properties):
            self.received = data.copy()
            self.properties = properties
        self.entry.session.set_image = set_image
        self.app = FastAPI()
        def auth(authorization: str = Header(default="")):
            if authorization != "Bearer test":
                raise HTTPException(401)
        def lease(x_lease_token: str = Header(default="")):
            if x_lease_token != "lease":
                raise HTTPException(410)
            return self.entry
        @self.app.post("/set_image", dependencies=[Depends(auth)])
        async def upstream(request: Request, entry=Depends(lease)):
            return {}
        install_native_routes(self.app)
        self.http = TestClient(self.app, headers={"Authorization": "Bearer test", "X-Lease-Token": "lease"})
        self.addCleanup(self.http.close)
        self.session = Transport(self.http)

    def test_native_transport_reconstructs_exact_voxels_for_all_acquisition_directions(self):
        angle = np.deg2rad(23)
        directions = [None, (0, 0, -1, 0, 1, 0, 1, 0, 0),
                      (-1, 0, 0, 0, -1, 0, 0, 0, 1),
                      (1, 0, 0, 0, np.cos(angle), -np.sin(angle), 0, np.sin(angle), np.cos(angle))]
        for dtype in (np.int16, np.float32, np.float64):
            for direction in directions:
                source, prepared = volume(dtype, direction)
                original = prepared.source_data.copy()
                report = upload_prepared(self.session, prepared, prepared.data[None], (1, 1, 1))
                self.assertEqual(report["mode"], "native")
                self.assertEqual(prepared.source_data.dtype, np.dtype(dtype))
                np.testing.assert_array_equal(self.received[0], prepared.data)
                np.testing.assert_array_equal(prepared.source_data, original)
                self.assertEqual(self.session.original_image_shape, (1, *prepared.data.shape))
                self.assertEqual(self.properties["spacing"], [1, 1, 1])
                # Patient-space export remains correct after native transfer.
                point = source.TransformIndexToPhysicalPoint((4, 3, 2))
                index = prepared.working.TransformPhysicalPointToIndex(point)
                mask = np.zeros(prepared.data.shape, np.uint8)
                mask[tuple(slice(max(0, n - 1), n + 2) for n in index[::-1])] = 1
                self.assertEqual(prepared.mask_in_original_space(mask)[4, 3, 2], 1)

    def test_transposed_active_layer_has_distinct_source_and_export_grids(self):
        source, _ = volume()
        source_data = sitk.GetArrayFromImage(source)
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as folder:
            path = str(Path(folder) / "image.nii.gz")
            sitk.WriteImage(source, path)
            prepared = prepare_layer(source_data.transpose(2, 1, 0), np.eye(4), {}, "active", path)
        self.assertNotEqual(prepared.source_grid["size_xyz"], list(prepared.original.GetSize()))
        self.assertEqual(upload_prepared(self.session, prepared, prepared.data[None], (1, 1, 1))["mode"], "native")
        np.testing.assert_array_equal(self.received[0], prepared.data)

    def test_authentication_and_lease_are_preserved(self):
        _, prepared = volume()
        meta = {"protocol_version": 1, "source_grid": prepared.source_grid,
                "target_grid": grid_metadata(prepared.working)}
        body = pack_array(prepared.source_data[None])
        for header, value, expected in (("Authorization", "", 401), ("X-Lease-Token", "", 410)):
            response = self.http.post(NATIVE_PATH, headers={header: value, "X-Meta": json.dumps(meta)}, content=body)
            self.assertEqual(response.status_code, expected)
        self.assertEqual(self.entry.marks, 0)

    def test_invalid_grids_protocol_and_shape_are_rejected_without_initialization(self):
        _, prepared = volume()
        original = {"protocol_version": 1, "source_grid": prepared.source_grid,
                    "target_grid": grid_metadata(prepared.working)}
        changes = [lambda m: m.update(protocol_version=999),
                   lambda m: m["target_grid"].update(size_xyz=[MAX_VOXELS + 1, 2, 2]),
                   lambda m: m["target_grid"].update(spacing_xyz_mm=[2, 1, 1]),
                   lambda m: m["target_grid"].update(origin_xyz_lps_mm=[0, 0, 0]),
                   lambda m: m["source_grid"].update(size_xyz=[1, 1, 1]),
                   lambda m: m["source_grid"].update(direction_xyz_lps=[0] * 9)]
        for change in changes:
            meta = json.loads(json.dumps(original))
            change(meta)
            response = self.http.post(NATIVE_PATH, headers={"X-Meta": json.dumps(meta)},
                                      content=pack_array(prepared.source_data[None]))
            self.assertEqual(response.status_code, 400, response.text)
        self.assertFalse(hasattr(self, "received"))

    def test_wrong_response_grid_or_values_stop_the_client_without_fallback(self):
        _, prepared = volume()
        for tamper in (lambda info: info["working_grid"].update(origin_xyz_lps_mm=[0, 0, 0]),
                       lambda info: info.update(original_image_shape=[1, 2, 3, 4]),
                       lambda info: info["sample_values"].__setitem__(0, 99999)):
            self.session.tamper = tamper
            with self.assertRaises(ValueError):
                upload_prepared(self.session, prepared, prepared.data[None], (1, 1, 1))
        self.assertEqual(self.session.legacy_posts, 0)

    def test_legacy_server_and_smaller_working_image_use_the_existing_transport(self):
        _, prepared = volume()
        self.session._get_json = lambda path: httpx.Response(404, request=httpx.Request("GET", "http://test" + path)).raise_for_status()
        report = upload_prepared(self.session, prepared, prepared.data[None], (1, 1, 1))
        self.assertEqual(report["mode"], "working")
        self.assertEqual(self.session.native_posts, 0)
        self.assertIs(self.session.legacy_image.base, prepared.data)
        image = sitk.GetImageFromArray(np.ones((5, 7, 9), np.float32))
        image.SetSpacing((0.5, 0.5, 0.5))
        prepared = prepare_volume(image, "already high resolution")
        report = upload_prepared(self.session, prepared, prepared.data[None], (1, 1, 1))
        self.assertEqual(report["reason"], "working image is smaller")

    def test_non_404_capability_errors_do_not_trigger_a_large_fallback_upload(self):
        _, prepared = volume()
        def unavailable(path):
            httpx.Response(401, request=httpx.Request("GET", "http://test" + path)).raise_for_status()
        self.session._get_json = unavailable
        with self.assertRaises(httpx.HTTPStatusError):
            upload_prepared(self.session, prepared, prepared.data[None], (1, 1, 1))
        self.assertEqual(self.session.legacy_posts, 0)

    def test_grid_validation_and_installation_are_idempotent(self):
        _, prepared = volume()
        grid = grid_metadata(prepared.working)
        self.assertTrue(same_grid(grid, grid))
        self.assertFalse(same_grid(None, grid))
        install_native_routes(self.app)
        self.assertEqual(sum(getattr(route, "path", None) == NATIVE_PATH for route in self.app.routes), 1)
        self.assertEqual(self.http.get(CAPABILITIES_PATH).json()["protocol_version"], PROTOCOL_VERSION)

    def test_optional_digest_checks_all_working_voxels_before_backend_mutates_them(self):
        import hashlib
        _, prepared = volume()
        def mutate(data, properties):
            data[:] = 12345
        self.entry.session.set_image = mutate
        meta = {"protocol_version": 1, "source_grid": prepared.source_grid,
                "target_grid": grid_metadata(prepared.working), "include_sha256": True}
        response = self.http.post(NATIVE_PATH, headers={"X-Meta": json.dumps(meta)},
                                  content=pack_array(prepared.source_data[None]))
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["working_sha256"],
                         hashlib.sha256(memoryview(prepared.data).cast("B")).hexdigest())


if __name__ == "__main__":
    unittest.main()
