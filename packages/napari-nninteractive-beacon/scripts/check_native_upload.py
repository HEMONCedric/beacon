"""Synthetic-only smoke/benchmark: no patient images, always releases its lease.

Run from the BEACON environment with --url http://127.0.0.1:1528.
--large benchmarks a larger native CT; --predict also tests three GPU lassos.
"""

import argparse
import hashlib
import json
from time import perf_counter

import numpy as np
import SimpleITK as sitk

from nnInteractive.inference.remote import nnInteractiveRemoteInferenceSession
from nnInteractive.inference.remote.serialization import pack_array
from napari_nninteractive_beacon.native_grid import NATIVE_PATH
from napari_nninteractive_beacon.native_upload import upload_prepared
from napari_nninteractive_beacon.volume_geometry import LassoPrompt, predict_three_lassos, prepare_volume


class VerifyingSession(nnInteractiveRemoteInferenceSession):
    def _post_binary(self, path, meta, *args, **kwargs):
        if path == NATIVE_PATH:
            meta = {**meta, "include_sha256": True}
        response = super()._post_binary(path, meta, *args, **kwargs)
        if path == NATIVE_PATH:
            self.native_info = response.json()
        return response


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--large", action="store_true")
    parser.add_argument("--predict", action="store_true")
    args = parser.parse_args()
    shape = (96, 384, 384) if args.large else (20, 32, 36)
    rng = np.random.default_rng(140)
    z, y, x = np.ogrid[tuple(slice(n) for n in shape)]
    body = ((y - shape[1] / 2) / (shape[1] * .4))**2 + ((x - shape[2] / 2) / (shape[2] * .35))**2 < 1
    data = np.where(body, 40 + .7 * x + .3 * y + .9 * z, -1024)
    data = (data + rng.normal(0, 10, shape)).astype(np.int16)
    image = sitk.GetImageFromArray(data)
    image.SetSpacing((0.8, 0.8, 3.0))
    image.SetOrigin((123, -50, -1572.5))
    angle = np.deg2rad(23)
    image.SetDirection((1, 0, 0, 0, np.cos(angle), -np.sin(angle), 0, np.sin(angle), np.cos(angle)))
    prepared = prepare_volume(image, "synthetic CT")
    started = perf_counter()
    working_compressed_bytes = len(pack_array(prepared.data[None]))
    working_compression_s = perf_counter() - started
    with VerifyingSession(args.url) as session:
        report = upload_prepared(session, prepared, prepared.data[None], [1, 1, 1])
        if report["mode"] != "native":
            raise AssertionError("Server did not use native upload: " + str(report))
        digest = hashlib.sha256(memoryview(prepared.data).cast("B")).hexdigest()
        assert session.native_info["working_sha256"] == digest, "Full working-volume voxels differ"
        report.update(working_compressed_bytes=working_compressed_bytes,
                      working_compression_s=working_compression_s, all_voxels_identical=True,
                      native_dtype=str(data.dtype), working_shape=list(prepared.data.shape))
        if args.predict:
            prompts = []
            for axis in range(3):
                bbox = [[max(0, n // 2 - 3), min(n, n // 2 + 4)] for n in prepared.data.shape]
                bbox[axis] = [prepared.data.shape[axis] // 2, prepared.data.shape[axis] // 2 + 1]
                crop = np.ones(tuple(hi - lo for lo, hi in bbox), np.uint8)
                prompts.append(LassoPrompt(axis, crop, bbox))
            started = perf_counter()
            result = predict_three_lassos(session, prompts, prepared.data.shape)
            assert result.shape == prepared.data.shape
            exported = prepared.mask_in_original_space(result)
            assert exported.GetSize() == image.GetSize()
            assert exported.GetOrigin() == image.GetOrigin()
            np.testing.assert_allclose(exported.GetDirection(), image.GetDirection())
            report.update(prediction_s=perf_counter() - started, foreground=int(np.count_nonzero(result)),
                          original_grid_export_verified=True)
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
