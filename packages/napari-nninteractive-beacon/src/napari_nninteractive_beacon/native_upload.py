"""Negotiate native uploads, keeping legacy nnInteractive servers compatible."""

from time import perf_counter

import httpx
import numpy as np

from .native_grid import (
    CAPABILITIES_PATH, NATIVE_PATH, PROTOCOL_VERSION, grid_metadata, same_grid, sample_values,
)


def upload_prepared(session, volume, working_image, spacing, progress=lambda message: None):
    def legacy(reason):
        progress("Sending the 1 mm image (" + reason + ")…")
        started = perf_counter()
        session.set_image(working_image, {"spacing": spacing})
        return {"mode": "working", "reason": reason, "request_s": perf_counter() - started}

    if volume.source_data is None or volume.source_grid is None or not hasattr(session, "_get_json"):
        return legacy("native upload unavailable")
    if volume.source_data.nbytes >= working_image.nbytes:
        return legacy("working image is smaller")
    try:
        caps = session._get_json(CAPABILITIES_PATH)
    except httpx.HTTPStatusError as error:
        if error.response.status_code in (404, 405):
            return legacy("standard server")
        raise
    if caps.get("protocol_version") != PROTOCOL_VERSION:
        return legacy("unsupported native-upload version")

    from nnInteractive.inference.remote.serialization import pack_array
    from nnInteractive.inference.remote.remote_session import _compression_threads

    progress("Compressing the original image…")
    started = perf_counter()
    payload = pack_array(volume.source_data[None], nthreads=_compression_threads())
    compression_s = perf_counter() - started
    target_grid = grid_metadata(volume.working)
    metadata = {"protocol_version": PROTOCOL_VERSION, "source_grid": volume.source_grid,
                "target_grid": target_grid}
    progress("Sending original image; reconstructing the exact 1 mm grid on the server…")
    started = perf_counter()
    response = session._post_binary(NATIVE_PATH, metadata, payload, timeout=session._set_image_timeout)
    request_s = perf_counter() - started
    info = response.json()
    if (info.get("protocol_version") != PROTOCOL_VERSION
            or not same_grid(info.get("working_grid"), target_grid)
            or info.get("original_image_shape") != list(working_image.shape)):
        raise ValueError("Server working grid differs from BEACON; refusing misaligned prompts")
    expected_samples = np.asarray(sample_values(working_image))
    received_samples = np.asarray(info.get("sample_values"), dtype=float)
    if received_samples.shape != expected_samples.shape or not np.allclose(
        received_samples, expected_samples, rtol=1e-6, atol=1e-4, equal_nan=True,
    ):
        raise ValueError("Server resampling differs from the displayed image; initialization stopped")
    session.original_image_shape = tuple(info["original_image_shape"])
    return {"mode": "native", "source_bytes": volume.source_data.nbytes,
            "working_bytes": working_image.nbytes, "compressed_bytes": len(payload),
            "compression_s": compression_s, "request_s": request_s,
            "server_timings": info.get("timings", {})}
