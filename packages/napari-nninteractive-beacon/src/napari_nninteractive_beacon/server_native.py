"""Add exact-grid native uploads to the existing nnInteractive server CLI.

Run python -m napari_nninteractive_beacon.server_native with the normal server
arguments. No napari/Qt installation is required by this extension.
"""

import inspect
import hashlib
import json
import logging
from time import perf_counter

from fastapi import HTTPException, Request
from starlette.concurrency import run_in_threadpool

from .native_grid import (
    CAPABILITIES_PATH, NATIVE_PATH, PROTOCOL_VERSION, resample_native, sample_values, validate_grid,
)

logger = logging.getLogger("beacon.native_upload")


def install_native_routes(app):
    if any(getattr(route, "path", None) == NATIVE_PATH for route in app.routes):
        return app
    upstream = next(route for route in app.routes
                    if getattr(route, "path", None) == "/set_image" and "POST" in route.methods)
    # Reuse the real server's authentication AND lease dependencies. Never
    # create a second model registry or bypass per-session locks/lease clocks.
    dependency = inspect.signature(upstream.endpoint).parameters["entry"].default

    async def capabilities():
        return {"protocol_version": PROTOCOL_VERSION, "resampling": "SimpleITK-linear-nearest-extrapolation"}

    async def set_native_image(request: Request, entry=dependency):
        try:
            meta = json.loads(request.headers.get("X-Meta", "{}"))
            if meta.get("protocol_version") != PROTOCOL_VERSION:
                raise ValueError("Unsupported native-upload protocol")
            validate_grid(meta.get("source_grid"))
            validate_grid(meta.get("target_grid"))
        except (ValueError, TypeError, AttributeError) as error:
            raise HTTPException(400, str(error)) from error
        body = await request.body()

        def work():
            from nnInteractive.inference.remote.serialization import unpack_array

            entry.mark_active()
            with entry.lock:
                try:
                    started = perf_counter()
                    source = unpack_array(body)
                    decode_s = perf_counter() - started
                    started = perf_counter()
                    working = resample_native(source, meta["source_grid"], meta["target_grid"])
                    resample_s = perf_counter() - started
                    del source
                    # Compute diagnostics before set_image: some versions
                    # preprocess/mutate the array asynchronously in place.
                    samples = sample_values(working)
                    digest = (hashlib.sha256(memoryview(working).cast("B")).hexdigest()
                              if meta.get("include_sha256") is True else None)
                    started = perf_counter()
                    entry.session.set_image(working, {"spacing": [1., 1., 1.]})
                    initialize_s = perf_counter() - started
                    entry.mark_active()
                except (ValueError, TypeError, AssertionError, RuntimeError) as error:
                    raise HTTPException(400, str(error)) from error
            timings = {"decode_s": decode_s, "resample_s": resample_s, "initialize_s": initialize_s}
            logger.info("native upload: compressed=%d bytes, shape=%s, decode=%.3fs resample=%.3fs initialize=%.3fs",
                        len(body), tuple(working.shape), decode_s, resample_s, initialize_s)
            return {"protocol_version": PROTOCOL_VERSION, "working_grid": meta["target_grid"],
                    "original_image_shape": list(working.shape), "sample_values": samples,
                    "working_sha256": digest, "timings": timings}

        return await run_in_threadpool(work)

    app.add_api_route(CAPABILITIES_PATH, capabilities, methods=["GET"], dependencies=upstream.dependencies)
    app.add_api_route(NATIVE_PATH, set_native_image, methods=["POST"], dependencies=upstream.dependencies)
    app.openapi_schema = None
    return app


def main():
    from nnInteractive.inference.server import main as server_main

    original_factory = server_main.make_app

    def make_extended_app(*args, **kwargs):
        return install_native_routes(original_factory(*args, **kwargs))

    server_main.make_app = make_extended_app
    try:
        return server_main.main()
    finally:
        server_main.make_app = original_factory


if __name__ == "__main__":
    raise SystemExit(main())
