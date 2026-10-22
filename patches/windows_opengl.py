"""Keep Qt and VisPy on the same OpenGL implementation on Windows.

Call this before importing napari or Qt.  On other platforms it does nothing.
Users can opt out by setting ``QT_OPENGL`` themselves before starting BEACON.
"""

from __future__ import annotations

import os
import sys
from importlib.util import find_spec
from pathlib import Path


def configure_windows_opengl() -> None:
    if sys.platform != "win32":
        return

    os.environ.setdefault("QT_OPENGL", "software")
    if os.environ["QT_OPENGL"].lower() != "software" or os.environ.get("VISPY_GL_LIB"):
        return

    pyqt6 = find_spec("PyQt6")
    if pyqt6 is None or pyqt6.origin is None:
        return
    software_gl = Path(pyqt6.origin).parent / "Qt6" / "bin" / "opengl32sw.dll"
    if software_gl.is_file():
        os.environ["VISPY_GL_LIB"] = str(software_gl)
    else:
        print(f"BEACON warning: Qt software OpenGL library not found: {software_gl}", flush=True)
