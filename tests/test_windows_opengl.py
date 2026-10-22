from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from patches.windows_opengl import configure_windows_opengl


class WindowsOpenGLTests(unittest.TestCase):
    def test_other_platforms_are_unchanged(self) -> None:
        with patch.object(sys, "platform", "linux"), patch.dict(os.environ, {}, clear=True):
            configure_windows_opengl()
            self.assertNotIn("QT_OPENGL", os.environ)
            self.assertNotIn("VISPY_GL_LIB", os.environ)

    def test_windows_uses_matching_software_libraries(self) -> None:
        origin = "C:/Python/Lib/site-packages/PyQt6/__init__.py"
        with (
            patch.object(sys, "platform", "win32"),
            patch.dict(os.environ, {}, clear=True),
            patch("patches.windows_opengl.find_spec", return_value=SimpleNamespace(origin=origin)),
            patch.object(Path, "is_file", return_value=True),
        ):
            configure_windows_opengl()
            self.assertEqual(os.environ["QT_OPENGL"], "software")
            self.assertTrue(os.environ["VISPY_GL_LIB"].endswith("opengl32sw.dll"))

    def test_explicit_desktop_opengl_is_respected(self) -> None:
        with patch.object(sys, "platform", "win32"), patch.dict(os.environ, {"QT_OPENGL": "desktop"}, clear=True):
            configure_windows_opengl()
            self.assertNotIn("VISPY_GL_LIB", os.environ)

    def test_explicit_vispy_library_is_respected(self) -> None:
        with patch.object(sys, "platform", "win32"), patch.dict(
            os.environ, {"VISPY_GL_LIB": "custom-opengl.dll"}, clear=True
        ):
            configure_windows_opengl()
            self.assertEqual(os.environ["VISPY_GL_LIB"], "custom-opengl.dll")


if __name__ == "__main__":
    unittest.main()
