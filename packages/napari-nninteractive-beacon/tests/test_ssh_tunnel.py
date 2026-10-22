from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from napari_nninteractive_beacon.ssh_tunnel import (
    TunnelError,
    TunnelManager,
    spec_from_url,
)


class TunnelSpecTests(unittest.TestCase):
    def test_localhost_url_requests_a_local_forward(self) -> None:
        self.assertEqual(spec_from_url("http://127.0.0.1:1527").local_port, 1527)
        self.assertEqual(spec_from_url("http://localhost:1527").remote_port, 1527)

    def test_direct_server_url_does_not_request_a_tunnel(self) -> None:
        self.assertIsNone(spec_from_url("https://gpu.example.org:1527"))

    def test_invalid_url_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            spec_from_url("not-a-url")


class TunnelManagerTests(unittest.TestCase):
    def test_existing_local_service_needs_no_ssh_target(self) -> None:
        tunnel = TunnelManager()
        with patch("napari_nninteractive_beacon.ssh_tunnel.port_open", return_value=True):
            self.assertIn("already in use", tunnel.ensure(spec_from_url("http://localhost:1527"), ""))

    def test_missing_target_is_reported_without_starting_ssh(self) -> None:
        tunnel = TunnelManager()
        with patch("napari_nninteractive_beacon.ssh_tunnel.port_open", return_value=False):
            with self.assertRaisesRegex(TunnelError, "SSH target"):
                tunnel.ensure(spec_from_url("http://localhost:1527"), "")

    def test_ssh_forward_is_localhost_only_and_has_no_api_key(self) -> None:
        tunnel = TunnelManager(["ssh"])
        process = MagicMock()
        process.poll.return_value = None
        with (
            patch("napari_nninteractive_beacon.ssh_tunnel.port_open", side_effect=[False, True]),
            patch("napari_nninteractive_beacon.ssh_tunnel.subprocess.Popen", return_value=process) as popen,
        ):
            self.assertIn("open", tunnel.ensure(spec_from_url("http://localhost:1527"), "user@host"))
        command = popen.call_args.args[0]
        self.assertIn("127.0.0.1:1527:127.0.0.1:1527", command)
        self.assertIn("BatchMode=yes", command)
        self.assertEqual(command[-1], "user@host")
        self.assertNotIn("api", " ".join(command).lower())
        tunnel.close_all()


class WidgetTunnelTests(unittest.TestCase):
    def test_connect_opens_tunnel_before_claiming_session(self) -> None:
        from napari_nninteractive_beacon.widget_nninteractive import nnInteractiveWidgetBeacon

        widget = MagicMock()
        widget._ssh_tunnel_worker = None
        widget.server_url_edit.text.return_value = "http://127.0.0.1:1527"
        widget.ssh_target_edit.text.return_value = "user@host"
        widget._viewer = SimpleNamespace(status="")
        worker = MagicMock()
        tunnel = MagicMock()
        with (
            patch("napari_nninteractive_beacon.widget_nninteractive.manager", return_value=tunnel),
            patch("napari_nninteractive_beacon.widget_nninteractive.create_worker", return_value=worker) as create_worker,
        ):
            nnInteractiveWidgetBeacon._start_claim(widget)

        self.assertIs(create_worker.call_args.args[0], tunnel.ensure)
        self.assertEqual(create_worker.call_args.args[2], "user@host")
        self.assertEqual(create_worker.call_args.args[1].local_port, 1527)
        worker.start.assert_called_once_with()

    def test_minimal_widget_can_take_ssh_alias_from_config(self) -> None:
        from napari_nninteractive_minimal.widget_nninteractive import nnInteractiveWidgetMinimal

        widget = MagicMock()
        nnInteractiveWidgetMinimal._apply_inference_config(widget, {"ssh_target": "gpu-cluster"})
        widget.ssh_target_edit.setText.assert_called_once_with("gpu-cluster")


if __name__ == "__main__":
    unittest.main()
