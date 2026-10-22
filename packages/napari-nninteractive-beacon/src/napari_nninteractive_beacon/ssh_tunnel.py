"""Optional localhost-only SSH forwarding for a remote nnInteractive server."""

from __future__ import annotations

import atexit
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass
from typing import BinaryIO, Sequence
from urllib.parse import urlparse

DEFAULT_SERVER_URL = "http://127.0.0.1:1527"
SSH_TARGET_SETTING = "beacon/ssh_target"
LOCAL_HOSTS = {"127.0.0.1", "localhost", "::1"}


class TunnelError(RuntimeError):
    """The SSH forward could not be opened."""


@dataclass(frozen=True)
class TunnelSpec:
    local_port: int
    remote_port: int


def spec_from_url(url: str) -> TunnelSpec | None:
    """Return the forward needed for a localhost URL; direct URLs need none."""

    parsed = urlparse((url or "").strip())
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("Enter an http(s) server URL, including its host.")
    if parsed.hostname.lower() not in LOCAL_HOSTS:
        return None
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    return TunnelSpec(local_port=port, remote_port=port)


def port_open(port: int, timeout: float = 0.3) -> bool:
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=timeout):
            return True
    except OSError:
        return False


class TunnelManager:
    """Own the SSH forwards opened by BEACON, not forwards opened elsewhere."""

    def __init__(self, ssh_command: Sequence[str] | None = None) -> None:
        self._ssh_command = list(ssh_command) if ssh_command else None
        self._processes: dict[int, tuple[str, subprocess.Popen, BinaryIO]] = {}
        self._lock = threading.Lock()

    def ensure(self, spec: TunnelSpec, target: str, timeout: float = 20.0) -> str:
        """Open or reuse a local forward before the HTTP client connects."""

        target = target.strip()
        with self._lock:
            owned = self._processes.get(spec.local_port)
            if owned and owned[0] == target and owned[1].poll() is None and port_open(spec.local_port):
                return "SSH tunnel already running"
            if owned:
                self._stop(spec.local_port)
            if port_open(spec.local_port):
                return f"Local port {spec.local_port} already in use; using its existing service"
            if not target:
                raise TunnelError("Enter an SSH target (user@host) or start the server on this workstation.")
            executable = self._ssh_command or ([shutil.which("ssh")] if shutil.which("ssh") else None)
            if not executable:
                raise TunnelError("SSH client not found. Install OpenSSH or Git for Windows.")
            log = tempfile.TemporaryFile()
            flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if sys.platform == "win32" else 0
            process = subprocess.Popen(
                [
                    *executable,
                    "-N",
                    "-o", "BatchMode=yes",
                    "-o", "ExitOnForwardFailure=yes",
                    "-o", "ConnectTimeout=10",
                    "-o", "ServerAliveInterval=30",
                    "-o", "ServerAliveCountMax=3",
                    "-L", f"127.0.0.1:{spec.local_port}:127.0.0.1:{spec.remote_port}",
                    target,
                ],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=log,
                creationflags=flags,
            )
            self._processes[spec.local_port] = (target, process, log)

        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if process.poll() is not None:
                with self._lock:
                    log.seek(0)
                    detail = log.read().decode("utf-8", errors="replace").strip()
                    self._stop(spec.local_port)
                raise TunnelError(f"SSH tunnel failed: {detail or f'exit code {process.returncode}'}")
            if port_open(spec.local_port):
                return f"SSH tunnel open to {target}:{spec.remote_port}"
            time.sleep(0.2)
        with self._lock:
            self._stop(spec.local_port)
        raise TunnelError(f"SSH tunnel did not open within {timeout:.0f} seconds.")

    def close_all(self) -> None:
        with self._lock:
            for port in list(self._processes):
                self._stop(port)

    def _stop(self, port: int) -> None:
        entry = self._processes.pop(port, None)
        if entry is None:
            return
        _, process, log = entry
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=3)
        log.close()


_MANAGER = TunnelManager()
atexit.register(_MANAGER.close_all)


def manager() -> TunnelManager:
    return _MANAGER
