"""IPC client for communicating with the daemon from the interactive CLI."""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from pathlib import Path

from coolcode.daemon.server import SOCKET_PATH, is_daemon_running

logger = logging.getLogger(__name__)


@dataclass
class DaemonInsight:
    """An insight received from the daemon."""

    message: str
    severity: str = "info"
    source: str = "daemon"
    related_files: list[str] | None = None


class DaemonClient:
    """Client for the daemon's Unix socket IPC."""

    def _send_command(self, cmd: str) -> dict | None:
        """Send a command to the daemon and return the response."""
        if not is_daemon_running():
            return None

        try:
            # Use synchronous socket for simplicity in the CLI context
            import socket
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            sock.settimeout(3.0)
            sock.connect(str(SOCKET_PATH))
            sock.sendall(json.dumps({"cmd": cmd}).encode() + b"\n")

            data = b""
            while True:
                chunk = sock.recv(4096)
                if not chunk:
                    break
                data += chunk
                if b"\n" in data:
                    break

            sock.close()
            return json.loads(data.decode().strip())
        except (ConnectionError, OSError, json.JSONDecodeError, TimeoutError) as e:
            logger.debug(f"Daemon IPC error: {e}")
            return None

    def get_insights(self) -> list[DaemonInsight]:
        """Get pending insights from the daemon."""
        response = self._send_command("get_insights")
        if not response:
            return []

        return [
            DaemonInsight(
                message=i["message"],
                severity=i.get("severity", "info"),
                source=i.get("source", "daemon"),
                related_files=i.get("related_files"),
            )
            for i in response.get("insights", [])
        ]

    def get_status(self) -> dict | None:
        """Get daemon status."""
        return self._send_command("status")

    def stop(self) -> bool:
        """Ask daemon to stop."""
        response = self._send_command("stop")
        return response is not None

    @staticmethod
    def is_running() -> bool:
        return is_daemon_running()
