"""Daemon server — background process that watches and learns.

Runs as a detached process with Unix socket IPC.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import signal
import sys
import time
from pathlib import Path

from coolcode.daemon.insights import Insight, InsightEngine
from coolcode.daemon.watchers import FileWatcher, GitWatcher, WatchEvent

logger = logging.getLogger(__name__)

PID_FILE = Path.home() / ".coolcode" / "daemon.pid"
SOCKET_PATH = Path.home() / ".coolcode" / "daemon.sock"
LOG_FILE = Path.home() / ".coolcode" / "daemon.log"


class DaemonServer:
    """Background daemon that watches the project and generates insights.

    Communicates with the interactive CLI via Unix domain socket.
    """

    def __init__(self, project_dir: str):
        self.project_dir = project_dir
        self.file_watcher = FileWatcher(project_dir)
        self.git_watcher = GitWatcher(project_dir)
        self.insight_engine = InsightEngine()
        self._pending_insights: list[Insight] = []
        self._running = False
        self._server: asyncio.AbstractServer | None = None

    async def start(self) -> None:
        """Start the daemon: watchers + IPC server."""
        self._running = True
        self._write_pid()

        logger.info(f"Daemon starting for: {self.project_dir}")

        # Start IPC server
        if SOCKET_PATH.exists():
            SOCKET_PATH.unlink()

        self._server = await asyncio.start_unix_server(
            self._handle_client, path=str(SOCKET_PATH)
        )
        logger.info(f"IPC socket: {SOCKET_PATH}")

        # Run watchers concurrently
        try:
            await asyncio.gather(
                self._run_file_watcher(),
                self._run_git_watcher(),
                self._server.serve_forever(),
            )
        except asyncio.CancelledError:
            pass
        finally:
            await self.stop()

    async def stop(self) -> None:
        """Stop the daemon gracefully."""
        self._running = False
        self.file_watcher.stop()
        self.git_watcher.stop()
        if self._server:
            self._server.close()
        self._remove_pid()
        if SOCKET_PATH.exists():
            SOCKET_PATH.unlink()
        logger.info("Daemon stopped")

    async def _run_file_watcher(self) -> None:
        """Run file watcher and feed events to insight engine."""
        async for event in self.file_watcher.watch():
            if not self._running:
                break
            insight = self.insight_engine.analyze(event)
            if insight:
                self._pending_insights.append(insight)
                logger.info(f"Insight: {insight.message}")

    async def _run_git_watcher(self) -> None:
        """Run git watcher and feed events to insight engine."""
        async for event in self.git_watcher.watch():
            if not self._running:
                break
            insight = self.insight_engine.analyze(event)
            if insight:
                self._pending_insights.append(insight)
                logger.info(f"Git insight: {insight.message}")

    async def _handle_client(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        """Handle a client connection on the Unix socket."""
        try:
            data = await asyncio.wait_for(reader.readline(), timeout=5.0)
            if not data:
                return

            request = json.loads(data.decode())
            cmd = request.get("cmd", "")

            if cmd == "get_insights":
                insights = list(self._pending_insights)
                self._pending_insights.clear()
                response = {
                    "insights": [
                        {
                            "message": i.message,
                            "severity": i.severity,
                            "source": i.source,
                            "related_files": i.related_files,
                        }
                        for i in insights
                    ]
                }

            elif cmd == "status":
                response = {
                    "running": True,
                    "project_dir": self.project_dir,
                    "uptime_s": time.time() - self._start_time,
                    "stats": self.insight_engine.stats,
                }

            elif cmd == "stop":
                response = {"status": "stopping"}
                writer.write(json.dumps(response).encode() + b"\n")
                await writer.drain()
                writer.close()
                asyncio.get_event_loop().call_soon(
                    lambda: asyncio.ensure_future(self.stop())
                )
                return

            else:
                response = {"error": f"Unknown command: {cmd}"}

            writer.write(json.dumps(response).encode() + b"\n")
            await writer.drain()
        except (asyncio.TimeoutError, json.JSONDecodeError, ConnectionError) as e:
            logger.warning(f"Client error: {e}")
        finally:
            writer.close()

    def _write_pid(self) -> None:
        PID_FILE.parent.mkdir(parents=True, exist_ok=True)
        PID_FILE.write_text(str(os.getpid()))
        self._start_time = time.time()

    def _remove_pid(self) -> None:
        if PID_FILE.exists():
            PID_FILE.unlink()


def daemonize(project_dir: str) -> None:
    """Fork into a background daemon process."""
    # Double fork to fully detach
    pid = os.fork()
    if pid > 0:
        # Parent exits
        print(f"Daemon started (PID: {pid})")
        return

    os.setsid()

    pid = os.fork()
    if pid > 0:
        os._exit(0)

    # Redirect stdout/stderr to log file
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    sys.stdout = open(LOG_FILE, "a")
    sys.stderr = sys.stdout

    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [daemon] %(levelname)s: %(message)s",
        stream=sys.stdout,
    )

    # Handle signals
    def handle_signal(signum, frame):
        logger.info(f"Received signal {signum}, stopping...")
        sys.exit(0)

    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)

    # Run the daemon
    server = DaemonServer(project_dir)
    try:
        asyncio.run(server.start())
    except Exception as e:
        logger.error(f"Daemon crashed: {e}")
    finally:
        if PID_FILE.exists():
            PID_FILE.unlink()
        if SOCKET_PATH.exists():
            SOCKET_PATH.unlink()


def is_daemon_running() -> bool:
    """Check if a daemon is currently running."""
    if not PID_FILE.exists():
        return False
    try:
        pid = int(PID_FILE.read_text().strip())
        os.kill(pid, 0)  # check if process exists
        return True
    except (ValueError, OSError):
        PID_FILE.unlink(missing_ok=True)
        return False


def get_daemon_pid() -> int | None:
    """Get the daemon's PID, or None if not running."""
    if not is_daemon_running():
        return None
    try:
        return int(PID_FILE.read_text().strip())
    except (ValueError, OSError):
        return None


def stop_daemon() -> bool:
    """Stop the running daemon."""
    pid = get_daemon_pid()
    if pid is None:
        return False
    try:
        os.kill(pid, signal.SIGTERM)
        # Wait for it to actually stop
        for _ in range(10):
            time.sleep(0.5)
            if not is_daemon_running():
                return True
        os.kill(pid, signal.SIGKILL)
        return True
    except OSError:
        PID_FILE.unlink(missing_ok=True)
        return False
