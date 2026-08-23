"""Desktop launcher — runs the backend and shows a native window.

Packaged app: FastAPI serves the built frontend same-origin; the window loads
that URL. A free port is chosen to avoid conflicts, and we poll /api/health
before opening the window.

Dev with Vite HMR: set SUPERAGENT_UI_URL=http://localhost:5173 and run the Vite
dev server separately; the window will load that instead.
"""

from __future__ import annotations

import os
import socket
import threading
import time
import urllib.request

import uvicorn
import webview

from app.main import app

HOST = "127.0.0.1"


def _free_port() -> int:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind((HOST, 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


def _wait_healthy(port: int, timeout: float = 20.0) -> None:
    url = f"http://{HOST}:{port}/api/health"
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=1) as resp:
                if resp.status == 200:
                    return
        except Exception:
            time.sleep(0.2)


def main() -> None:
    port = int(os.getenv("BACKEND_PORT", "0")) or _free_port()

    def _run_backend() -> None:
        uvicorn.run(app, host=HOST, port=port, log_level="warning")

    threading.Thread(target=_run_backend, daemon=True).start()
    _wait_healthy(port)

    ui_url = os.getenv("SUPERAGENT_UI_URL") or f"http://{HOST}:{port}/"
    window = webview.create_window("SuperAgent", ui_url, width=1100, height=780)

    def _pick_folder():
        """Open a native folder dialog and return the chosen path (or None)."""
        result = window.create_file_dialog(webview.FOLDER_DIALOG)
        if not result:
            return None
        return result[0] if isinstance(result, (list, tuple)) else str(result)

    # Expose the picker to the /api/projects/pick-folder endpoint.
    app.state.pick_folder = _pick_folder
    webview.start()


if __name__ == "__main__":
    main()
