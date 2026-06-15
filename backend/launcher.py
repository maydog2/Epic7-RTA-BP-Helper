"""Launch the packaged E7 BP Helper app in a local browser."""

from __future__ import annotations

import json
import os
import socket
import sys
import threading
import time
import urllib.error
import urllib.request
import webbrowser

PREFERRED_PORT = 17890
STARTUP_TIMEOUT_SECONDS = 180.0


def configure_production_defaults() -> None:
    os.environ.setdefault("RECOMMENDER_RERANKER", "true")
    os.environ.setdefault("RECOMMENDER_DEBUG", "false")
    os.environ.setdefault("HOST", "127.0.0.1")


def find_available_port(preferred: int = PREFERRED_PORT) -> int:
    candidates = [preferred, *range(preferred + 1, preferred + 20)]
    for port in candidates:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind(("127.0.0.1", port))
            except OSError:
                continue
            return port
    raise RuntimeError("Could not find an available localhost port")


def wait_for_server(port: int, timeout_seconds: float = STARTUP_TIMEOUT_SECONDS) -> bool:
    status_url = f"http://127.0.0.1:{port}/api/status"
    home_url = f"http://127.0.0.1:{port}/"
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(status_url, timeout=2) as response:
                if response.status != 200:
                    raise urllib.error.URLError("bad status")
                payload = json.loads(response.read().decode("utf-8"))
                if payload.get("app") != "e7_bp_helper":
                    time.sleep(0.25)
                    continue
                if not payload.get("frontend_enabled"):
                    raise RuntimeError(
                        f"Frontend bundle missing. Expected index.html under packaged frontend/dist."
                    )
            with urllib.request.urlopen(home_url, timeout=2) as response:
                if response.status != 200:
                    raise urllib.error.URLError("home page unavailable")
                if b"E7 BP Helper" not in response.read():
                    raise urllib.error.URLError("unexpected home page")
            return True
        except (urllib.error.URLError, TimeoutError, ConnectionResetError, json.JSONDecodeError):
            time.sleep(0.25)
    return False


def start_server_thread(port: int) -> tuple[threading.Thread, list[BaseException]]:
    from .recommender_service import load_recommender, run_server

    startup_errors: list[BaseException] = []

    def _run() -> None:
        try:
            load_recommender()
            run_server(host="127.0.0.1", port=port)
        except BaseException as exc:  # noqa: BLE001
            startup_errors.append(exc)

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    return thread, startup_errors


def main() -> int:
    configure_production_defaults()
    from .recommender_service import frontend_enabled

    if not frontend_enabled():
        print("Frontend build is missing. Re-run .\\build_exe.ps1 before launching the packaged app.")
        return 1

    port = find_available_port(int(os.environ.get("PORT", PREFERRED_PORT)))
    os.environ["PORT"] = str(port)

    server_thread, startup_errors = start_server_thread(port)
    if not wait_for_server(port):
        if startup_errors:
            print(f"Failed to start E7 BP Helper backend: {startup_errors[0]}")
        else:
            print("Failed to start E7 BP Helper backend.")
            print(
                "If you still have an old dev server running on port 5000, close it and try again."
            )
        return 1

    app_url = f"http://127.0.0.1:{port}/"
    webbrowser.open(app_url)

    print("E7 BP Helper is running.")
    print(f"Open this address if your browser did not launch automatically: {app_url}")
    print("Close this window to stop the app.")

    try:
        while server_thread.is_alive():
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("Stopping E7 BP Helper...")
    return 0


if __name__ == "__main__":
    sys.exit(main())
