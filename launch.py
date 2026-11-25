import argparse
import ctypes
import os
import socket
import sys
import threading
import time
import urllib.request
import webbrowser
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.config import ARTIFACTS_DIR, DEFAULT_ARTIFACTS_URL, REQUIRED_RUNTIME_ARTIFACTS


def _disable_windows_quickedit() -> None:
    if sys.platform != "win32":
        return
    try:
        kernel32 = ctypes.windll.kernel32
        handle = kernel32.GetStdHandle(-10)
        mode = ctypes.c_uint32()
        if kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
            new_mode = (mode.value & ~0x0040) | 0x0080
            kernel32.SetConsoleMode(handle, new_mode)
    except Exception:
        pass


def _check_huefit_health(host: str, port: int, timeout: float = 1.2) -> bool:
    url = f"http://{host}:{port}/api/health"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            if resp.status != 200:
                return False
            body = resp.read().decode("utf-8", errors="ignore")
            return "catalog_outfits" in body
    except Exception:
        return False


def _is_port_in_use(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.3)
        return s.connect_ex((host, port)) == 0


def _pick_port(host: str, preferred: int) -> tuple[int, bool]:
    if _is_port_in_use(host, preferred):
        if _check_huefit_health(host, preferred):
            return preferred, True
        for cand in range(preferred + 1, preferred + 21):
            if not _is_port_in_use(host, cand):
                return cand, False
        raise RuntimeError(f"Ports {preferred}-{preferred + 20} are all busy.")
    return preferred, False


def _ensure_runtime_artifacts() -> None:
    missing = [
        name
        for name in REQUIRED_RUNTIME_ARTIFACTS
        if not (ARTIFACTS_DIR / name).exists()
    ]
    if not missing:
        return
    print(f"[HueFit] Missing runtime artifacts: {', '.join(missing)}")
    if DEFAULT_ARTIFACTS_URL:
        from scripts.download_artifacts import download_and_extract

        download_and_extract(DEFAULT_ARTIFACTS_URL)
        return
    raise SystemExit(
        "\n[HueFit] Runtime model files are missing from artifacts/.\n"
        "Download 'huefit-runtime-artifacts.zip' from the GitHub Releases page and extract it into artifacts/,\n"
        "or run scripts/download_data.py -> scripts/preprocess.py -> scripts/train.py to build them locally.\n"
    )


def _open_browser_when_ready(host: str, port: int, stop_event: threading.Event) -> None:
    app_url = f"http://{host}:{port}"
    while not stop_event.is_set():
        if _check_huefit_health(host, port, timeout=1.0):
            print(f"\n  [OK] HueFit is live at {app_url}")
            print(
                "  Keep this window open (or minimized) while using HueFit.\n",
                flush=True,
            )
            webbrowser.open(app_url)
            return
        time.sleep(0.4)


def main() -> None:
    _disable_windows_quickedit()

    parser = argparse.ArgumentParser(description="HueFit Local Launcher")
    parser.add_argument(
        "--host", default="127.0.0.1", help="Bind host (default: 127.0.0.1)"
    )
    parser.add_argument(
        "--port", type=int, default=8000, help="Preferred port (default: 8000)"
    )
    parser.add_argument(
        "--no-browser", action="store_true", help="Start server without opening browser"
    )
    args = parser.parse_args()

    _ensure_runtime_artifacts()

    port, already_running = _pick_port(args.host, args.port)
    app_url = f"http://{args.host}:{port}"

    if already_running:
        print(f"[HueFit] Already running at {app_url} — opening browser...")
        if not args.no_browser:
            webbrowser.open(app_url)
        return

    print("============================================================")
    print("  HueFit - Outfit Color Recommender")
    print(f"  URL  : {app_url}")
    print("  Keep this window open (or minimized) while using HueFit.")
    print("  Close this window or press Ctrl+C to stop.")
    print("============================================================")
    print("  Loading local models (first run downloads vision weights)...", flush=True)

    stop_event = threading.Event()
    if not args.no_browser:
        opener = threading.Thread(
            target=_open_browser_when_ready,
            args=(args.host, port, stop_event),
            daemon=True,
        )
        opener.start()

    import uvicorn

    try:
        uvicorn.run(
            "src.api:app",
            host=args.host,
            port=port,
            log_level="warning",
        )
    except KeyboardInterrupt:
        pass
    finally:
        stop_event.set()


if __name__ == "__main__":
    main()
