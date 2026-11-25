import json
import py_compile
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def check_python_syntax() -> None:
    py_files = (
        [ROOT_DIR / "launch.py"]
        + sorted((ROOT_DIR / "src").glob("*.py"))
        + sorted((ROOT_DIR / "scripts").glob("*.py"))
    )
    for pf in py_files:
        py_compile.compile(str(pf), doraise=True)
    print(f"[PASS] Python syntax check ({len(py_files)} files compiled cleanly)")


def check_required_files() -> None:
    expected = [
        "launch.py",
        "HueFit.bat",
        "requirements.txt",
        "requirements-train.txt",
        "README.md",
        "ARCHITECTURE.md",
        ".gitignore",
        "LICENSE",
        "static/index.html",
        "static/styles.css",
        "static/huefit.ico",
        "artifacts/.gitkeep",
        "artifacts/huefit_v2.pt",
        "artifacts/catalog.json",
        "artifacts/catalog_tensors.npz",
        "artifacts/catalog_item_embeddings.npy",
        "dist/huefit-runtime-artifacts.zip",
        "docs/figures/01_architecture_pipeline.png",
        "docs/figures/02_training_dynamics.png",
        "docs/figures/03_roc_pr_calibration.png",
        "docs/figures/04_model_comparison_fitb.png",
        "docs/figures/05_confusion_and_sensitivity.png",
    ]
    missing = [rel for rel in expected if not (ROOT_DIR / rel).exists()]
    if missing:
        raise AssertionError(f"Missing expected files: {missing}")

    html_txt = (ROOT_DIR / "static" / "index.html").read_text(encoding="utf-8")
    css_txt = (ROOT_DIR / "static" / "styles.css").read_text(encoding="utf-8")
    assert (
        "/static/styles.css" in html_txt
    ), "index.html does not link /static/styles.css"
    assert (
        "http://" not in html_txt and "https://" not in html_txt
    ), "External CDN URL found in index.html"
    assert (
        len(css_txt) > 5000
    ), f"static/styles.css looks truncated ({len(css_txt)} bytes)"
    assert css_txt.count("{") == css_txt.count(
        "}"
    ), "Unbalanced braces in static/styles.css"
    print(
        f"[PASS] Project files & static asset integrity ({len(expected)} files verified)"
    )


def check_live_launcher_http() -> None:
    port = _free_port()
    base = f"http://127.0.0.1:{port}"
    proc = subprocess.Popen(
        [
            sys.executable,
            str(ROOT_DIR / "launch.py"),
            "--port",
            str(port),
            "--no-browser",
        ],
        cwd=str(ROOT_DIR),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        ready = False
        for _ in range(90):
            if proc.poll() is not None:
                out, err = proc.communicate()
                raise RuntimeError(
                    f"launch.py exited early (code {proc.returncode}):\nSTDOUT:\n{out}\nSTDERR:\n{err}"
                )
            try:
                with urllib.request.urlopen(f"{base}/api/health", timeout=1.0) as r:
                    if r.status == 200:
                        health = json.loads(r.read().decode("utf-8"))
                        ready = True
                        break
            except Exception:
                time.sleep(0.5)
        if not ready:
            raise TimeoutError("launch.py did not become healthy within 45 seconds.")

        with urllib.request.urlopen(f"{base}/", timeout=5.0) as r:
            body_html = r.read().decode("utf-8")
            assert (
                r.status == 200
                and "<title>HueFit" in body_html
                and "styles.css" in body_html
            )

        with urllib.request.urlopen(f"{base}/static/styles.css", timeout=5.0) as r:
            ctype = r.headers.get("Content-Type", "")
            body_css = r.read().decode("utf-8")
            assert (
                r.status == 200 and "text/css" in ctype.lower() and len(body_css) > 5000
            ), f"CSS load issue: status={r.status}, Content-Type={ctype!r}, len={len(body_css)}"

        with urllib.request.urlopen(f"{base}/favicon.ico", timeout=5.0) as r:
            assert r.status == 200 and len(r.read()) > 1000

        req = urllib.request.Request(
            f"{base}/api/recommend",
            data=json.dumps(
                {
                    "locks": {"top": {"hex": "#1D3557"}},
                    "style": "All",
                    "outerwear_mode": "any",
                }
            ).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=10.0) as r:
            rec = json.loads(r.read().decode("utf-8"))
            assert r.status == 200 and len(rec.get("options", [])) == 8

        print(
            f"[PASS] Live TCP launch.py test ({base}) -> HTML 200 | CSS 200 ({ctype}) | "
            f"ICO 200 | Recommend 8 looks ({rec['latency_ms']} ms) | Device={health['device']}"
        )
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            proc.kill()


def main() -> None:
    print("=== HueFit — Pre-GitHub Preflight Check ===")
    check_python_syntax()
    check_required_files()
    check_live_launcher_http()
    print("\nALL PREFLIGHT CHECKS PASSED — READY FOR GITHUB!")


if __name__ == "__main__":
    main()
