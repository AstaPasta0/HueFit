import argparse
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path
from typing import List

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.config import ARTIFACTS_DIR, DEFAULT_ARTIFACTS_URL, REQUIRED_RUNTIME_ARTIFACTS


def missing_runtime_artifacts() -> List[str]:
    return [
        name
        for name in REQUIRED_RUNTIME_ARTIFACTS
        if not (ARTIFACTS_DIR / name).exists()
    ]


def _safe_extract_zip(zip_path: Path, target_dir: Path) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    target_resolved = target_dir.resolve()
    with zipfile.ZipFile(zip_path, "r") as zf:
        for member in zf.infolist():
            if member.is_dir():
                continue
            dest = (target_resolved / Path(member.filename).name).resolve()
            if dest.parent != target_resolved:
                raise ValueError(f"Unsafe path in archive: {member.filename}")
            with zf.open(member, "r") as src_f, open(dest, "wb") as dst_f:
                dst_f.write(src_f.read())


def download_and_extract(url: str) -> None:
    if not url:
        raise RuntimeError(
            "No runtime artifact URL configured. Pass --url <release_zip_url> or set HUEFIT_ARTIFACTS_URL."
        )
    print(f"Downloading HueFit runtime bundle from:\n  {url}")
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_zip = Path(tmp_dir) / "huefit-runtime-artifacts.zip"

        def _progress(block_num: int, block_size: int, total_size: int) -> None:
            if total_size > 0:
                pct = min(100.0, (block_num * block_size / total_size) * 100.0)
                print(f"\r  Downloading... {pct:5.1f}%", end="", flush=True)

        urllib.request.urlretrieve(url, tmp_zip, reporthook=_progress)
        print("\r  Download complete. Extracting into artifacts/ ...   ")
        _safe_extract_zip(tmp_zip, ARTIFACTS_DIR)

    still_missing = missing_runtime_artifacts()
    if still_missing:
        raise RuntimeError(
            f"Downloaded bundle is missing required files: {still_missing}"
        )
    print("[OK] All runtime artifacts installed in artifacts/.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download prebuilt HueFit runtime artifacts."
    )
    parser.add_argument(
        "--url",
        type=str,
        default=DEFAULT_ARTIFACTS_URL,
        help="Direct URL to huefit-runtime-artifacts.zip",
    )
    parser.add_argument(
        "--if-missing",
        action="store_true",
        help="Exit cleanly if runtime artifacts already exist.",
    )
    args = parser.parse_args()

    missing = missing_runtime_artifacts()
    if not missing:
        print("[OK] Runtime artifacts already present in artifacts/.")
        return

    if not args.url:
        print(f"[INFO] Missing runtime artifacts: {missing}")
        print(
            "[INFO] Download huefit-runtime-artifacts.zip from the GitHub Releases page and extract it into artifacts/,"
        )
        print(
            "       or set HUEFIT_ARTIFACTS_URL and re-run `python scripts/download_artifacts.py`."
        )
        if args.if_missing:
            return
        sys.exit(1)

    download_and_extract(args.url)


if __name__ == "__main__":
    main()
