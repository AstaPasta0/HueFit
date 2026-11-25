import hashlib
import json
import sys
import zipfile
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.config import (
    ARTIFACTS_DIR,
    DEFAULT_SEM_CAT_TO_IDX,
    DIST_DIR,
    OPTIONAL_RUNTIME_ARTIFACTS,
    REQUIRED_RUNTIME_ARTIFACTS,
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def build_release_bundle() -> Path:
    missing = [
        name
        for name in REQUIRED_RUNTIME_ARTIFACTS
        if not (ARTIFACTS_DIR / name).exists()
    ]
    if missing:
        raise FileNotFoundError(
            f"Missing required runtime artifacts in {ARTIFACTS_DIR}: {missing}. "
            "Run scripts/preprocess.py and scripts/train.py first."
        )

    DIST_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = DIST_DIR / "huefit-runtime-artifacts.zip"

    print("=== HueFit — Packaging GitHub Release Runtime Bundle ===")
    with zipfile.ZipFile(
        zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6
    ) as zf:
        for name in REQUIRED_RUNTIME_ARTIFACTS:
            src = ARTIFACTS_DIR / name
            zf.write(src, arcname=name)
            print(f"  [+] {name:<32s} {src.stat().st_size / (1024 ** 2):6.2f} MB")

        mappings_src = ARTIFACTS_DIR / "item_mappings.json"
        if mappings_src.exists():
            with open(mappings_src, "r", encoding="utf-8") as fh:
                raw_map = json.load(fh)
            slim_map = {
                "sem_cat_to_idx": raw_map.get("sem_cat_to_idx", DEFAULT_SEM_CAT_TO_IDX),
                "num_cats": raw_map.get("num_cats"),
                "num_sem_cats": raw_map.get("num_sem_cats"),
            }
        else:
            slim_map = {"sem_cat_to_idx": DEFAULT_SEM_CAT_TO_IDX}
        slim_bytes = json.dumps(slim_map, indent=2).encode("utf-8")
        zf.writestr("item_mappings.json", slim_bytes)
        print(
            f"  [+] {'item_mappings.json (slim)':<32s} {len(slim_bytes) / 1024:6.1f} KB"
        )

        for name in OPTIONAL_RUNTIME_ARTIFACTS:
            if name == "item_mappings.json":
                continue
            src = ARTIFACTS_DIR / name
            if src.exists():
                zf.write(src, arcname=name)
                print(f"  [+] {name:<32s} {src.stat().st_size / 1024:6.1f} KB")

    with zipfile.ZipFile(zip_path, "r") as zf:
        bad = zf.testzip()
        if bad is not None:
            raise RuntimeError(f"Corrupt entry detected in generated archive: {bad}")
        entries = zf.namelist()

    checksum = sha256_file(zip_path)
    sha_path = DIST_DIR / "huefit-runtime-artifacts.zip.sha256"
    sha_path.write_text(f"{checksum}  {zip_path.name}\n", encoding="utf-8")

    size_mb = zip_path.stat().st_size / (1024**2)
    print(f"\n[OK] Bundle ready : {zip_path} ({size_mb:.2f} MB, {len(entries)} files)")
    print(f"[OK] SHA-256      : {checksum}")
    print("\nNext step for GitHub Release:")
    print(
        "  1. Upload dist/huefit-runtime-artifacts.zip to a GitHub Release (e.g. tag v1.0.0)."
    )
    print(
        "  2. Paste its release download URL into DEFAULT_ARTIFACTS_URL in src/config.py."
    )
    return zip_path


if __name__ == "__main__":
    build_release_bundle()
