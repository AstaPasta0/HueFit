import json
import sys
import tarfile
import zipfile
from pathlib import Path
from typing import Dict, List, Set, Tuple

import pyarrow.parquet as pq
import torch
from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.utils import GatedRepoError, HfHubHTTPError

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.config import ARTIFACTS_DIR, DATA_DIR, POLYVORE_REPO_ID, ensure_dirs

ARCHIVE_EXTENSIONS = (".tar.gz", ".tgz", ".tar", ".zip", ".tar.bz2", ".7z")
REQUIRED_METADATA = ("polyvore_item_metadata.json",)
OPTIONAL_METADATA = ("categories.csv",)


def check_runtime_environment() -> Dict[str, object]:
    cuda_ok = torch.cuda.is_available()
    info: Dict[str, object] = {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "cuda_available": cuda_ok,
        "cuda_version": torch.version.cuda,
        "gpu_name": None,
        "compute_capability": None,
        "vram_gb": None,
    }
    if cuda_ok:
        props = torch.cuda.get_device_properties(0)
        info["gpu_name"] = props.name
        info["compute_capability"] = f"sm_{props.major}{props.minor}"
        info["vram_gb"] = round(props.total_memory / (1024**3), 2)
    return info


def discover_repo_files(repo_id: str) -> Dict[str, List[str]]:
    api = HfApi()
    try:
        all_files = api.list_repo_files(repo_id=repo_id, repo_type="dataset")
    except (GatedRepoError, HfHubHTTPError) as exc:
        raise RuntimeError(
            f"Unable to access gated Hugging Face dataset '{repo_id}'. "
            "Accept access on Hugging Face and run `hf auth login` (or set HF_TOKEN) first."
        ) from exc

    metadata_files = sorted(
        f for f in all_files if f in REQUIRED_METADATA or f in OPTIONAL_METADATA
    )
    disjoint_files = sorted(
        f
        for f in all_files
        if f.startswith("disjoint/") and (f.endswith(".json") or f.endswith(".txt"))
    )
    disjoint_parquets = sorted(
        f
        for f in all_files
        if f.lower().endswith(".parquet")
        and "disjoint/" in f.lower()
        and "nondisjoint/" not in f.lower()
    )
    all_parquets = sorted(f for f in all_files if f.lower().endswith(".parquet"))
    archive_files = sorted(
        f
        for f in all_files
        if any(f.lower().endswith(ext) for ext in ARCHIVE_EXTENSIONS)
    )
    loose_image_files = [
        f
        for f in all_files
        if f.lower().startswith("images/")
        and f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    known = (
        set(metadata_files)
        | set(disjoint_files)
        | set(all_parquets)
        | set(archive_files)
        | set(loose_image_files)
    )
    other_files = sorted(f for f in all_files if f not in known)

    return {
        "metadata": metadata_files,
        "disjoint": disjoint_files,
        "disjoint_parquets": disjoint_parquets,
        "all_parquets": all_parquets,
        "archives": archive_files,
        "loose_images": loose_image_files,
        "other": other_files,
    }


def download_repo_file(repo_id: str, filename: str, local_dir: Path) -> Path:
    path_str = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        repo_type="dataset",
        local_dir=str(local_dir),
    )
    return Path(path_str)


def verify_parquet_file(parquet_path: Path) -> Dict[str, object]:
    pf = pq.ParquetFile(parquet_path)
    meta = pf.metadata
    schema = pf.schema_arrow
    columns = {field.name: str(field.type) for field in schema}
    return {
        "format": "parquet",
        "size_mb": round(parquet_path.stat().st_size / (1024**2), 2),
        "num_rows": int(meta.num_rows),
        "num_row_groups": int(meta.num_row_groups),
        "columns": columns,
    }


def verify_archive_integrity(archive_path: Path) -> Dict[str, object]:
    size_mb = round(archive_path.stat().st_size / (1024**2), 2)
    lower = archive_path.name.lower()
    if lower.endswith(".zip"):
        with zipfile.ZipFile(archive_path, "r") as zf:
            names = zf.namelist()
            img_count = sum(
                1 for n in names if n.lower().endswith((".jpg", ".jpeg", ".png"))
            )
        return {"format": "zip", "size_mb": size_mb, "image_entries": img_count}
    if lower.endswith((".tar.gz", ".tgz")):
        with open(archive_path, "rb") as fh:
            magic = fh.read(2)
        if magic != b"\x1f\x8b":
            raise ValueError(f"Invalid gzip header in {archive_path}")
        return {
            "format": "tar.gz",
            "size_mb": size_mb,
            "image_entries": "streamed_on_preprocess",
        }
    if lower.endswith(".tar"):
        if not tarfile.is_tarfile(archive_path):
            raise ValueError(f"Invalid tar archive: {archive_path}")
        return {
            "format": "tar",
            "size_mb": size_mb,
            "image_entries": "streamed_on_preprocess",
        }
    return {"format": "unknown", "size_mb": size_mb, "image_entries": "unknown"}


def resolve_token(tok: str, set_index_to_item: Dict[str, str]) -> str:
    return set_index_to_item.get(tok, tok)


def parse_disjoint_splits(
    data_dir: Path,
    disjoint_rel_files: List[str],
    item_meta: Dict[str, dict],
) -> Tuple[Dict[str, object], Dict[str, List[str]]]:
    set_index_to_item: Dict[str, str] = {}
    split_item_ids: Dict[str, Set[str]] = {
        "train": set(),
        "valid": set(),
        "test": set(),
    }
    split_outfit_counts: Dict[str, int] = {}

    for split in ("train", "valid", "test"):
        split_path = data_dir / "disjoint" / f"{split}.json"
        if not split_path.exists():
            raise FileNotFoundError(f"Missing required split file: {split_path}")
        with open(split_path, "r", encoding="utf-8") as fh:
            outfits = json.load(fh)
        split_outfit_counts[split] = len(outfits)
        for outfit in outfits:
            sid = str(outfit["set_id"])
            for it in outfit.get("items", []):
                iid = str(it["item_id"])
                set_index_to_item[f"{sid}_{it['index']}"] = iid
                split_item_ids[split].add(iid)

    compat_stats: Dict[str, Dict[str, object]] = {}
    compat_item_ids: Dict[str, Set[str]] = {
        "train": set(),
        "valid": set(),
        "test": set(),
    }

    for split in ("train", "valid", "test"):
        cpath = data_dir / "disjoint" / f"compatibility_{split}.txt"
        if not cpath.exists():
            continue
        n_pos = 0
        n_neg = 0
        with open(cpath, "r", encoding="utf-8") as fh:
            for line in fh:
                parts = line.strip().split()
                if len(parts) < 3:
                    continue
                label = int(parts[0])
                if label == 1:
                    n_pos += 1
                else:
                    n_neg += 1
                for tok in parts[1:]:
                    iid = resolve_token(tok, set_index_to_item)
                    compat_item_ids[split].add(iid)
        total = n_pos + n_neg
        compat_stats[split] = {
            "total": total,
            "positive": n_pos,
            "negative": n_neg,
            "pos_pct": round((n_pos / total * 100.0) if total else 0.0, 2),
        }

    fitb_stats: Dict[str, Dict[str, int]] = {}
    fitb_item_ids: Dict[str, Set[str]] = {"train": set(), "valid": set(), "test": set()}
    for rel in disjoint_rel_files:
        name = Path(rel).name
        if not name.startswith("fill_in_blank_") or not name.endswith(".json"):
            continue
        split_key = name.replace("fill_in_blank_", "").replace(".json", "")
        fpath = data_dir / rel
        with open(fpath, "r", encoding="utf-8") as fh:
            questions = json.load(fh)
        q_items: Set[str] = set()
        for q in questions:
            for tok in q.get("question", []):
                if not tok or tok == "___":
                    continue
                iid = resolve_token(str(tok), set_index_to_item)
                q_items.add(iid)
            for tok in q.get("answers", []):
                if not tok:
                    continue
                iid = resolve_token(str(tok), set_index_to_item)
                q_items.add(iid)
        if split_key in fitb_item_ids:
            fitb_item_ids[split_key].update(q_items)
        else:
            fitb_item_ids["test"].update(q_items)
        fitb_stats[split_key] = {
            "questions": len(questions),
            "unique_items": len(q_items),
        }

    train_ids = (
        split_item_ids["train"] | compat_item_ids["train"] | fitb_item_ids["train"]
    )
    valid_ids = (
        split_item_ids["valid"] | compat_item_ids["valid"] | fitb_item_ids["valid"]
    )
    test_ids = split_item_ids["test"] | compat_item_ids["test"] | fitb_item_ids["test"]
    used_ids = train_ids | valid_ids | test_ids

    overlap_tv = len(train_ids & valid_ids)
    overlap_tt = len(train_ids & test_ids)
    overlap_vt = len(valid_ids & test_ids)

    in_meta = sum(1 for iid in used_ids if iid in item_meta)
    stats = {
        "total_metadata_items": len(item_meta),
        "used_item_ids_count": len(used_ids),
        "used_items_with_metadata": in_meta,
        "unused_metadata_items_skipped": max(0, len(item_meta) - in_meta),
        "split_outfits": split_outfit_counts,
        "split_unique_items": {
            "train": len(train_ids),
            "valid": len(valid_ids),
            "test": len(test_ids),
        },
        "disjoint_overlap_check": {
            "train_valid": overlap_tv,
            "train_test": overlap_tt,
            "valid_test": overlap_vt,
        },
        "compatibility": compat_stats,
        "fill_in_blank": fitb_stats,
    }
    id_lists = {
        "used_item_ids": sorted(used_ids),
        "train_item_ids": sorted(train_ids),
        "valid_item_ids": sorted(valid_ids),
        "test_item_ids": sorted(test_ids),
    }
    return stats, id_lists


def main() -> None:
    ensure_dirs()
    env_info = check_runtime_environment()

    print("=== HueFit v2 — Step 1: Environment & Dataset Setup ===")
    print(
        f"Python {env_info['python']} | PyTorch {env_info['torch']} | "
        f"CUDA: {env_info['cuda_available']} ({env_info['gpu_name']} / "
        f"{env_info['compute_capability']} / {env_info['vram_gb']} GB VRAM)"
    )

    print(f"\nInspecting Hugging Face dataset repo: {POLYVORE_REPO_ID} ...")
    repo_files = discover_repo_files(POLYVORE_REPO_ID)
    print(
        f"Discovered -> metadata: {len(repo_files['metadata'])} | "
        f"disjoint files: {len(repo_files['disjoint'])} | "
        f"disjoint parquets: {len(repo_files['disjoint_parquets'])} | "
        f"archives: {len(repo_files['archives'])} | "
        f"other: {len(repo_files['other'])}"
    )

    downloaded_metadata: List[str] = []
    for rel in repo_files["metadata"]:
        p = download_repo_file(POLYVORE_REPO_ID, rel, DATA_DIR)
        downloaded_metadata.append(rel)
        print(f"  [OK] {rel} ({p.stat().st_size / (1024 ** 2):.2f} MB)")

    downloaded_disjoint: List[str] = []
    for rel in repo_files["disjoint"]:
        p = download_repo_file(POLYVORE_REPO_ID, rel, DATA_DIR)
        downloaded_disjoint.append(rel)
        print(f"  [OK] {rel} ({p.stat().st_size / (1024 ** 2):.2f} MB)")

    meta_path = DATA_DIR / "polyvore_item_metadata.json"
    with open(meta_path, "r", encoding="utf-8") as fh:
        item_meta = json.load(fh)

    stats, id_lists = parse_disjoint_splits(DATA_DIR, downloaded_disjoint, item_meta)

    image_sources: List[Dict[str, object]] = []
    if repo_files["disjoint_parquets"]:
        print(
            "\nDownloading disjoint Parquet image shards (kept packed, zero loose files) ..."
        )
        for rel in repo_files["disjoint_parquets"]:
            pq_path = download_repo_file(POLYVORE_REPO_ID, rel, DATA_DIR)
            meta_pq = verify_parquet_file(pq_path)
            meta_pq["relative_path"] = str(pq_path.relative_to(ROOT_DIR)).replace(
                "\\", "/"
            )
            meta_pq["repo_filename"] = rel
            image_sources.append(meta_pq)
            print(
                f"  [OK] {rel} -> parquet ({meta_pq['size_mb']} MB, "
                f"{meta_pq['num_rows']:,} rows, {meta_pq['num_row_groups']} row groups)"
            )
            print(f"       columns: {meta_pq['columns']}")
    elif repo_files["archives"]:
        for rel in repo_files["archives"]:
            print(
                f"\nDownloading image archive (kept compressed, NOT unpacked): {rel} ..."
            )
            archive_path = download_repo_file(POLYVORE_REPO_ID, rel, DATA_DIR)
            meta_arch = verify_archive_integrity(archive_path)
            meta_arch["relative_path"] = str(
                archive_path.relative_to(ROOT_DIR)
            ).replace("\\", "/")
            meta_arch["repo_filename"] = rel
            image_sources.append(meta_arch)
            print(f"  [OK] {rel} -> {meta_arch['format']} ({meta_arch['size_mb']} MB)")
    else:
        raise RuntimeError(
            f"No disjoint Parquet shards or image archives found in '{POLYVORE_REPO_ID}'. "
            f"Discovered files: {repo_files}"
        )

    manifest = {
        "repo_id": POLYVORE_REPO_ID,
        "environment": env_info,
        "metadata_files": downloaded_metadata,
        "disjoint_files": downloaded_disjoint,
        "image_sources": image_sources,
        "image_archives": image_sources,
        "stats": stats,
        "train_item_ids": id_lists["train_item_ids"],
        "valid_item_ids": id_lists["valid_item_ids"],
        "test_item_ids": id_lists["test_item_ids"],
        "used_item_ids": id_lists["used_item_ids"],
    }

    manifest_path = ARTIFACTS_DIR / "disjoint_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh)

    print("\n=== Disjoint Split & Item Statistics ===")
    print(f"Total metadata items          : {stats['total_metadata_items']:,}")
    print(
        f"Disjoint used_item_ids        : {stats['used_item_ids_count']:,} (skipped {stats['unused_metadata_items_skipped']:,} unused)"
    )
    print(
        f"Metadata coverage (used IDs)  : {stats['used_items_with_metadata']:,}/{stats['used_item_ids_count']:,}"
    )
    print(
        f"Disjoint outfit counts        : train={stats['split_outfits']['train']:,} | "
        f"valid={stats['split_outfits']['valid']:,} | test={stats['split_outfits']['test']:,}"
    )
    print(
        f"Disjoint unique items         : train={stats['split_unique_items']['train']:,} | "
        f"valid={stats['split_unique_items']['valid']:,} | test={stats['split_unique_items']['test']:,}"
    )
    ov = stats["disjoint_overlap_check"]
    print(
        f"Disjoint item-leakage check   : train∩valid={ov['train_valid']} | "
        f"train∩test={ov['train_test']} | valid∩test={ov['valid_test']}"
    )
    for split, cs in stats["compatibility"].items():
        print(
            f"compatibility_{split:<5s}.txt       : {cs['total']:,} outfits "
            f"({cs['positive']:,} pos / {cs['negative']:,} neg, {cs['pos_pct']:.1f}% pos)"
        )
    for split, fs in stats["fill_in_blank"].items():
        print(
            f"fill_in_blank_{split:<5s}.json      : {fs['questions']:,} questions "
            f"({fs['unique_items']:,} unique items)"
        )
    print(f"\nSaved manifest -> {manifest_path}")


if __name__ == "__main__":
    main()
