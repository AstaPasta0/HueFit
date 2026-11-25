import os
from pathlib import Path
import torch

ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT_DIR / "src"
SCRIPTS_DIR = ROOT_DIR / "scripts"
DATA_DIR = ROOT_DIR / "data" / "polyvore_raw"
ARTIFACTS_DIR = ROOT_DIR / "artifacts"
STATIC_DIR = ROOT_DIR / "static"
DIST_DIR = ROOT_DIR / "dist"

POLYVORE_REPO_ID = "mvasil/polyvore-outfits"

SEGFORMER_MODEL_ID = "mattmdjaga/segformer_b2_clothes"
BIREFNET_MODEL_ID = "ZhengPeng7/BiRefNet_lite"
FASHIONSIGLIP_MODEL_ID = "hf-hub:Marqo/marqo-fashionSigLIP"

ROLES = ("top", "bottom", "outerwear", "footwear")
MAX_OUTFIT_LEN = 12
VISUAL_DIM = 768
SEED = 42

DEFAULT_SEM_CAT_TO_IDX = {
    "other": 1,
    "tops": 2,
    "bottoms": 3,
    "outerwear": 4,
    "shoes": 5,
    "all-body": 6,
    "bags": 7,
    "accessories": 8,
    "jewellery": 9,
    "hats": 10,
    "sunglasses": 11,
    "scarves": 12,
}

REQUIRED_RUNTIME_ARTIFACTS = (
    "huefit_v2.pt",
    "catalog.json",
    "catalog_tensors.npz",
    "catalog_item_embeddings.npy",
)

OPTIONAL_RUNTIME_ARTIFACTS = (
    "item_mappings.json",
    "evaluation_report.json",
)

DEFAULT_ARTIFACTS_URL = os.environ.get(
    "HUEFIT_ARTIFACTS_URL",
    "https://github.com/AstaPasta0/HueFit/releases/latest/download/huefit-runtime-artifacts.zip",
).strip()


def ensure_dirs() -> None:
    for d in (SRC_DIR, SCRIPTS_DIR, DATA_DIR, ARTIFACTS_DIR, STATIC_DIR):
        d.mkdir(parents=True, exist_ok=True)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_amp_dtype(device: torch.device) -> torch.dtype:
    if device.type == "cuda":
        try:
            if torch.cuda.is_bf16_supported():
                return torch.bfloat16
        except Exception:
            pass
        return torch.float16
    return torch.float32
