import io
import sys
from pathlib import Path

from fastapi.testclient import TestClient
from PIL import Image, ImageDraw

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.api import app
from src.config import DATA_DIR


def _load_test_garment_bytes() -> tuple[bytes, str]:
    val_pq = DATA_DIR / "data" / "disjoint" / "validation.parquet"
    if val_pq.exists():
        try:
            import pyarrow.parquet as pq

            pf = pq.ParquetFile(val_pq)
            first_batch = next(
                pf.iter_batches(batch_size=1, columns=["item_id", "image"])
            )
            return (
                first_batch.column("image").to_pylist()[0]["bytes"],
                "polyvore_validation.parquet",
            )
        except Exception:
            pass

    img = Image.new("RGB", (384, 384), (248, 248, 248))
    draw = ImageDraw.Draw(img)
    draw.rectangle([96, 54, 288, 330], fill=(35, 35, 35))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=92)
    return buf.getvalue(), "synthetic_garment_jpeg"


def main() -> None:
    print("=== HueFit — End-to-End API, Recommender & Persistence Verification ===")
    with TestClient(app) as client:
        r_health = client.get("/api/health")
        assert r_health.status_code == 200, r_health.text
        h = r_health.json()
        print(
            f"[PASS] GET /api/health -> device={h['device']} | VRAM={h['vram_allocated_mb']} MB | "
            f"catalog={h['catalog_outfits']:,} | role_clf={h['role_classifier_loaded']} | saved={h['saved_looks_count']}"
        )

        r_ui = client.get("/")
        assert r_ui.status_code == 200 and "HueFit" in r_ui.text
        print("[PASS] GET / (UI HTML)                     -> HTTP 200")

        r_cw = client.post(
            "/api/recommend",
            json={
                "locks": {"top": {"hex": "#1D3557"}},
                "style": "Formal",
                "outerwear_mode": "without",
            },
        )
        assert r_cw.status_code == 200, r_cw.text
        cw = r_cw.json()
        assert cw["scoring_mode"] == "color_only"
        assert len(cw["options"]) == 8
        assert all(
            not o["has_outerwear"] and o["colors"]["outerwear"] is None
            for o in cw["options"]
        )
        assert "secondary_color_names" in cw["options"][0]
        o0 = cw["options"][0]
        print(
            f"[PASS] Color-Only (Navy Top, Formal, 3-pc) -> latency={cw['latency_ms']} ms | "
            f"Top1={o0['compat_score']}% | ΔE00={o0['delta_e2000']} | harmony='{o0['harmony_label']}'"
        )

        r_multi = client.post(
            "/api/recommend",
            json={
                "locks": {
                    "top": {"hex": "#1E4231"},
                    "footwear": {"hex": "#141418"},
                },
                "style": "All",
                "outerwear_mode": "with",
            },
        )
        assert r_multi.status_code == 200, r_multi.text
        ml = r_multi.json()
        assert all(
            o["has_outerwear"] and o["colors"]["outerwear"] is not None
            for o in ml["options"]
        )
        m0 = ml["options"][0]
        print(
            f"[PASS] Multi-Slot Pin (Top+Shoes, 4-pc)    -> latency={ml['latency_ms']} ms | "
            f"Top1={m0['compat_score']}% | outer={m0['colors']['outerwear']} ({m0['titles']['outerwear']})"
        )

        r_save = client.post("/api/saved-looks", json=m0)
        assert r_save.status_code == 200, r_save.text
        saved_entry = r_save.json()["saved"]
        sid = saved_entry["saved_id"]

        r_dup = client.post("/api/saved-looks", json=m0)
        assert r_dup.status_code == 200 and r_dup.json()["created"] is False

        r_list = client.get("/api/saved-looks")
        assert r_list.status_code == 200 and any(
            x["saved_id"] == sid for x in r_list.json()["looks"]
        )

        r_del = client.delete(f"/api/saved-looks/{sid}")
        assert r_del.status_code == 200 and r_del.json()["deleted"] is True
        print(
            f"[PASS] Saved Looks Disk Persistence        -> create/dedupe/list/delete OK (id={sid})"
        )

        img_bytes, img_src = _load_test_garment_bytes()
        r_up = client.post(
            "/api/analyze-image",
            files={"file": ("sample.jpg", img_bytes, "image/jpeg")},
        )
        assert r_up.status_code == 200, r_up.text
        up = r_up.json()
        uid = up["upload_id"]
        reg0 = up["regions"][0]
        print(
            f"[PASS] POST /api/analyze-image ({img_src}) -> {up['mode']}/{up['segmenter']} | "
            f"primary={reg0['hex']} ({reg0['color_name']}) | role={reg0['role']} | latency={up['latency_ms']} ms"
        )

        r_px = client.post(
            "/api/sample-pixel",
            json={
                "upload_id": uid,
                "x_pct": 50.0,
                "y_pct": 50.0,
                "target_role": "outerwear",
            },
        )
        assert r_px.status_code == 200, r_px.text
        px = r_px.json()
        assert (
            px["role"] == "outerwear"
        ), f"Expected outerwear override, got {px['role']}"
        print(
            f"[PASS] POST /api/sample-pixel (slot-sync)  -> {px['hex']} ({px['color_name']}) | "
            f"assigned_role={px['role']} | region={px['region_index']}"
        )

        r_mm = client.post(
            "/api/recommend",
            json={
                "upload_id": uid,
                "locks": {
                    reg0["role"]: {
                        "hex": reg0["hex"],
                        "secondary_hex": reg0["secondary_hex"],
                        "primary_share": reg0["primary_share"],
                        "region_index": reg0["region_index"],
                    }
                },
                "style": "All",
                "outerwear_mode": "any",
            },
        )
        assert r_mm.status_code == 200, r_mm.text
        mm = r_mm.json()
        assert mm["scoring_mode"] == "multimodal"
        mm0 = mm["options"][0]
        print(
            f"[PASS] Multimodal Recommend (Photo Mode)   -> latency={mm['latency_ms']} ms | "
            f"Top1={mm0['compat_score']}% | harmony='{mm0['harmony_label']}'"
        )

        r_bad = client.post(
            "/api/analyze-image",
            files={"file": ("bad.jpg", b"not-an-image", "image/jpeg")},
        )
        assert r_bad.status_code == 400
        r_bad_hex = client.post(
            "/api/recommend", json={"locks": {"top": {"hex": "ZZZZZZ"}}}
        )
        assert r_bad_hex.status_code == 400
        print("[PASS] Validation Guards (corrupt/bad hex) -> HTTP 400")

    print("\nALL END-TO-END CHECKS PASSED!")


if __name__ == "__main__":
    main()
