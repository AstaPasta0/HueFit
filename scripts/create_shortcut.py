import os
import subprocess
import sys
from pathlib import Path

from PIL import Image, ImageDraw

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.config import STATIC_DIR


def generate_huefit_ico(out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    size = 512
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    margin = 20
    draw.rounded_rectangle(
        [margin, margin, size - margin, size - margin],
        radius=104,
        fill=(245, 243, 238, 255),
        outline=(22, 22, 22, 255),
        width=14,
    )

    pad_x = 96
    gap = 22
    w_full = size - 2 * pad_x

    y1, h1 = 92, 118
    half_w = (w_full - gap) // 2
    draw.rounded_rectangle(
        [pad_x, y1, pad_x + half_w, y1 + h1], radius=24, fill=(29, 53, 87, 255)
    )
    draw.rounded_rectangle(
        [pad_x + half_w + gap, y1, pad_x + w_full, y1 + h1],
        radius=24,
        fill=(193, 154, 107, 255),
    )

    y2, h2 = y1 + h1 + gap, 118
    draw.rounded_rectangle(
        [pad_x, y2, pad_x + w_full, y2 + h2],
        radius=24,
        fill=(216, 200, 180, 255),
        outline=(180, 165, 146, 255),
        width=4,
    )

    y3, h3 = y2 + h2 + gap, 68
    draw.rounded_rectangle(
        [pad_x, y3, pad_x + w_full, y3 + h3], radius=22, fill=(28, 28, 32, 255)
    )

    ico_sizes = [
        (16, 16),
        (24, 24),
        (32, 32),
        (48, 48),
        (64, 64),
        (128, 128),
        (256, 256),
    ]
    img.save(out_path, format="ICO", sizes=ico_sizes)
    return out_path


def create_windows_desktop_shortcut(target_bat: Path, icon_path: Path) -> Path:
    if sys.platform != "win32":
        raise RuntimeError(
            "Desktop .lnk shortcut creation is only supported on Windows."
        )

    ps_script = r"""
$WshShell = New-Object -ComObject WScript.Shell
$Desktop = [Environment]::GetFolderPath('Desktop')
$LinkPath = Join-Path $Desktop 'HueFit.lnk'
$Shortcut = $WshShell.CreateShortcut($LinkPath)
$Shortcut.TargetPath = $env:HUEFIT_TARGET_BAT
$Shortcut.WorkingDirectory = $env:HUEFIT_WORKDIR
$Shortcut.IconLocation = $env:HUEFIT_ICON_PATH
$Shortcut.Description = 'HueFit — Outfit Color Recommender'
$Shortcut.Save()
Write-Output $LinkPath
"""
    env = os.environ.copy()
    env["HUEFIT_TARGET_BAT"] = str(target_bat.resolve())
    env["HUEFIT_WORKDIR"] = str(target_bat.resolve().parent)
    env["HUEFIT_ICON_PATH"] = str(icon_path.resolve())

    proc = subprocess.run(
        [
            "powershell",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-Command",
            ps_script,
        ],
        capture_output=True,
        text=True,
        env=env,
        check=True,
    )
    created = proc.stdout.strip().splitlines()[-1].strip()
    return Path(created)


def main() -> None:
    ico_path = STATIC_DIR / "huefit.ico"
    generate_huefit_ico(ico_path)
    print(f"[OK] Generated icon -> {ico_path}")

    bat_path = ROOT_DIR / "HueFit.bat"
    if not bat_path.exists():
        print(
            f"[WARN] {bat_path.name} not found yet — save HueFit.bat in the root folder and re-run to create Desktop shortcut."
        )
        return

    if sys.platform == "win32":
        lnk = create_windows_desktop_shortcut(bat_path, ico_path)
        print(f"[OK] Created Desktop shortcut -> {lnk}")
    else:
        print("[INFO] Non-Windows OS detected; run `python launch.py` to launch.")


if __name__ == "__main__":
    main()
