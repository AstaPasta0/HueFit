import math
from typing import List, Sequence, Tuple
import numpy as np


def hex_to_rgb(hex_str: str) -> Tuple[int, int, int]:
    h = hex_str.strip().lstrip("#")
    if len(h) != 6:
        raise ValueError(f"Expected 6-digit hex color, got '{hex_str}'")
    return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))


def rgb_to_hex(rgb: Sequence[float]) -> str:
    return "#{:02X}{:02X}{:02X}".format(
        int(np.clip(round(float(rgb[0])), 0, 255)),
        int(np.clip(round(float(rgb[1])), 0, 255)),
        int(np.clip(round(float(rgb[2])), 0, 255)),
    )


def _srgb_to_linear(c: np.ndarray) -> np.ndarray:
    return np.where(
        c <= 0.04045, c / 12.92, np.power(np.clip((c + 0.055) / 1.055, 0.0, None), 2.4)
    )


def _linear_to_srgb(c: np.ndarray) -> np.ndarray:
    c = np.clip(c, 0.0, 1.0)
    return np.where(c <= 0.0031308, 12.92 * c, 1.055 * np.power(c, 1.0 / 2.4) - 0.055)


def srgb_to_oklab(rgb: np.ndarray) -> np.ndarray:
    arr = np.asarray(rgb, dtype=np.float64) / 255.0
    lin = _srgb_to_linear(np.clip(arr, 0.0, 1.0))
    r, g, b = lin[..., 0], lin[..., 1], lin[..., 2]

    l = 0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b
    m = 0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b
    s = 0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b

    l_ = np.cbrt(np.clip(l, 0.0, None))
    m_ = np.cbrt(np.clip(m, 0.0, None))
    s_ = np.cbrt(np.clip(s, 0.0, None))

    L = 0.2104542553 * l_ + 0.7936177850 * m_ - 0.0040720468 * s_
    a = 1.9779984951 * l_ - 2.4285922050 * m_ + 0.4505937099 * s_
    b_ = 0.0259040371 * l_ + 0.7827717662 * m_ - 0.8086757660 * s_
    return np.stack([L, a, b_], axis=-1).astype(np.float32)


def oklab_to_srgb(oklab: np.ndarray) -> np.ndarray:
    arr = np.asarray(oklab, dtype=np.float64)
    L, a, b = arr[..., 0], arr[..., 1], arr[..., 2]

    l_ = L + 0.3963377774 * a + 0.2158037573 * b
    m_ = L - 0.1055613458 * a - 0.0638541728 * b
    s_ = L - 0.0894841775 * a - 1.2914855480 * b

    l = l_ * l_ * l_
    m = m_ * m_ * m_
    s = s_ * s_ * s_

    r = +4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s
    g = -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s
    b_ = -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s

    srgb = _linear_to_srgb(np.stack([r, g, b_], axis=-1))
    return np.clip(np.round(srgb * 255.0), 0, 255).astype(np.uint8)


def hex_to_oklab(hex_str: str) -> np.ndarray:
    return srgb_to_oklab(np.array(hex_to_rgb(hex_str), dtype=np.float32))


def oklab_to_hex(oklab: Sequence[float]) -> str:
    return rgb_to_hex(oklab_to_srgb(np.asarray(oklab, dtype=np.float32)))


def oklab_to_oklch(oklab: np.ndarray) -> np.ndarray:
    arr = np.asarray(oklab, dtype=np.float64)
    L = arr[..., 0]
    a = arr[..., 1]
    b = arr[..., 2]
    C = np.hypot(a, b)
    h = np.mod(np.degrees(np.arctan2(b, a)), 360.0)
    return np.stack([L, C, h], axis=-1).astype(np.float32)


def chroma_gated_hue(oklab: np.ndarray, eps: float = 0.02) -> np.ndarray:
    arr = np.asarray(oklab, dtype=np.float64)
    a, b = arr[..., 1], arr[..., 2]
    C = np.hypot(a, b)
    denom = C + eps
    safe_c = np.where(C > 1e-8, C, 1.0)
    gate = C / denom
    cos_h = np.where(C > 1e-8, gate * (a / safe_c), 0.0)
    sin_h = np.where(C > 1e-8, gate * (b / safe_c), 0.0)
    return np.stack([C, cos_h, sin_h], axis=-1).astype(np.float32)


def encode_garment_color(
    primary_oklab: Sequence[float],
    secondary_oklab: Sequence[float],
    primary_share: float,
) -> np.ndarray:
    p = np.asarray(primary_oklab, dtype=np.float32)
    s = np.asarray(secondary_oklab, dtype=np.float32)
    p_h = chroma_gated_hue(p)
    s_h = chroma_gated_hue(s)
    share = np.array([float(np.clip(primary_share, 0.0, 1.0))], dtype=np.float32)
    return np.concatenate([p, p_h, s, s_h, share], axis=0)


def srgb_to_cielab(rgb: np.ndarray) -> np.ndarray:
    arr = np.asarray(rgb, dtype=np.float64) / 255.0
    lin = _srgb_to_linear(np.clip(arr, 0.0, 1.0))
    r, g, b = lin[..., 0], lin[..., 1], lin[..., 2]

    x = (r * 0.4124564 + g * 0.3575761 + b * 0.1804375) / 0.95047
    y = (r * 0.2126729 + g * 0.7151522 + b * 0.0721750) / 1.00000
    z = (r * 0.0193339 + g * 0.1191920 + b * 0.9503041) / 1.08883

    eps = 216.0 / 24389.0
    kappa = 24389.0 / 27.0
    f = lambda t: np.where(
        t > eps, np.cbrt(np.clip(t, 0.0, None)), (kappa * t + 16.0) / 116.0
    )
    fx, fy, fz = f(x), f(y), f(z)

    L = 116.0 * fy - 16.0
    a = 500.0 * (fx - fy)
    b_ = 200.0 * (fy - fz)
    return np.stack([L, a, b_], axis=-1).astype(np.float32)


def ciede2000_delta_e(lab1: Sequence[float], lab2: Sequence[float]) -> float:
    L1, a1, b1 = float(lab1[0]), float(lab1[1]), float(lab1[2])
    L2, a2, b2 = float(lab2[0]), float(lab2[1]), float(lab2[2])

    C1 = math.hypot(a1, b1)
    C2 = math.hypot(a2, b2)
    C_bar = 0.5 * (C1 + C2)
    C_bar7 = C_bar**7
    G = 0.5 * (1.0 - math.sqrt(C_bar7 / (C_bar7 + 25.0**7)))

    a1_p = (1.0 + G) * a1
    a2_p = (1.0 + G) * a2
    C1_p = math.hypot(a1_p, b1)
    C2_p = math.hypot(a2_p, b2)

    def _hp(b_val: float, a_p: float) -> float:
        if b_val == 0.0 and a_p == 0.0:
            return 0.0
        return math.degrees(math.atan2(b_val, a_p)) % 360.0

    h1_p = _hp(b1, a1_p)
    h2_p = _hp(b2, a2_p)

    dL_p = L2 - L1
    dC_p = C2_p - C1_p

    if C1_p * C2_p == 0.0:
        dh_p = 0.0
    else:
        diff = h2_p - h1_p
        if abs(diff) <= 180.0:
            dh_p = diff
        elif diff > 180.0:
            dh_p = diff - 360.0
        else:
            dh_p = diff + 360.0

    dH_p = 2.0 * math.sqrt(C1_p * C2_p) * math.sin(math.radians(dh_p / 2.0))
    L_bar_p = 0.5 * (L1 + L2)
    C_bar_p = 0.5 * (C1_p + C2_p)

    if C1_p * C2_p == 0.0:
        h_bar_p = h1_p + h2_p
    else:
        if abs(h1_p - h2_p) <= 180.0:
            h_bar_p = 0.5 * (h1_p + h2_p)
        elif (h1_p + h2_p) < 360.0:
            h_bar_p = 0.5 * (h1_p + h2_p + 360.0)
        else:
            h_bar_p = 0.5 * (h1_p + h2_p - 360.0)

    T = (
        1.0
        - 0.17 * math.cos(math.radians(h_bar_p - 30.0))
        + 0.24 * math.cos(math.radians(2.0 * h_bar_p))
        + 0.32 * math.cos(math.radians(3.0 * h_bar_p + 6.0))
        - 0.20 * math.cos(math.radians(4.0 * h_bar_p - 63.0))
    )

    d_theta = 30.0 * math.exp(-(((h_bar_p - 275.0) / 25.0) ** 2))
    C_bar_p7 = C_bar_p**7
    R_C = 2.0 * math.sqrt(C_bar_p7 / (C_bar_p7 + 25.0**7))

    L_diff50_sq = (L_bar_p - 50.0) ** 2
    S_L = 1.0 + (0.015 * L_diff50_sq) / math.sqrt(20.0 + L_diff50_sq)
    S_C = 1.0 + 0.045 * C_bar_p
    S_H = 1.0 + 0.015 * C_bar_p * T
    R_T = -math.sin(math.radians(2.0 * d_theta)) * R_C

    term_L = dL_p / S_L
    term_C = dC_p / S_C
    term_H = dH_p / S_H
    val = term_L * term_L + term_C * term_C + term_H * term_H + R_T * term_C * term_H
    return float(math.sqrt(max(0.0, val)))


def ciede2000_hex(hex1: str, hex2: str) -> float:
    lab1 = srgb_to_cielab(np.array(hex_to_rgb(hex1), dtype=np.float32))
    lab2 = srgb_to_cielab(np.array(hex_to_rgb(hex2), dtype=np.float32))
    return ciede2000_delta_e(lab1, lab2)


def ciede2000_oklab(ok1: Sequence[float], ok2: Sequence[float]) -> float:
    rgb1 = oklab_to_srgb(np.asarray(ok1, dtype=np.float32))
    rgb2 = oklab_to_srgb(np.asarray(ok2, dtype=np.float32))
    return ciede2000_delta_e(srgb_to_cielab(rgb1), srgb_to_cielab(rgb2))


FASHION_COLOR_ANCHORS: List[Tuple[str, str]] = [
    ("Black", "#141418"),
    ("Off Black", "#232428"),
    ("Charcoal", "#323741"),
    ("Slate Grey", "#55606E"),
    ("Heather Grey", "#9CA3AF"),
    ("Light Grey", "#D1D5DB"),
    ("White", "#F9FAFB"),
    ("Ivory", "#F6F2E9"),
    ("Cream", "#EFE9D5"),
    ("Beige", "#D8C8B4"),
    ("Sand", "#E2D5BE"),
    ("Taupe", "#8D847B"),
    ("Camel", "#C19A6B"),
    ("Tan", "#C8A47B"),
    ("Cognac", "#9A4E24"),
    ("Brown", "#6B4423"),
    ("Espresso", "#3B2418"),
    ("Navy", "#1D3557"),
    ("Midnight Blue", "#1A2744"),
    ("Denim Blue", "#3B5998"),
    ("Cobalt Blue", "#1D4ED8"),
    ("Steel Blue", "#4A749C"),
    ("Sky Blue", "#67B7E8"),
    ("Baby Blue", "#A9D3F5"),
    ("Teal", "#0F766E"),
    ("Turquoise", "#26B5A8"),
    ("Mint", "#9ADAC9"),
    ("Forest Green", "#1E4231"),
    ("Pine Green", "#2B5B46"),
    ("Emerald", "#117A5B"),
    ("Olive", "#55642F"),
    ("Sage", "#8CA37A"),
    ("Khaki", "#A99F79"),
    ("Burgundy", "#6E2632"),
    ("Wine", "#7E2B3E"),
    ("Crimson", "#A61C24"),
    ("Red", "#CC292B"),
    ("Coral", "#E2725B"),
    ("Terracotta", "#C25934"),
    ("Rust", "#A84426"),
    ("Orange", "#DC7018"),
    ("Peach", "#EBB18D"),
    ("Mustard", "#C99A2E"),
    ("Gold", "#C5A059"),
    ("Yellow", "#E8BE2C"),
    ("Lemon", "#F5DF62"),
    ("Blush", "#E6B4B8"),
    ("Dusty Rose", "#C48494"),
    ("Pink", "#E3669B"),
    ("Magenta", "#BA226A"),
    ("Plum", "#672C57"),
    ("Purple", "#6D34B8"),
    ("Lavender", "#A393CC"),
    ("Lilac", "#C6B6EC"),
]

_ANCHOR_NAMES = [name for name, _ in FASHION_COLOR_ANCHORS]
_ANCHOR_OKLAB = srgb_to_oklab(
    np.array([hex_to_rgb(hx) for _, hx in FASHION_COLOR_ANCHORS], dtype=np.float32)
)


def describe_oklab_color(oklab: Sequence[float]) -> str:
    vec = np.asarray(oklab, dtype=np.float32)
    dists = np.linalg.norm(_ANCHOR_OKLAB - vec[None, :], axis=1)
    return _ANCHOR_NAMES[int(np.argmin(dists))]


def describe_hex_color(hex_str: str) -> str:
    return describe_oklab_color(hex_to_oklab(hex_str))


def classify_palette_harmony(oklabs: Sequence[Sequence[float]]) -> str:
    if len(oklabs) < 2:
        return "Single Garment"
    arr = np.asarray(oklabs, dtype=np.float32)
    lch = oklab_to_oklch(arr)
    chromas = lch[:, 1]
    lightness = lch[:, 0]
    hues = lch[:, 2]

    chromatic_idx = np.where(chromas >= 0.045)[0]
    n_chromatic = len(chromatic_idx)
    l_spread = float(lightness.max() - lightness.min())

    if n_chromatic == 0:
        return (
            "Classic Neutral Contrast" if l_spread >= 0.35 else "Tonal Neutral Minimal"
        )
    if n_chromatic == 1:
        return "Neutral-Anchored Accent"

    ch_hues = hues[chromatic_idx]
    hue_diffs = []
    for i in range(len(ch_hues)):
        for j in range(i + 1, len(ch_hues)):
            d = abs(float(ch_hues[i] - ch_hues[j]))
            hue_diffs.append(min(d, 360.0 - d))
    max_dh = max(hue_diffs)
    min_dh = min(hue_diffs)

    if max_dh <= 20.0:
        return "Monochromatic Tonal"
    if max_dh <= 55.0:
        return "Analogous Harmony"
    if any(145.0 <= d <= 180.0 for d in hue_diffs):
        return "Complementary Contrast"
    if len(ch_hues) >= 3 and min_dh >= 80.0 and max_dh <= 145.0:
        return "Triadic Balance"
    if n_chromatic < len(oklabs):
        return "Neutral-Balanced Contrast"
    return "Split-Complementary Statement"
