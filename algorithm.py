"""algorithm.py — Gemini 浮水印核心演算法。"""

from typing import Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# 區域計算
# ---------------------------------------------------------------------------

def get_region(img_w: int, img_h: int, profile: dict) -> dict:
    """
    計算水印的矩形區域（右下角固定位置）。

    Returns:
        dict with keys: x, y, w, h
    """
    logo_size = profile["logo_size"]
    mr = profile["margin_right"]
    mb = profile["margin_bottom"]
    return {
        "x": img_w - mr - logo_size,
        "y": img_h - mb - logo_size,
        "w": logo_size,
        "h": logo_size,
    }


# ---------------------------------------------------------------------------
# 水印偵測
# ---------------------------------------------------------------------------

def detect_watermark(
    pixels: np.ndarray,
    alpha_map: np.ndarray,
    region: dict,
    alpha_scale: float = 1.0,
) -> bool:
    """
    偵測指定區域是否存在 Gemini 半透明水印。

    以高 alpha (≥0.15) 和低 alpha (≤0.02) 像素的平均亮度計算比值：
        ratio = (Z - b) / (alpha_scale × (255 - b))
    比值落在 [0.4, 1.5] 視為有水印；近乎純白區域 (j≥0.98, k≥245) 直接返回 False。
    """
    x, y, w, h = region["x"], region["y"], region["w"], region["h"]

    patch = pixels[y: y + h, x: x + w, :].astype(np.float32)  # h×w×3
    flat = patch.reshape(-1, 3)                                 # (n, 3)
    alpha_flat = alpha_map                                      # length n

    brightness = flat.mean(axis=1)   # (n,) 亮度 = mean(R,G,B)
    k = brightness.mean()
    std = brightness.std()

    # 接近白色像素比例
    white_mask = (flat[:, 0] >= 250) & (flat[:, 1] >= 250) & (flat[:, 2] >= 250)
    j = white_mask.mean()

    # 排除近純白區域（本身就是白底，不是水印造成的）
    if j >= 0.98 and k >= 245 and std <= 5:
        return False

    # --- 標準模式：高/低 Alpha 分組對比 ---
    high_mask = alpha_flat >= 0.15
    low_mask = alpha_flat <= 0.02

    if high_mask.sum() == 0 or low_mask.sum() == 0:
        return False

    Z = brightness[high_mask].mean()  # 高 Alpha 區域平均亮度
    b = brightness[low_mask].mean()   # 低 Alpha 區域平均亮度

    D = alpha_scale * (255.0 - b)
    if abs(D) < 1e-8:
        return False

    X = Z - b
    ratio = X / D
    return 0.4 <= ratio <= 1.5


# ---------------------------------------------------------------------------
# 水印移除
# ---------------------------------------------------------------------------

def remove_watermark(
    pixels: np.ndarray,
    alpha_map: np.ndarray,
    region: dict,
    alpha_scale: float = 1.0,
    logo_map: Optional[np.ndarray] = None,
    logo_value: int = 255,
) -> np.ndarray:
    """
    反向 Alpha 合成還原背景像素：B = (C - α·L) / (1 - α)

    alpha < 0.05 的像素直接跳過，避免碰觸水印邊框外的背景。
    """
    x, y, w, h = region["x"], region["y"], region["w"], region["h"]

    result = pixels.astype(np.float32).copy()
    patch = result[y: y + h, x: x + w, :]    # h×w×3 view
    flat_patch = patch.reshape(-1, 3)         # (n, 3)

    alpha = (alpha_map * alpha_scale).clip(0.0, 0.99)   # (n,)
    skip_mask = alpha < 0.05                             # 僅處理水印像素，跳過背景
    inv_alpha = 1.0 - alpha                              # (n,)

    if logo_map is not None:
        logo = logo_map.astype(np.float32)               # (n, 3)
    else:
        logo = np.full_like(flat_patch, float(logo_value))

    alpha_col = alpha[:, np.newaxis]       # (n, 1)
    inv_col = inv_alpha[:, np.newaxis]     # (n, 1)

    original = (flat_patch - alpha_col * logo) / inv_col
    original = np.clip(np.round(original), 0, 255)

    # 僅替換非跳過的像素
    mask_col = (~skip_mask)[:, np.newaxis]
    flat_patch[:] = np.where(mask_col, original, flat_patch)

    patch[:] = flat_patch.reshape(h, w, 3)
    result[y: y + h, x: x + w, :] = patch

    return result.astype(np.uint8)


# ---------------------------------------------------------------------------
# 評分函式
# ---------------------------------------------------------------------------

def score_fn(
    pixels: np.ndarray,
    alpha_map: np.ndarray,
    region: dict,
) -> float:
    """
    移除品質評分：對每通道做 Z-score 標準化，計算與 alpha_map 的加權相關均值，
    三通道以均方根合成。分數越低代表移除效果越好。
    """
    x, y, w, h = region["x"], region["y"], region["w"], region["h"]
    patch = pixels[y: y + h, x: x + w, :].astype(np.float32)
    flat = patch.reshape(-1, 3)   # (n, 3)
    weights = alpha_map            # (n,)
    n = len(weights)

    total = 0.0
    for ch in range(3):
        channel = flat[:, ch]
        std = channel.std()
        if std < 1e-8:
            continue
        normed = (channel - channel.mean()) / std   # z-score
        weighted_sum = np.sum(weights * normed) / n
        total += weighted_sum ** 2

    return float(np.sqrt(total))


# ---------------------------------------------------------------------------
# 最佳 Profile 搜尋
# ---------------------------------------------------------------------------

_DEFAULT_ALPHA_SCALE_CANDIDATES = [
    0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5,
]


def select_best_profile(
    pixels: np.ndarray,
    alpha_map: np.ndarray,
    region: dict,
    logo_map: Optional[np.ndarray] = None,
    alpha_scale_candidates: Optional[list] = None,
) -> Tuple[float, Optional[np.ndarray], int]:
    """
    對每個候選 alpha_scale 分別測試純白 logo 與彩色 logo_map，
    選取 score_fn 最低的組合，返回 (best_alpha_scale, best_logo_map, best_logo_value)。
    """
    if alpha_scale_candidates is None:
        alpha_scale_candidates = _DEFAULT_ALPHA_SCALE_CANDIDATES

    best_score = float("inf")
    best_alpha_scale = 1.0
    best_logo_map: Optional[np.ndarray] = None
    best_logo_value = 255

    for scale in alpha_scale_candidates:
        # --- 方案 1：純白 Logo ---
        removed = remove_watermark(pixels, alpha_map, region, scale, None, 255)
        s = score_fn(removed, alpha_map, region)
        if s < best_score:
            best_score = s
            best_alpha_scale = scale
            best_logo_map = None
            best_logo_value = 255

        # --- 方案 2：彩色 Logo Map（若有提供）---
        if logo_map is not None:
            removed2 = remove_watermark(pixels, alpha_map, region, scale, logo_map, 255)
            s2 = score_fn(removed2, alpha_map, region)
            if s2 < best_score:
                best_score = s2
                best_alpha_scale = scale
                best_logo_map = logo_map
                best_logo_value = 255

    return best_alpha_scale, best_logo_map, best_logo_value
