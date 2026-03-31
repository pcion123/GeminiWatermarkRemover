"""alpha_map.py — Alpha 遵罩輔助工具：尺寸判斷、澬e mask 提取。"""

import base64
import io

import numpy as np
from PIL import Image

# 從內嵌資料檔匯入 base64 字串
from _embedded_data import BG48_B64 as _BG48_PNG_B64, BG96_B64 as _BG96_PNG_B64
def get_logo_profile(width: int, height: int) -> dict:
    """
    依圖片尺寸返回 Gemini 水印的 logo 規格。
    寬高均 > 1024 使用 96×96 px，否則使用 48×48 px。
    """
    if width > 1024 and height > 1024:
        return {"logo_size": 96, "margin_right": 64, "margin_bottom": 64}
    else:
        return {"logo_size": 48, "margin_right": 32, "margin_bottom": 32}


def get_embedded_alpha_map(logo_size: int) -> np.ndarray:
    """
    從內嵌 PNG bytes 提取 Gemini 水印的逐像素 Alpha 強度 (0.0~1.0)。
    取每個像素 max(R,G,B) / 255，以白色前景亮度反推水印不透明度。
    """
    b64 = _BG48_PNG_B64 if logo_size == 48 else _BG96_PNG_B64
    raw = base64.b64decode(b64)
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    arr = np.array(img, dtype=np.float32)       # (H, W, 3)
    max_ch = np.max(arr[:, :, :3], axis=2)      # (H, W)
    return (max_ch / 255.0).flatten().astype(np.float32)


def load_alpha_map(path: str) -> np.ndarray:
    """
    從外部參考底圖（純色底 + Gemini 水印）提取逐像素 Alpha 強度 (0.0~1.0)。
    取每個像素 max(R,G,B) / 255，以白色前景亮度反推水印不透明度。
    """
    img = Image.open(path).convert("RGB")
    data = np.array(img, dtype=np.float32)          # shape: (H, W, 3)
    max_channel = np.max(data[:, :, :3], axis=2)    # shape: (H, W)
    alpha_map = (max_channel / 255.0).flatten().astype(np.float32)
    return alpha_map
