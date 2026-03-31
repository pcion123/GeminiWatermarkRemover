"""main.py — Gemini 浮水印移除工具 CLI 入口。

用法：
    python main.py <input> [--output <dir>] [--zip <file.zip>] [--ref <ref_dir>]
"""

import argparse
import sys
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

from alpha_map import get_embedded_alpha_map, get_logo_profile, load_alpha_map
from algorithm import (
    detect_watermark,
    get_region,
    remove_watermark,
    select_best_profile,
)

# ---------------------------------------------------------------------------
# 常數
# ---------------------------------------------------------------------------

SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}
MAX_FILE_BYTES = 20 * 1024 * 1024  # 20 MB


# ---------------------------------------------------------------------------
# 參考圖載入
# ---------------------------------------------------------------------------

def load_ref_alpha_maps(ref_dir: Path) -> dict:
    """
    從 ref/ 讀取 bg48.png / bg96.png。
    檔案不存在時 fallback 至內嵌 alpha map。
    """
    maps: dict = {}
    for size in (48, 96):
        p = ref_dir / f"bg{size}.png"
        if p.exists():
            maps[size] = load_alpha_map(str(p))
            print(f"  [ref] Loaded bg{size}.png  ({size}×{size})")
        else:
            maps[size] = get_embedded_alpha_map(size)
            print(f"  [ref] bg{size}.png not found → using embedded alpha map")
    return maps


# ---------------------------------------------------------------------------
# 單張圖片處理
# ---------------------------------------------------------------------------

def process_image(img_path: Path, alpha_maps: dict) -> tuple:
    """
    處理單張圖片：偵測水印 → 搜尋最佳參數 → 移除水印。
    返回 (PIL.Image | None, status: str)。
    """
    # --- 大小限制 ---
    if img_path.stat().st_size > MAX_FILE_BYTES:
        return None, "SKIP  (file > 20 MB)"

    # --- 載入圖片 ---
    try:
        img = Image.open(img_path).convert("RGB")
    except Exception as e:
        return None, f"ERROR (cannot open: {e})"

    pixels = np.array(img, dtype=np.uint8)
    img_h, img_w = pixels.shape[:2]

    # --- 選擇 profile ---
    profile = get_logo_profile(img_w, img_h)
    logo_size = profile["logo_size"]

    alpha_map = alpha_maps.get(logo_size)   # 必然有值（至少為均勻遮罩）

    # --- 計算區域 ---
    region = get_region(img_w, img_h, profile)

    # 邊界安全檢查
    if (
        region["x"] < 0
        or region["y"] < 0
        or region["x"] + region["w"] > img_w
        or region["y"] + region["h"] > img_h
    ):
        return None, "SKIP  (image too small for watermark region)"

    # --- 偵測 ---
    if not detect_watermark(pixels, alpha_map, region):
        return None, "SKIP  (no watermark detected)"

    # --- 搜尋最佳參數 ---
    best_scale, best_logo_map, best_logo_value = select_best_profile(
        pixels, alpha_map, region
    )

    # --- 移除 ---
    result = remove_watermark(
        pixels, alpha_map, region, best_scale, best_logo_map, best_logo_value
    )

    result_img = Image.fromarray(result, "RGB")
    return result_img, f"OK    (scale={best_scale:.1f})"


# ---------------------------------------------------------------------------
# 檔案收集
# ---------------------------------------------------------------------------

def collect_images(input_path: Path) -> list:
    if input_path.is_file():
        if input_path.suffix.lower() in SUPPORTED_EXTENSIONS:
            return [input_path]
        return []
    return sorted(
        p
        for p in input_path.rglob("*")
        if p.suffix.lower() in SUPPORTED_EXTENSIONS
    )


# ---------------------------------------------------------------------------
# CLI 入口
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Remove Gemini watermark from images (100%% local, math-only).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py water.png
  python main.py water.png --output result/
  python main.py images/   --output result/
  python main.py images/   --zip batch_result.zip
""",
    )
    parser.add_argument(
        "input",
        type=Path,
        help="Input image file or directory",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output directory (default: <input_parent>/output/)",
    )
    parser.add_argument(
        "--zip",
        type=Path,
        default=None,
        help="Pack all output files into a ZIP archive at this path",
    )
    parser.add_argument(
        "--ref",
        type=Path,
        default=Path(__file__).parent / "ref",
        help="Directory containing bg48.png / bg96.png  (default: ./ref/)",
    )
    args = parser.parse_args()

    # --- 輸入驗證 ---
    if not args.input.exists():
        print(f"[ERROR] Input path does not exist: {args.input}", file=sys.stderr)
        sys.exit(1)

    # --- 載入參考圖 ---
    print(f"\n=== Gemini Watermark Remover ===")
    print(f"Reference dir : {args.ref}")
    alpha_maps = load_ref_alpha_maps(args.ref)

    if not any((args.ref / f"bg{s}.png").exists() for s in (48, 96)):
        print(
            "[INFO] No reference images in ref/ → running in uniform-alpha fallback mode.\n"
            "       For better edge accuracy, place bg48.png / bg96.png in ref/.\n"
        )

    # --- 收集圖片 ---
    images = collect_images(args.input)
    if not images:
        print("[ERROR] No supported image files found.", file=sys.stderr)
        sys.exit(1)

    # --- 決定輸出目錄 ---
    if args.output:
        out_dir = args.output
    elif args.input.is_dir():
        out_dir = args.input / "output"
    else:
        out_dir = args.input.parent / "output"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Output dir    : {out_dir}")
    print(f"Images found  : {len(images)}\n")

    # --- 逐檔處理 ---
    saved_paths: list = []
    for img_path in images:
        print(f"  {img_path.name:<40} ", end="", flush=True)
        result_img, status = process_image(img_path, alpha_maps)
        print(status)

        if result_img is not None:
            out_name = img_path.stem + "_nowm.png"
            out_path = out_dir / out_name
            result_img.save(out_path, format="PNG")
            saved_paths.append(out_path)

    # --- 可選：打包 ZIP ---
    if args.zip is not None and saved_paths:
        zip_path = args.zip
        zip_path.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for p in saved_paths:
                zf.write(p, arcname=p.name)
        print(f"\nZIP created : {zip_path}  ({len(saved_paths)} files)")

    # --- 統計 ---
    processed = len(saved_paths)
    skipped = len(images) - processed
    print(f"\n{'='*34}")
    print(f"Done.  Processed: {processed}  /  Skipped: {skipped}")
    if saved_paths:
        print(f"Output dir  : {out_dir}")


if __name__ == "__main__":
    main()
