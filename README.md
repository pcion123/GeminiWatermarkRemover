# Gemini Watermark Remover

從 Google Gemini 生成的圖片中移除右下角半透明四角星浮水印，完全本機運算，無需網路。

## 原理

Gemini 圖片的浮水印合成公式為：

$$C = (1 - \alpha) \cdot B + \alpha \cdot L$$

其中 $C$ 為合成後像素、$B$ 為原始背景、$\alpha$ 為逐像素不透明度、$L$ 為水印顏色（白色）。

反向求解背景：

$$B = \frac{C - \alpha \cdot L}{1 - \alpha}$$

$\alpha$ 的逐像素分布從內嵌的參考圖提取（`_embedded_data.py`），根據圖片尺寸選用 48×48 或 96×96 的遮罩。

## 偵測邏輯

在水印 bounding box 內，以高 alpha（≥0.15）與低 alpha（≤0.02）像素的平均亮度計算比值：

$$\text{ratio} = \frac{Z - b}{\alpha_{\text{scale}} \times (255 - b)}$$

比值落在 `[0.4, 1.5]` 判定有水印；近乎純白區域直接跳過（$j \geq 0.98,\ k \geq 245$）。

## 精確像素處理

`alpha < 0.05` 的像素（水印邊框以外的背景）不會套用反向合成，避免整個 bounding box 產生矩形色塊。

## 參數搜尋

對 `alpha_scale ∈ {0.5, 0.6, ..., 1.5}` 逐一測試，選取使品質分數最低的組合。品質分數：

$$\text{score} = \sqrt{\sum_{ch} \left( \frac{\sum_i w_i \cdot z_{ch,i}}{n} \right)^2}$$

其中 $w$ 為 alpha_map 加權、$z$ 為 Z-score 標準化後的通道值，分數越低代表移除效果越好。

## 安裝

```bash
pip install -r requirements.txt
```

需要 Python 3.8+、Pillow ≥ 10.0.0、NumPy ≥ 1.24.0。

## 使用

```bash
# 單張圖片
python main.py water.png

# 指定輸出目錄
python main.py water.png --output result/

# 整個資料夾
python main.py images/ --output result/

# 批次處理並打包 ZIP
python main.py images/ --zip batch.zip
```

## 參數

| 參數 | 預設值 | 說明 |
|------|--------|------|
| `input` | — | 輸入圖片檔或資料夾 |
| `--output` | `<input_parent>/output/` | 輸出目錄 |
| `--zip` | — | 將所有輸出打包為 ZIP |
| `--ref` | `./ref/` | 放置 bg48.png / bg96.png 的目錄 |

支援格式：`.png` `.jpg` `.jpeg` `.webp`，單檔上限 20 MB。

## 參考圖（可選）

`ref/bg48.png` 和 `ref/bg96.png` 為帶有 Gemini 水印的純色底圖，提供更精確的 per-pixel alpha 遮罩。若不存在則自動使用內嵌版本。

## 專案結構

```
GeminiWatermarkRemover/
  main.py             CLI 入口、流程協調
  algorithm.py        核心演算法（偵測、移除、評分、參數搜尋）
  alpha_map.py        Alpha 遮罩工具（尺寸判斷、遮罩提取）
  _embedded_data.py   內嵌的水印參考圖 base64 資料
  README.md           專案說明文件
  requirements.txt    相依套件
  assets/             靜態資源目錄
  ref/                外部參考底圖目錄（可選）
  output/             處理結果輸出目錄
```
