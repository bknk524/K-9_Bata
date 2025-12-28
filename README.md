<h1>説明</h1>
# Local Doc Search（Office/PDF/TXT 類似検索）  
ChromaDB + Sentence-Transformers（bge-m3）でローカル文書を索引化し、Streamlit UIから類似検索できるアプリです。  
**ローカル完結**で、指定フォルダ配下の **docx / pptx / xlsx / pdf / txt** をテキスト抽出 → チャンク化 → 埋め込み → 永続保存し、検索語の埋め込みで近いファイルを返します。

---

## 機能
- フォルダ配下のファイルを索引化（変更分のみ再索引化）
- 類似検索（ファイル単位でランキング、抜粋表示）
- GUIで設定変更（フォルダ / チャンク / topK / device など）→ 保存
- Windowsでフォルダ選択ダイアログ（Explorer）対応
- CPU / GPU(CUDA) / NPU（UIで選択、未対応ならCPUへフォールバック）

---

## 対応ファイル形式
- Word: `.docx`
- PowerPoint: `.pptx`
- Excel: `.xlsx`
- PDF: `.pdf`（テキスト抽出できるPDFのみ。スキャンPDFは別途OCRが必要）
- Text: `.txt`

---

## 推奨環境
- OS: Windows（ローカル実行前提）
- Python: 3.12.x
- 仮想環境: venv
- GPU: NVIDIA + CUDA（任意。TorchのCUDA版が必要）

---

## フォルダ構成（推奨）
```txt
local-doc-search/
├─ app/
│  ├─ __init__.py
│  ├─ settings.py        # 変更可能な設定 + JSON永続化
│  ├─ core.py            # 索引化/検索の根幹（UI非依存）
│  └─ ui_app.py          # Streamlit GUI
├─ data/
│  ├─ documents/         # 検索対象ファイルを入れる
│  ├─ chroma_store/      # ChromaDB 永続保存領域
│  └─ app_settings.json  # 設定保存（初回起動後に生成）
├─ requirements-base.txt # Torch以外（共通）
└─ README.md

```
<h1>初回起動手順</h1>

1.venvインストール
```bash
py -3.12 -m venv venv
```
2.venv有効化(windows)
```bash
venv\Scripts\activate
```
3.依存インストール（Torch以外）
```bash
pip install -r requirements.txt
```
4.Torch のインストール（CPU / GPU）
cpu
```bash
pip install "torch==2.9.1"
```
gpu現状確認できず選定必要
```bash
pip install "torch>=2.6.0"
```
6.起動
```bash
streamlit run .\app\ui_app.py
```

DB初期化
```bash
Remove-Item -Recurse -Force .\data\chroma_store
```

