<h1>説明</h1>
# Local Doc Search（Office/PDF/TXT 類似検索）  
ChromaDB + Sentence-Transformers（bge-m3）でローカル文書を索引化し、Streamlit UIから類似検索できるアプリです。  
**ローカル完結**で、指定フォルダ配下の **docx / pptx / xlsx / pdf / txt** をテキスト抽出 → チャンク化 → 埋め込み → 永続保存し、検索語に対して本文、ファイル名、フォルダ名の候補を返します。
埋め込みモデルは `data/embedding_model/` フォルダを使います。空の場合は `BAAI/bge-m3` をそのフォルダへ自動インストールして使います。

---

## 機能
- フォルダ配下のファイルを索引化（変更分のみ再索引化、削除・移動済みファイルの古い索引も掃除）
- 類似検索（本文チャンク、ファイル名、フォルダ名を別枠でベクトル化し、候補をランキング）
- GUIで設定変更（フォルダ / チャンク / topK / device など）→ 保存
- アプリ起動時に対象フォルダを自動でインデックス更新
- 起動後も対象フォルダを監視し、追加・更新・削除されたファイルだけ自動反映
- Windowsでフォルダ選択ダイアログ（Explorer）対応
- CPU / GPU(CUDA) / NPU（UIで選択、未対応ならCPUへフォールバック）

---

## 対応ファイル形式
- Word: `.docx`
- PowerPoint: `.pptx`
- Excel: `.xlsx`
- PDF: `.pdf`（テキスト抽出できるPDFのみ。スキャンPDFは別途OCRが必要）
- Text: `.txt`
- 上記以外の拡張子は本文ベクトル化しませんが、ファイル名だけをベクトル化して検索対象に含めます
- 対象フォルダ配下のサブフォルダも、フォルダ名をベクトル化して検索対象に含めます

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
│  ├─ embedding_model/   # ローカル埋め込みモデル配置先
│  └─ app_settings.json  # 設定保存（初回起動後に生成）
├─ requirements.txt      # 依存関係
└─ README.md

```
## ダウンロードから起動まで

1. プロジェクトを取得
ZIPでダウンロードして展開するか、Git が使える場合は clone します。
```bash
git clone <このリポジトリURL>
cd K-9_Bata
```

2. Python 3.12 を用意
Windows で `py -3.12` が使える状態にしてください。

3. 必要に応じて Visual Studio Build Tools を入れる
一部ライブラリの導入で必要になる場合があります。
[Visual Studio ダウンロードページ](https://visualstudio.microsoft.com/ja/downloads/)
インストール時は「C++ によるデスクトップ開発」を選択してください。

4. 仮想環境を作成
```bash
py -3.12 -m venv venv
```

5. 仮想環境を有効化
```bash
venv\Scripts\activate
```

6. 依存関係をインストール
```bash
pip install -r requirements.txt
```

7. 検索対象ファイルを配置
検索したい `docx / pptx / xlsx / pdf / txt` を `data/documents/` に入れます。

8. アプリを起動
```bash
streamlit run .\app\ui_app.py
```

9. 起動時の自動インデックス更新を待つ
ブラウザで開いた UI の起動直後に、設定されている対象フォルダを自動でインデックス更新します。
`data/embedding_model/` が空の場合は、このタイミングで `BAAI/bge-m3` を自動ダウンロードしてから索引化します。

10. 起動後の自動監視
アプリ起動中は対象フォルダを監視し、ファイルの追加・更新・削除を検知すると変更があった分だけ自動で再反映します。

11. 必要に応じて手動で再インデックス
設定を変更した後や、対象フォルダの内容をすぐ反映したい場合は UI の「インデックス作成 / 更新」を押します。

## 補助操作

### DB初期化
```bash
Remove-Item -Recurse -Force .\data\chroma_store
```

### 埋め込みモデルを手動で差し替える
`data/embedding_model/` に SentenceTransformer / Hugging Face 形式のローカルモデルを配置してください。
次回の索引化・検索からそのモデルを使います。

## 補足
- インデックスは `data/chroma_store` に保存されます
- `data/embedding_model/` にローカルモデルを置くとそのフォルダを使います
- `data/embedding_model/` が空の場合は `BAAI/bge-m3` をそのフォルダへ自動インストールします
- ワークスペースを別の絶対パスへ移動した場合でも、次回の「インデックス作成 / 更新」で古い索引を自動で掃除します

