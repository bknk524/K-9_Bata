# Intel AI Boost NPU setup

`Intel(R) Core(TM) Ultra 7 258V` の `Intel(R) AI Boost` を使って、このリポジトリを `DirectML` ではなく `ONNX Runtime + OpenVINOExecutionProvider` で動かすための手順です。

この構成では、`data/embedding_model/onnx/model.onnx` にある埋め込みモデルを、NPU を使う設定で起動できるようにします。

## 前提

- Windows PowerShell を使用します
- リポジトリのルートディレクトリでコマンドを実行します
- Python 3.12 系が利用可能であることを想定しています

最初にプロジェクトのルートへ移動します。

```powershell
cd C:\Users\kkaleido\Documents\K-9_Bata
```

PowerShell の実行ポリシーで `.ps1` 実行が止まる場合は、そのセッションだけ一時的に許可してから進めてください。

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

## 1. NPU 用 venv を作成する

通常は次のコマンドを実行します。

```powershell
.\scripts\setup_npu_openvino.ps1
```

このスクリプトでは次を自動で行います。

- `venv_npu` を作成
- `requirements/requirements-npu-openvino.txt` をインストール
- 競合する標準版 `onnxruntime` を削除
- `onnxruntime-openvino==1.24.1` を再インストール
- `numpy==1.26.4` `packaging==24.2` `protobuf==5.29.6` に合わせる
- `OpenVINOExecutionProvider` とアプリ側の `NPU` 解決可否を検証

`python` コマンドが見つからない、または使いたい Python を明示したい場合は、Python 実行ファイルを指定して実行します。

```powershell
.\scripts\setup_npu_openvino.ps1 -PythonExe "C:\Path\To\Python312\python.exe"
```

たとえば Windows の `py` ランチャーで Python 3.12 の場所を確認したい場合は次を使えます。

```powershell
py -3.12 -c "import sys; print(sys.executable)"
```

## 2. NPU を使う設定にする

アプリ設定ファイル `data/app_settings.json` の `device` を `npu` にするには、次を実行します。

```powershell
.\scripts\run_npu_openvino.ps1 -PrepareSettings
```

設定ファイルの内容を確認するコマンド例です。

```powershell
Get-Content .\data\app_settings.json
```

## 3. アプリを起動する

NPU 用 venv でそのまま起動する場合は次です。

```powershell
.\scripts\run_npu_openvino.ps1 -Launch
```

実際に内部で実行されるのは、次の Streamlit 起動コマンドです。

```powershell
.\venv_npu\Scripts\python.exe -m streamlit run .\app\ui_app.py
```

ブラウザで UI を開き、埋め込みモデルの表示が `NPU` 系になっていることを確認してください。

## 4. 動作確認コマンド

利用可能な ONNX Runtime Provider を確認します。

```powershell
.\venv_npu\Scripts\python.exe -c "import onnxruntime as ort; print(ort.get_available_providers())"
```

期待される出力例:

```text
['OpenVINOExecutionProvider', 'CPUExecutionProvider']
```

アプリ側の NPU 解決結果を直接確認したい場合は、次のコマンドも使えます。

```powershell
.\venv_npu\Scripts\python.exe -c "from app.core import resolve_embedder_spec; from app.embedding_model import load_embedding_model_name; model=load_embedding_model_name(); print(resolve_embedder_spec('npu', model))"
```

## 補足

- `requirements/requirements-npu-directml.txt` は使用しません。この PC の `Intel AI Boost` では OpenVINO 構成を前提にしています
- 既存テストは Windows 上のディレクトリアクセス制約で失敗することがあります。テスト成功は NPU 動作確認の必須条件にはしていません
