起動（venv運用）
cd local-doc-search
py -3.12 -m venv venv
venv\Scripts\activate
pip install -r requirements-base.txt


あとは Torchだけ、CPU版かCUDA版を選んで入れます（CVE対策で torch>=2.6 推奨）。


cpu版
pip install "torch>=2.6.0" --index-url https://download.pytorch.org/whl/cpu

GPU版(例：cu124)
pip install "torch>=2.6.0" --index-url https://download.pytorch.org/whl/cu124


