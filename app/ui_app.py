import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # プロジェクト直下（K-9_Bata）
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.settings import AppSettings
from app.core import index_folder, search, resolve_device


import os
import tkinter as tk
from tkinter import filedialog

import streamlit as st


def pick_directory(title: str, initial_dir: str = "") -> str:
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    try:
        selected = filedialog.askdirectory(
            title=title,
            initialdir=initial_dir if initial_dir else None,
            mustexist=True
        )
    finally:
        root.destroy()
    return selected or ""


st.set_page_config(page_title="Local Doc Search", layout="wide")
st.title("📚 Local Doc Search（ローカル完結）")
st.caption("Office/PDF/TXTを索引化して類似検索（ChromaDB + bge-m3）")

# 設定ロード
settings = AppSettings.load()

# session_stateへ
if "settings" not in st.session_state:
    st.session_state["settings"] = settings


def s() -> AppSettings:
    return st.session_state["settings"]


with st.sidebar:
    st.header("設定（あとから変更可能）")

    # docs_dir（Explorer選択）
    c1, c2 = st.columns([3, 1])
    with c1:
        s().docs_dir = st.text_input("対象フォルダ", value=s().docs_dir)
    with c2:
        if st.button("参照…", key="pick_docs"):
            picked = pick_directory("対象フォルダを選択", s().docs_dir)
            if picked:
                s().docs_dir = os.path.abspath(picked)
                st.rerun()

    # chroma_dir（Explorer選択）
    d1, d2 = st.columns([3, 1])
    with d1:
        s().chroma_dir = st.text_input("Chroma保存先", value=s().chroma_dir)
    with d2:
        if st.button("参照…", key="pick_chroma"):
            picked = pick_directory("Chroma保存先を選択", s().chroma_dir)
            if picked:
                s().chroma_dir = os.path.abspath(picked)
                st.rerun()

    s().collection = st.text_input("Collection名", value=s().collection)
    s().model_name = st.text_input("埋め込みモデル", value=s().model_name)

    st.divider()
    st.subheader("性能設定")
    s().chunk_size = st.number_input("チャンクサイズ（文字数）", min_value=200, max_value=5000, value=int(s().chunk_size), step=50)
    s().chunk_overlap = st.number_input("チャンク重なり", min_value=0, max_value=2000, value=int(s().chunk_overlap), step=20)

    st.divider()
    st.subheader("検索設定")
    s().top_k_files = st.slider("返すファイル数", 1, 30, int(s().top_k_files))
    s().top_k_chunks = st.slider("内部で見るチャンク数", 3, 80, int(s().top_k_chunks))

    st.divider()
    st.subheader("デバイス")
    s().device = st.selectbox("実行デバイス", ["auto", "cpu", "cuda", "npu"], index=["auto","cpu","cuda","npu"].index(s().device if s().device else "auto"))

    device_resolved, note = resolve_device(s().device)
    st.caption(f"判定: **{device_resolved}**（{note}）")

    st.divider()
    if st.button("設定を保存", type="primary"):
        s().save()
        st.success("設定を保存いたしましたわ。")


col1, col2 = st.columns([1, 2], gap="large")

with col1:
    st.subheader("① インデックス作成")
    st.write("対象フォルダ内のファイルを索引化（ベクトル化）します。更新分は再登録します。")

    if st.button("インデックス作成 / 更新", type="primary"):
        with st.spinner("索引化中…"):
            indexed, skipped, chunks, note = index_folder(s())
        st.success(f"完了：索引化 {indexed} 件 / スキップ {skipped} 件 / 追加チャンク {chunks} 件（{note}）")

    st.divider()
    st.subheader("② 検索")
    q = st.text_input("検索ワード", value="")
    do = st.button("検索", disabled=not bool(q.strip()))

with col2:
    st.subheader("検索結果")
    if do:
        with st.spinner("検索中…"):
            results = search(s(), q.strip())

        if not results:
            st.info("該当なしでございます。")
        else:
            for fp, best, hits in results:
                with st.expander(f"📄 {os.path.basename(fp)}  |  score={best:.4f}", expanded=False):
                    st.write(f"**Path:** `{fp}`")
                    for h in sorted(hits, key=lambda x: x["score"], reverse=True)[:5]:
                        st.markdown(
                            f"- score={h['score']:.4f} / chunk={h['chunk_index']} / ext={h['file_ext']}\n\n"
                            f"  > {h['snippet']}"
                        )
