import html
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st
import tkinter as tk
from tkinter import filedialog

from app.auto_index import AutoIndexManager
from app.embedding_model import DEFAULT_EMBEDDING_MODEL_DIR, load_embedding_model_name
from app.settings import AppSettings
from app.core import search, resolve_device


def pick_directory(title: str, initial_dir: str = "") -> str:
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    try:
        selected = filedialog.askdirectory(
            title=title,
            initialdir=initial_dir if initial_dir else None,
            mustexist=True,
        )
    finally:
        root.destroy()
    return selected or ""


def inject_styles() -> None:
    css = """
        <style>
        .stApp {
            background:
                radial-gradient(circle at top right, rgba(59, 130, 246, 0.18), transparent 28%),
                linear-gradient(180deg, #0b1220 0%, #121a29 100%);
            color: #dce7f5;
        }
        .block-container {
            max-width: 1180px;
            padding-top: 2rem;
            padding-bottom: 3rem;
        }
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0f172a 0%, #172033 100%);
            border-right: 1px solid rgba(255, 255, 255, 0.06);
        }
        [data-testid="stSidebar"] * {
            color: #edf3fb;
        }
        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] .stMarkdown,
        [data-testid="stSidebar"] .stCaption,
        [data-testid="stSidebar"] [data-testid="stExpander"] summary,
        [data-testid="stSidebar"] [data-testid="stWidgetLabel"] {
            color: #edf3fb !important;
        }
        [data-testid="stSidebar"] input,
        [data-testid="stSidebar"] textarea,
        [data-testid="stSidebar"] [data-baseweb="select"] > div {
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid rgba(255, 255, 255, 0.10);
        }
        [data-testid="stSidebar"] input,
        [data-testid="stSidebar"] textarea,
        [data-testid="stSidebar"] [data-baseweb="select"] input,
        [data-testid="stSidebar"] [data-baseweb="select"] span,
        [data-testid="stSidebar"] [data-baseweb="select"] div {
            color: #edf3fb !important;
            -webkit-text-fill-color: #edf3fb !important;
        }
        [data-testid="stSidebar"] input::placeholder,
        [data-testid="stSidebar"] textarea::placeholder {
            color: #a9bbcf !important;
            -webkit-text-fill-color: #a9bbcf !important;
        }
        .hero-panel {
            background: linear-gradient(135deg, rgba(15,23,42,0.92), rgba(25,35,58,0.88));
            border: 1px solid rgba(71, 85, 105, 0.55);
            border-radius: 24px;
            padding: 1.8rem 2rem;
            box-shadow: 0 20px 45px rgba(0, 0, 0, 0.22);
            margin-bottom: 1.2rem;
        }
        .hero-eyebrow {
            display: inline-block;
            font-size: 0.82rem;
            font-weight: 700;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            color: #93c5fd;
            background: rgba(59, 130, 246, 0.18);
            border-radius: 999px;
            padding: 0.32rem 0.7rem;
            margin-bottom: 0.9rem;
        }
        .hero-panel h1 {
            margin: 0;
            font-size: 2rem;
            line-height: 1.25;
            color: #f8fafc;
        }
        .hero-panel p {
            margin: 0.85rem 0 0;
            color: #c2d1e3;
            font-size: 1rem;
            line-height: 1.75;
        }
        .flow-card {
            background: rgba(15, 23, 42, 0.78);
            border: 1px solid rgba(71, 85, 105, 0.5);
            border-radius: 18px;
            padding: 1rem 1.05rem;
            min-height: 128px;
            box-shadow: 0 14px 35px rgba(0, 0, 0, 0.18);
            margin-bottom: 1rem;
        }
        .flow-step {
            width: 2rem;
            height: 2rem;
            border-radius: 999px;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            background: rgba(59, 130, 246, 0.20);
            color: #93c5fd;
            font-weight: 700;
            margin-bottom: 0.65rem;
        }
        .flow-card h3 {
            margin: 0;
            font-size: 1rem;
            color: #eff6ff;
        }
        .flow-card p {
            margin: 0.55rem 0 0;
            color: #b3c1d1;
            font-size: 0.93rem;
            line-height: 1.6;
        }
        div[data-testid="stMetric"] {
            background: rgba(15, 23, 42, 0.82);
            border: 1px solid rgba(71, 85, 105, 0.55);
            border-radius: 18px;
            padding: 0.8rem 1rem;
            box-shadow: 0 14px 35px rgba(0, 0, 0, 0.18);
        }
        div[data-testid="stMetricLabel"] p {
            color: #a9bbcf;
            font-weight: 600;
        }
        div[data-testid="stMetricValue"] {
            color: #f8fafc;
        }
        div.stButton > button {
            border-radius: 999px;
            border: none;
            padding: 0.72rem 1.2rem;
            font-weight: 700;
            box-shadow: 0 12px 28px rgba(37, 99, 235, 0.16);
        }
        div.stButton > button[kind="primary"] {
            background: linear-gradient(135deg, #2563eb, #1d4ed8);
            color: white;
        }
        div.stButton > button[kind="secondary"] {
            background: rgba(15, 23, 42, 0.9);
            color: #e2e8f0;
            border: 1px solid rgba(71, 85, 105, 0.7);
            box-shadow: none;
        }
        div[data-testid="stTextInput"] input,
        div[data-testid="stNumberInput"] input,
        div[data-baseweb="select"] > div {
            border-radius: 14px;
            border: 1px solid rgba(71, 85, 105, 0.7);
            background: rgba(15, 23, 42, 0.92);
        }
        div[data-testid="stForm"] {
            background: rgba(15, 23, 42, 0.84);
            border: 1px solid rgba(71, 85, 105, 0.55);
            border-radius: 22px;
            padding: 1.1rem 1.1rem 0.35rem;
            box-shadow: 0 16px 38px rgba(0, 0, 0, 0.18);
            margin-bottom: 1rem;
        }
        div[data-testid="stExpander"] {
            border: 1px solid rgba(71, 85, 105, 0.55);
            border-radius: 20px;
            background: rgba(15, 23, 42, 0.9);
            box-shadow: 0 16px 38px rgba(0, 0, 0, 0.16);
            overflow: hidden;
        }
        .section-title {
            margin: 1.2rem 0 0.6rem;
            color: #eff6ff;
            font-size: 1.15rem;
            font-weight: 700;
        }
        .soft-card {
            background: rgba(15, 23, 42, 0.84);
            border: 1px solid rgba(71, 85, 105, 0.55);
            border-radius: 18px;
            padding: 1rem 1.1rem;
            margin-bottom: 1rem;
        }
        .soft-card p {
            margin: 0.2rem 0;
            color: #bfd0e3;
            line-height: 1.6;
        }
        .hit-card {
            border: 1px solid rgba(71, 85, 105, 0.48);
            border-radius: 16px;
            background: rgba(30, 41, 59, 0.86);
            padding: 0.85rem 0.95rem;
            margin: 0.55rem 0;
        }
        .hit-meta {
            display: flex;
            justify-content: space-between;
            gap: 0.75rem;
            flex-wrap: wrap;
            margin-bottom: 0.45rem;
        }
        .hit-badge {
            display: inline-flex;
            align-items: center;
            border-radius: 999px;
            padding: 0.22rem 0.62rem;
            font-size: 0.78rem;
            font-weight: 700;
        }
        .hit-badge.name {
            background: rgba(59, 130, 246, 0.16);
            color: #93c5fd;
        }
        .hit-badge.content {
            background: rgba(45, 212, 191, 0.14);
            color: #5eead4;
        }
        .hit-score {
            color: #b3c1d1;
            font-size: 0.82rem;
            font-weight: 600;
        }
        .hit-snippet {
            color: #d8e2ef;
            line-height: 1.7;
            font-size: 0.94rem;
        }
        .empty-state {
            background: rgba(15, 23, 42, 0.8);
            border: 1px dashed rgba(100, 116, 139, 0.7);
            border-radius: 18px;
            padding: 1.4rem 1.2rem;
            color: #bfd0e3;
            line-height: 1.7;
        }
        </style>
        """
    st.markdown(css, unsafe_allow_html=True)


def format_timestamp(value: float | None) -> str:
    if value is None:
        return "-"
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(value))


def format_result_summary(result: tuple[int, int, int, int, str] | None) -> str:
    if result is None:
        return "まだ更新履歴がありません。"
    indexed, skipped, chunks, removed, note = result
    return (
        f"索引化 {indexed} 件 / スキップ {skipped} 件 / "
        f"追加チャンク {chunks} 件 / 削除 {removed} 件（{note}）"
    )


def open_file(path: str) -> None:
    os.startfile(path)


def reveal_file_in_explorer(path: str) -> None:
    target = path if os.path.isdir(path) else os.path.dirname(path)
    normalized = os.path.normpath(target)
    subprocess.Popen(["explorer.exe", normalized])


def render_hit(hit: dict) -> None:
    if hit["kind"] == "folder_name":
        source_label = "フォルダ名で一致"
        badge_class = "name"
    elif hit["kind"] == "file_name":
        source_label = "ファイル名で一致"
        badge_class = "name"
    else:
        source_label = "本文で一致"
        badge_class = "content"
    chunk_label = "-" if hit["chunk_index"] is None else str(hit["chunk_index"])
    snippet = html.escape(hit["snippet"]).replace("\n", "<br>")
    st.markdown(
        f"""
        <div class="hit-card">
            <div class="hit-meta">
                <span class="hit-badge {badge_class}">{source_label}</span>
                <span class="hit-score">score {hit["score"]:.4f} / chunk {chunk_label} / {hit["file_ext"]}</span>
            </div>
            <div class="hit-snippet">{snippet}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


st.set_page_config(
    page_title="Office Document Search",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

settings = AppSettings.load()

if "settings" not in st.session_state:
    st.session_state["settings"] = settings
if "startup_index_done" not in st.session_state:
    st.session_state["startup_index_done"] = False
if "startup_index_error" not in st.session_state:
    st.session_state["startup_index_error"] = None
if "ui_action_message" not in st.session_state:
    st.session_state["ui_action_message"] = None
if "ui_action_error" not in st.session_state:
    st.session_state["ui_action_error"] = None
if "search_query" not in st.session_state:
    st.session_state["search_query"] = ""
if "search_results" not in st.session_state:
    st.session_state["search_results"] = None
if "last_search_query" not in st.session_state:
    st.session_state["last_search_query"] = ""


def s() -> AppSettings:
    return st.session_state["settings"]


@st.cache_resource
def get_auto_index_manager() -> AutoIndexManager:
    return AutoIndexManager()


def run_startup_index(manager: AutoIndexManager) -> None:
    if st.session_state["startup_index_done"]:
        return
    try:
        with st.spinner("起動時に文書を確認しています…"):
            manager.run_now(s(), "startup")
        st.session_state["startup_index_error"] = None
    except Exception as exc:
        st.session_state["startup_index_error"] = str(exc)
    finally:
        st.session_state["startup_index_done"] = True


auto_index_manager = get_auto_index_manager()
run_startup_index(auto_index_manager)

with st.sidebar:
    st.markdown("## 利用設定")
    st.caption("通常は対象フォルダを確認すれば、そのまま使い始められます。")

    st.markdown("### 文書フォルダ")
    c1, c2 = st.columns([4, 1])
    with c1:
        s().docs_dir = st.text_input("検索対象フォルダ", value=s().docs_dir)
    with c2:
        if st.button("参照", key="pick_docs", use_container_width=True):
            picked = pick_directory("検索対象フォルダを選択", s().docs_dir)
            if picked:
                s().docs_dir = os.path.abspath(picked)
                st.rerun()
    if st.button("指定フォルダを開く", key="open_docs_dir", use_container_width=True):
        docs_dir = os.path.abspath(s().docs_dir)
        if os.path.isdir(docs_dir):
            try:
                reveal_file_in_explorer(docs_dir)
                st.session_state["ui_action_message"] = f"指定フォルダを開きました: {docs_dir}"
                st.session_state["ui_action_error"] = None
            except Exception as exc:
                st.session_state["ui_action_error"] = f"指定フォルダを開けませんでした: {exc}"
                st.session_state["ui_action_message"] = None
        else:
            st.session_state["ui_action_error"] = "指定フォルダが見つかりません。パスを確認してください。"
            st.session_state["ui_action_message"] = None

    d1, d2 = st.columns([4, 1])
    with d1:
        s().chroma_dir = st.text_input("インデックス保存先", value=s().chroma_dir)
    with d2:
        if st.button("参照", key="pick_chroma", use_container_width=True):
            picked = pick_directory("インデックス保存先を選択", s().chroma_dir)
            if picked:
                s().chroma_dir = os.path.abspath(picked)
                st.rerun()

    with st.expander("検索の件数", expanded=True):
        s().top_k_files = st.slider("表示するファイル数", 1, 30, int(s().top_k_files))
        s().top_k_chunks = st.slider("候補として確認する本文数", 3, 80, int(s().top_k_chunks))

    with st.expander("詳細設定", expanded=False):
        s().collection = st.text_input("Collection名", value=s().collection)
        s().chunk_size = st.number_input(
            "本文の分割サイズ",
            min_value=200,
            max_value=5000,
            value=int(s().chunk_size),
            step=50,
        )
        s().chunk_overlap = st.number_input(
            "分割の重なり",
            min_value=0,
            max_value=2000,
            value=int(s().chunk_overlap),
            step=20,
        )
        s().device = st.selectbox(
            "実行デバイス",
            ["auto", "cpu", "cuda", "npu"],
            index=["auto", "cpu", "cuda", "npu"].index(s().device if s().device else "auto"),
        )
        current_model = load_embedding_model_name()
        st.text_input("埋め込みモデルフォルダ", value=str(DEFAULT_EMBEDDING_MODEL_DIR), disabled=True)
        st.caption(f"現在の埋め込みモデル: `{current_model}`")

    if st.button("設定を保存", type="primary", use_container_width=True):
        s().save()
        st.success("設定を保存しました。必要に応じて右側の更新ボタンで反映してください。")

inject_styles()

auto_index_manager.configure(s())
auto_index_status = auto_index_manager.get_status()
device_resolved, device_note = resolve_device(s().device)
docs_dir_exists = os.path.isdir(os.path.abspath(s().docs_dir))
watch_state = "監視中" if auto_index_status.watching_path and docs_dir_exists else "対象フォルダ未確認"

st.markdown(
    """
    <div class="hero-panel">
        <div class="hero-eyebrow">Office Document Search</div>
        <h1>社内文書を、迷わず探せる検索画面</h1>
        <p>
            Word、Excel、PowerPoint、PDF、テキストをまとめて検索します。
            ファイル名と本文の両方を見に行くため、正式名称でも内容の言い回しでも見つけやすい設計です。
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

flow_cols = st.columns(3, gap="large")
flow_items = [
    ("1", "文書を入れる", "検索したい Office ファイルや PDF を、対象フォルダへ入れてください。"),
    ("2", "起動する", "起動時に自動で確認し、必要なファイルだけインデックスを更新します。"),
    ("3", "言葉で探す", "ファイル名でも本文でも一致する候補を、見やすく一覧で表示します。"),
]
for col, (step, title, body) in zip(flow_cols, flow_items):
    with col:
        st.markdown(
            f"""
            <div class="flow-card">
                <div class="flow-step">{step}</div>
                <h3>{title}</h3>
                <p>{body}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

status_cols = st.columns(4, gap="large")
status_cols[0].metric("監視状態", watch_state)
status_cols[1].metric("最終更新", format_timestamp(auto_index_status.last_completed_at))
status_cols[2].metric("表示件数", str(s().top_k_files))
status_cols[3].metric("実行デバイス", device_resolved)

status_left, status_right = st.columns([2.2, 1], gap="large")
with status_left:
    st.markdown("### 現在の状態")
    status_message = format_result_summary(auto_index_status.last_result)
    last_reason = auto_index_status.last_reason or "-"
    st.markdown(
        f"""
        <div class="soft-card">
            <p><strong>最終処理:</strong> {last_reason}</p>
            <p><strong>結果:</strong> {status_message}</p>
            <p><strong>対象フォルダ:</strong> {html.escape(os.path.abspath(s().docs_dir))}</p>
            <p><strong>インデックス保存先:</strong> {html.escape(os.path.abspath(s().chroma_dir))}</p>
            <p><strong>デバイス判定:</strong> {html.escape(device_note)}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if not docs_dir_exists:
        st.warning("検索対象フォルダが見つかりません。設定からフォルダを確認してください。")
    if st.session_state["startup_index_error"]:
        st.error(f"起動時の自動更新に失敗しました: {st.session_state['startup_index_error']}")
    if auto_index_status.last_error:
        st.error(f"自動更新でエラーが発生しました: {auto_index_status.last_error}")

with status_right:
    st.markdown("### 操作")
    manual_update = st.button("今すぐ更新", type="primary", use_container_width=True)
    refresh_status = st.button("画面を再表示", use_container_width=True)
    if refresh_status:
        st.rerun()
    if manual_update:
        with st.spinner("変更分を更新しています…"):
            indexed, skipped, chunks, removed, note = auto_index_manager.run_now(s(), "manual")
        st.success(
            f"更新完了：索引化 {indexed} 件 / スキップ {skipped} 件 / "
            f"追加チャンク {chunks} 件 / 削除 {removed} 件（{note}）"
        )
        auto_index_status = auto_index_manager.get_status()

if st.session_state["ui_action_message"]:
    st.success(st.session_state["ui_action_message"])
    st.session_state["ui_action_message"] = None
if st.session_state["ui_action_error"]:
    st.error(st.session_state["ui_action_error"])
    st.session_state["ui_action_error"] = None

st.markdown('<div class="section-title">文書を検索</div>', unsafe_allow_html=True)
with st.form("search_form", clear_on_submit=False):
    st.text_input(
        "探したい言葉を入力してください",
        key="search_query",
        placeholder="例: 予算計画、営業提案、リスク管理、AlphaZero",
        help="ファイル名と本文の両方を検索します。",
    )
    action_cols = st.columns([1, 1, 2], gap="small")
    search_submitted = action_cols[0].form_submit_button("検索する", type="primary", use_container_width=True)
    clear_submitted = action_cols[1].form_submit_button("入力をクリア", use_container_width=True)
    action_cols[2].markdown(
        "<div style='padding-top: 0.85rem; color: #64748b; font-size: 0.92rem;'>"
        "ファイル名の正式名称でも、本文中の言い回しでも探せます。"
        "</div>",
        unsafe_allow_html=True,
    )

if clear_submitted:
    st.session_state["search_query"] = ""
    st.session_state["search_results"] = None
    st.session_state["last_search_query"] = ""
    st.rerun()

if search_submitted:
    query = st.session_state["search_query"].strip()
    if not query:
        st.warning("検索キーワードを入力してください。")
    else:
        with st.spinner("文書を検索しています…"):
            st.session_state["search_results"] = search(s(), query)
            st.session_state["last_search_query"] = query

st.markdown('<div class="section-title">検索結果</div>', unsafe_allow_html=True)
results = st.session_state["search_results"]
last_query = st.session_state["last_search_query"]

if last_query:
    st.caption(f"検索語: 「{last_query}」")

if results is None:
    st.markdown(
        """
        <div class="empty-state">
            上の入力欄に探したい言葉を入れて検索してください。<br>
            文書の中身だけでなく、ファイル名からも候補を見つけます。
        </div>
        """,
        unsafe_allow_html=True,
    )
elif not results:
    st.info("一致する文書は見つかりませんでした。キーワードや言い回しを変えて再度お試しください。")
else:
    st.caption(f"{len(results)} 件の候補を表示しています。")
    for index, (fp, best, hits) in enumerate(results):
        file_name = os.path.basename(fp)
        is_folder_result = any(hit.get("entry_type") == "folder" for hit in hits)
        top_sources = " / ".join(
            sorted(
                {
                    "フォルダ名" if hit["kind"] == "folder_name"
                    else "ファイル名" if hit["kind"] == "file_name"
                    else "本文"
                    for hit in hits
                }
            )
        )
        result_label = "フォルダ" if is_folder_result else "ファイル"
        with st.expander(f"{result_label}: {file_name}  |  関連度 {best:.4f}  |  一致箇所: {top_sources}", expanded=False):
            st.write(f"**保存場所**: `{fp}`")
            action_cols = st.columns([1, 1, 3], gap="small")
            open_label = "フォルダを開く" if is_folder_result else "ファイルを開く"
            if action_cols[0].button(open_label, key=f"open_file_{index}", use_container_width=True):
                try:
                    open_file(fp)
                    st.session_state["ui_action_message"] = f"{result_label}を開きました: {file_name}"
                    st.session_state["ui_action_error"] = None
                except Exception as exc:
                    st.session_state["ui_action_error"] = f"{result_label}を開けませんでした: {exc}"
                    st.session_state["ui_action_message"] = None
                st.rerun()
            if action_cols[1].button("保存場所を開く", key=f"open_folder_{index}", use_container_width=True):
                try:
                    reveal_file_in_explorer(fp)
                    st.session_state["ui_action_message"] = f"保存場所を開きました: {file_name}"
                    st.session_state["ui_action_error"] = None
                except Exception as exc:
                    st.session_state["ui_action_error"] = f"保存場所を開けませんでした: {exc}"
                    st.session_state["ui_action_message"] = None
                st.rerun()
            for hit in sorted(hits, key=lambda item: item["score"], reverse=True)[:5]:
                render_hit(hit)
