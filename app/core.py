from __future__ import annotations
import os
import re
import json
import hashlib
from pathlib import Path
from typing import Iterable, List, Dict, Tuple

import chromadb
from chromadb.config import Settings as ChromaSettings
from sentence_transformers import SentenceTransformer

from docx import Document as DocxDocument
from pptx import Presentation
import openpyxl
from pypdf import PdfReader

from app.embedding_model import load_embedding_model_name
from app.settings import AppSettings

SUPPORTED_EXTS = {".docx", ".pptx", ".xlsx", ".pdf", ".txt"}
FILE_NAME_COLLECTION_SUFFIX = "__file_names"
KIND_SCORE_WEIGHTS = {
    "content": 1.0,
    "file_name": 0.85,
    "folder_name": 0.55,
}
_chroma_client_cache: Dict[str, chromadb.PersistentClient] = {}
_collection_cache: Dict[Tuple[str, str], object] = {}


# ---------- util ----------
def file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def normalize_whitespace(text: str) -> str:
    text = text.replace("\u00a0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def normalized_extension(path: str) -> str:
    return Path(path).suffix.lower()


def is_supported_file(path: str) -> bool:
    return normalized_extension(path) in SUPPORTED_EXTS


def should_index_file_name(path: str) -> bool:
    return bool(Path(path).name) and not os.path.isdir(path)


def should_index_folder_name(path: str) -> bool:
    return os.path.isdir(path) and bool(Path(path).name)


def should_index_name(path: str) -> bool:
    return should_index_file_name(path) or should_index_folder_name(path)


def chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    text = normalize_whitespace(text)
    if not text:
        return []
    chunks: List[str] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(n, start + chunk_size)
        chunks.append(text[start:end])
        if end == n:
            break
        start = max(0, end - overlap)
    return chunks


def iter_name_indexable_paths(folder: str) -> Iterable[str]:
    folder = os.path.abspath(folder)
    for root, _, files in os.walk(folder):
        if should_index_folder_name(root) and os.path.abspath(root) != folder:
            yield root
        for fn in files:
            path = os.path.join(root, fn)
            if os.path.isfile(path) and should_index_file_name(path):
                yield path


def collect_indexable_paths(folder: str) -> List[str]:
    return sorted(os.path.abspath(path) for path in iter_name_indexable_paths(folder))


def filename_to_embedding_text(path: str) -> str:
    file_name = os.path.basename(path)
    stem = Path(file_name).stem
    ext = Path(file_name).suffix.lower().lstrip(".")
    normalized = re.sub(r"[_\-.]+", " ", stem)
    normalized = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", normalized)
    parts = [file_name, stem, normalized.strip()]
    if ext:
        parts.append(ext)
    unique_parts = []
    for part in parts:
        part = normalize_whitespace(part)
        if part and part not in unique_parts:
            unique_parts.append(part)
    return "\n".join(unique_parts)


def path_signature(path: str) -> str:
    if os.path.isdir(path):
        return hashlib.sha256(f"folder|{Path(path).name}".encode("utf-8")).hexdigest()
    if os.path.isfile(path) and is_supported_file(path):
        return file_sha256(path)
    return hashlib.sha256(f"file_name|{Path(path).name}".encode("utf-8")).hexdigest()


def entry_type(path: str) -> str:
    return "folder" if os.path.isdir(path) else "file"


def manifest_entry_is_complete(entry: Dict | None) -> bool:
    return bool(entry) and "entry_type" in entry and "content_indexed" in entry


def manifest_entry_has_size(entry: Dict | None) -> bool:
    return bool(entry) and "size" in entry


def stat_size(path: str) -> int | None:
    return os.path.getsize(path) if os.path.isfile(path) else None


def resolve_path_signature(path: str, prev: Dict | None, *, content_supported: bool, mtime: float, size: int | None) -> Tuple[str, bool]:
    if not prev:
        sha = path_signature(path)
        return sha, False

    if content_supported:
        if manifest_entry_has_size(prev) and prev.get("mtime") == mtime and prev.get("size") == size:
            return prev["sha256"], True
        sha = path_signature(path)
        return sha, prev.get("sha256") == sha

    sha = path_signature(path)
    return sha, prev.get("sha256") == sha


def weighted_similarity_score(kind: str, distance: float) -> float:
    raw_score = 1.0 - float(distance)
    return raw_score * KIND_SCORE_WEIGHTS.get(kind, 1.0)


# ---------- extractors ----------
def extract_text_docx(path: str) -> str:
    doc = DocxDocument(path)
    parts: List[str] = []
    for p in doc.paragraphs:
        t = p.text.strip()
        if t:
            parts.append(t)
    for table in doc.tables:
        for row in table.rows:
            cells = [c.text.strip() for c in row.cells]
            cells = [c for c in cells if c]
            if cells:
                parts.append(" | ".join(cells))
    return "\n".join(parts)


def extract_text_pptx(path: str) -> str:
    prs = Presentation(path)
    parts: List[str] = []
    for i, slide in enumerate(prs.slides, start=1):
        slide_texts: List[str] = []
        for shape in slide.shapes:
            if hasattr(shape, "text"):
                t = (shape.text or "").strip()
                if t:
                    slide_texts.append(t)
        if slide_texts:
            parts.append(f"[Slide {i}]\n" + "\n".join(slide_texts))
    return "\n\n".join(parts)


def extract_text_xlsx(path: str) -> str:
    wb = openpyxl.load_workbook(path, data_only=True)
    parts: List[str] = []
    for ws in wb.worksheets:
        for row in ws.iter_rows(values_only=True):
            vals = [str(v).strip() for v in row if v is not None and str(v).strip() != ""]
            if vals:
                parts.append(f"[Sheet: {ws.title}] " + " / ".join(vals))
    return "\n".join(parts)


def extract_text_pdf(path: str) -> str:
    reader = PdfReader(path)
    parts: List[str] = []
    for i, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""
        text = text.strip()
        if text:
            parts.append(f"[Page {i}]\n{text}")
    return "\n\n".join(parts)


def extract_text_txt(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def extract_text(path: str) -> str:
    ext = normalized_extension(path)
    if ext == ".docx":
        return extract_text_docx(path)
    if ext == ".pptx":
        return extract_text_pptx(path)
    if ext == ".xlsx":
        return extract_text_xlsx(path)
    if ext == ".pdf":
        return extract_text_pdf(path)
    if ext == ".txt":
        return extract_text_txt(path)
    return ""


# ---------- chroma ----------
def get_chroma_client(chroma_dir: str) -> chromadb.PersistentClient:
    chroma_dir = os.path.abspath(chroma_dir)
    client = _chroma_client_cache.get(chroma_dir)
    if client is None:
        client = chromadb.PersistentClient(
            path=chroma_dir,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        _chroma_client_cache[chroma_dir] = client
    return client


def get_collection(chroma_dir: str, collection: str):
    chroma_dir = os.path.abspath(chroma_dir)
    key = (chroma_dir, collection)
    col = _collection_cache.get(key)
    if col is None:
        client = get_chroma_client(chroma_dir)
        col = client.get_or_create_collection(name=collection)
        _collection_cache[key] = col
    return col


def file_name_collection_name(collection: str) -> str:
    return f"{collection}{FILE_NAME_COLLECTION_SUFFIX}"


def delete_old_chunks_for_file(col, file_path: str) -> None:
    # 同一file_pathの古いチャンクを消してから入れ直す
    try:
        col.delete(where={"file_path": file_path})
    except Exception:
        pass


def delete_old_chunks_for_paths(cols: Iterable, file_paths: Iterable[str]) -> None:
    for file_path in file_paths:
        for col in cols:
            delete_old_chunks_for_file(col, file_path)


def prune_stale_files(manifest: Dict[str, Dict], active_files: Iterable[str]) -> List[str]:
    active_set = set(active_files)
    stale_paths = [file_path for file_path in manifest.keys() if file_path not in active_set]

    for file_path in stale_paths:
        manifest.pop(file_path, None)

    return stale_paths


# ---------- embedding / device ----------
def resolve_device(device_setting: str) -> Tuple[str, str]:
    """
    returns: (device_for_sentence_transformers, note)
    - cpu / cuda は torch の世界
    - npu は環境差が大きいので、現状は「対応バックエンドが無ければcpuへ」。
    """
    device_setting = (device_setting or "auto").lower()

    try:
        import torch
        cuda_ok = torch.cuda.is_available()
    except Exception:
        cuda_ok = False

    if device_setting == "cpu":
        return "cpu", "CPU指定"
    if device_setting == "cuda":
        if cuda_ok:
            return "cuda", "CUDA(GPU)指定"
        return "cpu", "CUDA指定でしたが利用不可のためCPUへフォールバック"
    if device_setting == "npu":
        # ここは“拡張ポイント”
        # 将来: ONNXRuntime(OpenVINO/DirectML)等の埋め込みバックエンドを実装したら切替可能
        return "cpu", "NPU指定（現構成では未対応のためCPUへフォールバック）"
    # auto
    return ("cuda" if cuda_ok else "cpu"), "AUTO判定"


_embedder_cache: Dict[Tuple[str, str], SentenceTransformer] = {}


def get_embedder(model_name: str, device: str) -> SentenceTransformer:
    key = (model_name, device)
    if key not in _embedder_cache:
        _embedder_cache[key] = SentenceTransformer(model_name, device=device)
    return _embedder_cache[key]


# ---------- manifest ----------
def manifest_path(chroma_dir: str) -> Path:
    return Path(chroma_dir) / "manifest.json"


def load_manifest(chroma_dir: str) -> Dict[str, Dict]:
    p = manifest_path(chroma_dir)
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return {}


def save_manifest(chroma_dir: str, manifest: Dict[str, Dict]) -> None:
    p = manifest_path(chroma_dir)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


# ---------- main ops ----------
def index_folder(settings: AppSettings) -> Tuple[int, int, int, int, str]:
    docs_dir = os.path.abspath(settings.docs_dir)
    os.makedirs(settings.chroma_dir, exist_ok=True)

    col = get_collection(settings.chroma_dir, settings.collection)
    file_name_col = get_collection(settings.chroma_dir, file_name_collection_name(settings.collection))
    device, note = resolve_device(settings.device)
    model_name = load_embedding_model_name()
    model: SentenceTransformer | None = None

    manifest = load_manifest(settings.chroma_dir)
    current_paths = collect_indexable_paths(docs_dir)
    stale_paths = prune_stale_files(manifest, current_paths)
    delete_old_chunks_for_paths([col, file_name_col], stale_paths)
    removed_files = len(stale_paths)
    rebuild_file_name_index = file_name_col.count() != len(current_paths)

    to_add_ids: List[str] = []
    to_add_docs: List[str] = []
    to_add_metas: List[Dict] = []
    to_add_embs: List[List[float]] = []
    to_add_file_name_ids: List[str] = []
    to_add_file_name_docs: List[str] = []
    to_add_file_name_metas: List[Dict] = []
    to_add_file_name_embs: List[List[float]] = []

    indexed_paths = 0
    skipped_paths = 0

    for abs_path in current_paths:
        ext = os.path.splitext(abs_path)[1].lower()
        mtime = os.path.getmtime(abs_path)
        size = stat_size(abs_path)

        prev = manifest.get(abs_path)
        content_supported = os.path.isfile(abs_path) and is_supported_file(abs_path)
        sha, file_is_current = resolve_path_signature(
            abs_path,
            prev,
            content_supported=content_supported,
            mtime=mtime,
            size=size,
        )
        needs_manifest_refresh = (
            not manifest_entry_is_complete(prev)
            or (content_supported and not manifest_entry_has_size(prev))
        )
        if file_is_current and not rebuild_file_name_index and not needs_manifest_refresh:
            skipped_paths += 1
            continue

        chunks: List[str] = []
        if file_is_current:
            delete_old_chunks_for_paths([file_name_col], [abs_path])
        else:
            delete_old_chunks_for_paths([col, file_name_col], [abs_path])
            if content_supported:
                text = extract_text(abs_path)
                chunks = chunk_text(text, settings.chunk_size, settings.chunk_overlap)

        file_name_text = filename_to_embedding_text(abs_path)
        texts_to_embed = [file_name_text, *chunks]

        if model is None:
            model = get_embedder(model_name, device)

        embs = model.encode(
            texts_to_embed,
            normalize_embeddings=True,
            batch_size=32,
            show_progress_bar=False,
        )

        file_name_emb = embs[0]
        file_name_id = hashlib.sha256(f"{abs_path}|{sha}|file_name".encode("utf-8")).hexdigest()
        to_add_file_name_ids.append(file_name_id)
        to_add_file_name_docs.append(file_name_text)
        to_add_file_name_metas.append({
            "file_path": abs_path,
            "file_ext": ext,
            "file_sha256": sha,
            "kind": "folder_name" if os.path.isdir(abs_path) else "file_name",
            "entry_type": entry_type(abs_path),
            "mtime": mtime,
        })
        to_add_file_name_embs.append(file_name_emb.tolist())

        if content_supported and not file_is_current:
            for i, (chunk, emb) in enumerate(zip(chunks, embs[1:])):
                # ★ID衝突回避：file_path + sha + i をhash化
                cid = hashlib.sha256(f"{abs_path}|{sha}|{i}".encode("utf-8")).hexdigest()
                to_add_ids.append(cid)
                to_add_docs.append(chunk)
                to_add_metas.append({
                    "file_path": abs_path,
                    "file_ext": ext,
                    "file_sha256": sha,
                    "chunk_index": i,
                    "kind": "content",
                    "entry_type": "file",
                    "mtime": mtime,
                })
                to_add_embs.append(emb.tolist())

        manifest[abs_path] = {
            "sha256": sha,
            "mtime": mtime,
            "ext": ext,
            "chunks": len(chunks),
            "size": size,
            "content_indexed": content_supported,
            "entry_type": entry_type(abs_path),
        }

        indexed_paths += 1

    if to_add_ids:
        col.add(ids=to_add_ids, documents=to_add_docs, metadatas=to_add_metas, embeddings=to_add_embs)
    if to_add_file_name_ids:
        file_name_col.add(
            ids=to_add_file_name_ids,
            documents=to_add_file_name_docs,
            metadatas=to_add_file_name_metas,
            embeddings=to_add_file_name_embs,
        )

    save_manifest(settings.chroma_dir, manifest)
    return indexed_paths, skipped_paths, len(to_add_ids), removed_files, note


def query_collection(col, q_emb: List[float], n_results: int) -> Tuple[List[str], List[Dict], List[float]]:
    if n_results <= 0 or col.count() == 0:
        return [], [], []

    res = col.query(
        query_embeddings=[q_emb],
        n_results=n_results,
        include=["documents", "metadatas", "distances"],
    )
    return res["documents"][0], res["metadatas"][0], res["distances"][0]


def merge_hits(
    file_map: Dict[str, Dict],
    docs: List[str],
    metas: List[Dict],
    dists: List[float],
    *,
    default_kind: str,
) -> None:
    for doc, meta, dist in zip(docs, metas, dists):
        fp = meta["file_path"]
        hit_kind = meta.get("kind", default_kind)
        score = weighted_similarity_score(hit_kind, dist)
        entry = file_map.setdefault(fp, {"best_score": score, "hits": []})
        entry["best_score"] = max(entry["best_score"], score)
        snippet = doc
        if hit_kind == "content":
            snippet = (doc[:260] + "…") if len(doc) > 260 else doc
        entry["hits"].append({
            "score": score,
            "chunk_index": meta.get("chunk_index"),
            "file_ext": meta.get("file_ext", ""),
            "kind": hit_kind,
            "entry_type": meta.get("entry_type", "file"),
            "snippet": snippet,
        })


def search(settings: AppSettings, query: str) -> List[Tuple[str, float, List[Dict]]]:
    col = get_collection(settings.chroma_dir, settings.collection)
    file_name_col = get_collection(settings.chroma_dir, file_name_collection_name(settings.collection))
    device, note = resolve_device(settings.device)
    model_name = load_embedding_model_name()
    model = get_embedder(model_name, device)

    q_emb = model.encode([query], normalize_embeddings=True)[0].tolist()

    file_map: Dict[str, Dict] = {}
    docs, metas, dists = query_collection(col, q_emb, settings.top_k_chunks)
    merge_hits(file_map, docs, metas, dists, default_kind="content")
    file_name_docs, file_name_metas, file_name_dists = query_collection(
        file_name_col,
        q_emb,
        max(settings.top_k_files * 3, settings.top_k_chunks),
    )
    merge_hits(file_map, file_name_docs, file_name_metas, file_name_dists, default_kind="file_name")

    ranked = sorted(file_map.items(), key=lambda kv: kv[1]["best_score"], reverse=True)[: settings.top_k_files]
    # note はUI側で表示したい場合、settings.deviceから resolve_device して出せます
    return [(fp, data["best_score"], data["hits"]) for fp, data in ranked]
