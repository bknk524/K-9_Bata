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

from app.settings import AppSettings

SUPPORTED_EXTS = {".docx", ".pptx", ".xlsx", ".pdf", ".txt"}


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


def iter_files(folder: str) -> Iterable[str]:
    for root, _, files in os.walk(folder):
        for fn in files:
            ext = os.path.splitext(fn)[1].lower()
            if ext in SUPPORTED_EXTS:
                yield os.path.join(root, fn)


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
    ext = os.path.splitext(path)[1].lower()
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
def get_collection(chroma_dir: str, collection: str):
    client = chromadb.PersistentClient(
        path=chroma_dir,
        settings=ChromaSettings(anonymized_telemetry=False),
    )
    return client.get_or_create_collection(name=collection)


def delete_old_chunks_for_file(col, file_path: str) -> None:
    # 同一file_pathの古いチャンクを消してから入れ直す
    try:
        col.delete(where={"file_path": file_path})
    except Exception:
        pass


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
def index_folder(settings: AppSettings) -> Tuple[int, int, int, str]:
    docs_dir = os.path.abspath(settings.docs_dir)
    os.makedirs(settings.chroma_dir, exist_ok=True)

    col = get_collection(settings.chroma_dir, settings.collection)
    device, note = resolve_device(settings.device)
    model = get_embedder(settings.model_name, device)

    manifest = load_manifest(settings.chroma_dir)

    to_add_ids: List[str] = []
    to_add_docs: List[str] = []
    to_add_metas: List[Dict] = []
    to_add_embs: List[List[float]] = []

    indexed_files = 0
    skipped_files = 0

    for path in iter_files(docs_dir):
        abs_path = os.path.abspath(path)
        ext = os.path.splitext(abs_path)[1].lower()
        mtime = os.path.getmtime(abs_path)

        sha = file_sha256(abs_path)
        prev = manifest.get(abs_path)
        if prev and prev.get("sha256") == sha:
            skipped_files += 1
            continue

        delete_old_chunks_for_file(col, abs_path)

        text = extract_text(abs_path)
        chunks = chunk_text(text, settings.chunk_size, settings.chunk_overlap)
        if not chunks:
            manifest[abs_path] = {"sha256": sha, "mtime": mtime, "ext": ext, "chunks": 0}
            indexed_files += 1
            continue

        embs = model.encode(
            chunks,
            normalize_embeddings=True,
            batch_size=32,
            show_progress_bar=False,
        )

        for i, (chunk, emb) in enumerate(zip(chunks, embs)):
            # ★ID衝突回避：file_path + sha + i をhash化
            cid = hashlib.sha256(f"{abs_path}|{sha}|{i}".encode("utf-8")).hexdigest()
            to_add_ids.append(cid)
            to_add_docs.append(chunk)
            to_add_metas.append({
                "file_path": abs_path,
                "file_ext": ext,
                "file_sha256": sha,
                "chunk_index": i,
                "mtime": mtime,
            })
            to_add_embs.append(emb.tolist())

        manifest[abs_path] = {"sha256": sha, "mtime": mtime, "ext": ext, "chunks": len(chunks)}
        indexed_files += 1

    if to_add_ids:
        col.add(ids=to_add_ids, documents=to_add_docs, metadatas=to_add_metas, embeddings=to_add_embs)

    save_manifest(settings.chroma_dir, manifest)
    return indexed_files, skipped_files, len(to_add_ids), note


def search(settings: AppSettings, query: str) -> List[Tuple[str, float, List[Dict]]]:
    col = get_collection(settings.chroma_dir, settings.collection)
    device, note = resolve_device(settings.device)
    model = get_embedder(settings.model_name, device)

    q_emb = model.encode([query], normalize_embeddings=True)[0].tolist()

    res = col.query(
        query_embeddings=[q_emb],
        n_results=settings.top_k_chunks,
        include=["documents", "metadatas", "distances"],
    )

    docs = res["documents"][0]
    metas = res["metadatas"][0]
    dists = res["distances"][0]

    file_map: Dict[str, Dict] = {}
    for doc, meta, dist in zip(docs, metas, dists):
        fp = meta["file_path"]
        score = 1.0 - float(dist)  # cosine distance想定の簡易スコア
        entry = file_map.setdefault(fp, {"best_score": score, "hits": []})
        entry["best_score"] = max(entry["best_score"], score)
        entry["hits"].append({
            "score": score,
            "chunk_index": meta["chunk_index"],
            "file_ext": meta.get("file_ext", ""),
            "snippet": (doc[:260] + "…") if len(doc) > 260 else doc,
        })

    ranked = sorted(file_map.items(), key=lambda kv: kv[1]["best_score"], reverse=True)[: settings.top_k_files]
    # note はUI側で表示したい場合、settings.deviceから resolve_device して出せます
    return [(fp, data["best_score"], data["hits"]) for fp, data in ranked]
