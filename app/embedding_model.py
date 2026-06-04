from __future__ import annotations

from pathlib import Path

from huggingface_hub import snapshot_download

DEFAULT_EMBEDDING_MODEL = "BAAI/bge-m3"
DEFAULT_EMBEDDING_MODEL_DIR = Path("data") / "embedding_model"
LOCAL_MODEL_MARKERS = (
    "config.json",
    "modules.json",
    "config_sentence_transformers.json",
    "sentence_bert_config.json",
)


def ensure_embedding_model_dir(path: Path = DEFAULT_EMBEDDING_MODEL_DIR) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def is_local_embedding_model_dir(path: Path) -> bool:
    return any((path / marker).exists() for marker in LOCAL_MODEL_MARKERS)


def install_default_embedding_model(path: Path = DEFAULT_EMBEDDING_MODEL_DIR) -> Path:
    model_dir = ensure_embedding_model_dir(path)
    if is_local_embedding_model_dir(model_dir):
        return model_dir

    snapshot_download(
        repo_id=DEFAULT_EMBEDDING_MODEL,
        local_dir=str(model_dir),
    )

    if not is_local_embedding_model_dir(model_dir):
        raise RuntimeError(f"Failed to install embedding model into {model_dir}")

    return model_dir


def load_embedding_model_name(path: Path = DEFAULT_EMBEDDING_MODEL_DIR) -> str:
    model_dir = install_default_embedding_model(path)
    return str(model_dir.resolve())
