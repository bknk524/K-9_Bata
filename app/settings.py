from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
import json

DEFAULT_SETTINGS_PATH = Path("data") / "app_settings.json"


@dataclass
class AppSettings:
    # パス類
    docs_dir: str = "data/documents"
    chroma_dir: str = "data/chroma_store"

    # Chroma
    collection: str = "office_index"

    # Embedding
    model_name: str = "BAAI/bge-m3"

    # Chunk
    chunk_size: int = 900
    chunk_overlap: int = 120

    # Search
    top_k_files: int = 5
    top_k_chunks: int = 12

    # Device: "auto" | "cpu" | "cuda" | "npu"
    device: str = "auto"

    # UI上で「保存」できるようにするためJSON永続化
    @staticmethod
    def load(path: Path = DEFAULT_SETTINGS_PATH) -> "AppSettings":
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            return AppSettings(**data)
        return AppSettings()

    def save(self, path: Path = DEFAULT_SETTINGS_PATH) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), ensure_ascii=False, indent=2), encoding="utf-8")
