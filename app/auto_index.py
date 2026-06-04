from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Callable, Iterable

from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

from app.core import index_folder, needs_index_update, should_index_name
from app.settings import AppSettings

# このファイルは、起動時の自動索引とフォルダ監視による再索引を管理する。
AUTO_INDEX_DEBOUNCE_SECONDS = 1.0


def clone_settings(settings) -> SimpleNamespace:
    return SimpleNamespace(**vars(settings).copy())


def should_trigger_reindex(event_type: str, is_directory: bool, paths: Iterable[str]) -> bool:
    if is_directory:
        return event_type in {"created", "deleted", "moved"}

    for path in paths:
        if not path:
            continue
        if should_index_name(path):
            return True
    return False


@dataclass
class AutoIndexStatus:
    watching_path: str | None = None
    last_reason: str | None = None
    last_completed_at: float | None = None
    last_result: tuple[int, int, int, int, str] | None = None
    last_error: str | None = None
    is_indexing: bool = False
    progress_current: int = 0
    progress_total: int = 0
    progress_path: str | None = None
    progress_stage: str | None = None


class _DocsDirEventHandler(FileSystemEventHandler):
    # watchdog のイベントを受けて、再索引が必要な変更だけを manager に渡す。
    def __init__(self, manager: "AutoIndexManager") -> None:
        self.manager = manager

    def on_any_event(self, event) -> None:
        paths = [getattr(event, "src_path", None), getattr(event, "dest_path", None)]
        if should_trigger_reindex(event.event_type, event.is_directory, paths):
            self.manager.request_reindex(f"watch:{event.event_type}")


class AutoIndexManager:
    # UI とは別スレッドで索引を動かし、監視状態と進捗も保持する。
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._index_lock = threading.Lock()
        self._observer: Observer | None = None
        self._worker_thread: threading.Thread | None = None
        self._watch_path: str | None = None
        self._settings = None
        self._timer: threading.Timer | None = None
        self._status = AutoIndexStatus()

    def configure(self, settings: AppSettings) -> None:
        settings_copy = clone_settings(settings)
        docs_dir = os.path.abspath(settings_copy.docs_dir)

        with self._lock:
            self._settings = settings_copy
            self._status.watching_path = docs_dir

            if self._watch_path == docs_dir and self._observer is not None:
                return

            self._stop_observer_locked()

            if not os.path.isdir(docs_dir):
                return

            event_handler = _DocsDirEventHandler(self)
            observer = Observer()
            observer.schedule(event_handler, docs_dir, recursive=True)
            observer.daemon = True
            observer.start()

            self._observer = observer
            self._watch_path = docs_dir

    def request_reindex(self, reason: str) -> None:
        with self._lock:
            if self._timer is not None:
                self._timer.cancel()
            timer = threading.Timer(AUTO_INDEX_DEBOUNCE_SECONDS, self._run_scheduled_reindex, args=(reason,))
            timer.daemon = True
            timer.start()
            self._timer = timer

    def run_now(
        self,
        settings: AppSettings,
        reason: str,
        progress_callback: Callable[[int, int, str, str], None] | None = None,
    ) -> tuple[int, int, int, int, str]:
        settings_copy = clone_settings(settings)
        with self._lock:
            self._settings = settings_copy
            if self._timer is not None:
                self._timer.cancel()
                self._timer = None
        return self._run_index(settings_copy, reason, progress_callback=progress_callback)

    def start_background(self, settings: AppSettings, reason: str) -> bool:
        settings_copy = clone_settings(settings)
        with self._lock:
            self._settings = settings_copy
            if self._timer is not None:
                self._timer.cancel()
                self._timer = None
            if self._worker_thread is not None and self._worker_thread.is_alive():
                return False
            if reason == "startup" and not needs_index_update(settings_copy):
                self._status.last_reason = reason
                self._status.last_completed_at = time.time()
                self._status.last_result = (0, 0, 0, 0, "起動時スキップ: 変更なし")
                self._status.last_error = None
                self._status.is_indexing = False
                self._status.progress_current = 0
                self._status.progress_total = 0
                self._status.progress_path = None
                self._status.progress_stage = None
                return False
            self._status.last_error = None
            self._status.is_indexing = True
            self._status.progress_current = 0
            self._status.progress_total = 0
            self._status.progress_path = None
            self._status.progress_stage = "読み込み中"
            worker = threading.Thread(
                target=self._run_background_index,
                args=(settings_copy, reason),
                daemon=True,
            )
            self._worker_thread = worker
            worker.start()
            return True

    def get_status(self) -> AutoIndexStatus:
        with self._lock:
            return AutoIndexStatus(
                watching_path=self._status.watching_path,
                last_reason=self._status.last_reason,
                last_completed_at=self._status.last_completed_at,
                last_result=self._status.last_result,
                last_error=self._status.last_error,
                is_indexing=self._status.is_indexing,
                progress_current=self._status.progress_current,
                progress_total=self._status.progress_total,
                progress_path=self._status.progress_path,
                progress_stage=self._status.progress_stage,
            )

    def _run_scheduled_reindex(self, reason: str) -> None:
        with self._lock:
            self._timer = None
            settings = clone_settings(self._settings) if self._settings is not None else None
        if settings is None:
            return
        self._run_index(settings, reason)

    def _run_background_index(self, settings, reason: str) -> None:
        try:
            self._run_index(settings, reason)
        finally:
            with self._lock:
                if self._worker_thread is threading.current_thread():
                    self._worker_thread = None

    def _update_progress(
        self,
        current: int,
        total: int,
        path: str,
        stage: str,
        external_callback: Callable[[int, int, str, str], None] | None = None,
    ) -> None:
        with self._lock:
            self._status.is_indexing = True
            self._status.progress_current = current
            self._status.progress_total = total
            self._status.progress_path = path or None
            self._status.progress_stage = stage
        if external_callback is not None:
            external_callback(current, total, path, stage)

    def _run_index(
        self,
        settings,
        reason: str,
        progress_callback: Callable[[int, int, str, str], None] | None = None,
    ) -> tuple[int, int, int, int, str]:
        with self._index_lock:
            with self._lock:
                self._status.is_indexing = True
                self._status.progress_current = 0
                self._status.progress_total = 0
                self._status.progress_path = None
                self._status.progress_stage = "準備中"
            try:
                result = index_folder(
                    settings,
                    progress_callback=lambda current, total, path, stage: self._update_progress(
                        current,
                        total,
                        path,
                        stage,
                        external_callback=progress_callback,
                    ),
                )
            except Exception as exc:
                with self._lock:
                    self._status.last_reason = reason
                    self._status.last_completed_at = time.time()
                    self._status.last_error = str(exc)
                    self._status.is_indexing = False
                    self._status.progress_stage = "失敗"
                raise

            with self._lock:
                self._status.last_reason = reason
                self._status.last_completed_at = time.time()
                self._status.last_result = result
                self._status.last_error = None
                self._status.is_indexing = False
                self._status.progress_current = self._status.progress_total
                self._status.progress_stage = "完了"

            return result

    def _stop_observer_locked(self) -> None:
        if self._observer is None:
            self._watch_path = None
            return

        observer = self._observer
        self._observer = None
        self._watch_path = None
        observer.stop()
        observer.join(timeout=5)
