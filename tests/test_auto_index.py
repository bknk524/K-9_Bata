import threading
import time
import unittest
from unittest.mock import patch

from app.auto_index import AutoIndexManager, should_trigger_reindex
from app.settings import AppSettings


class AutoIndexHelpersTest(unittest.TestCase):
    def test_supported_file_event_triggers_reindex(self):
        self.assertTrue(
            should_trigger_reindex("created", False, ["E:/docs/report.pdf"])
        )

    def test_unsupported_file_event_triggers_reindex_for_name_index(self):
        self.assertTrue(
            should_trigger_reindex("modified", False, ["E:/docs/image.png"])
        )

    def test_directory_created_triggers_reindex(self):
        self.assertTrue(
            should_trigger_reindex("created", True, ["E:/docs/new_folder"])
        )

    def test_directory_modified_does_not_trigger_reindex(self):
        self.assertFalse(
            should_trigger_reindex("modified", True, ["E:/docs"])
        )

    def test_moved_supported_file_triggers_reindex_from_dest_path(self):
        self.assertTrue(
            should_trigger_reindex("moved", False, ["E:/docs/tmp.bin", "E:/docs/notes.txt"])
        )

    def test_start_background_sets_loading_status_and_avoids_duplicate_start(self):
        manager = AutoIndexManager()
        started = threading.Event()
        release = threading.Event()

        def fake_run_index(self, settings, reason, progress_callback=None):
            started.set()
            release.wait(timeout=1.0)
            with self._lock:
                self._status.is_indexing = False
            return (0, 0, 0, 0, "ok")

        with patch("app.auto_index.needs_index_update", return_value=True), \
            patch.object(AutoIndexManager, "_run_index", autospec=True, side_effect=fake_run_index):
            self.assertTrue(manager.start_background(AppSettings(), "startup"))
            self.assertTrue(started.wait(timeout=1.0))
            self.assertFalse(manager.start_background(AppSettings(), "startup"))

            status = manager.get_status()
            self.assertTrue(status.is_indexing)
            self.assertEqual(status.progress_stage, "読み込み中")

            release.set()
            time.sleep(0.05)

    def test_start_background_skips_startup_when_no_changes(self):
        manager = AutoIndexManager()

        with patch("app.auto_index.needs_index_update", return_value=False):
            started = manager.start_background(AppSettings(), "startup")

        self.assertFalse(started)
        status = manager.get_status()
        self.assertFalse(status.is_indexing)
        self.assertEqual(status.last_reason, "startup")
        self.assertEqual(status.last_result, (0, 0, 0, 0, "起動時スキップ: 変更なし"))


if __name__ == "__main__":
    unittest.main()
