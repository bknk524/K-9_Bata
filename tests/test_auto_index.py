import unittest

from app.auto_index import should_trigger_reindex


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


if __name__ == "__main__":
    unittest.main()
