import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import app.core as core_module
from app.core import (
    collect_indexable_paths,
    delete_old_chunks_for_paths,
    entry_type,
    file_name_collection_name,
    filename_to_embedding_text,
    find_onnx_model_path,
    query_collection,
    index_folder,
    is_temporary_office_file,
    is_supported_file,
    manifest_entry_is_complete,
    manifest_entry_has_size,
    merge_hits,
    normalized_extension,
    path_signature,
    prune_stale_files,
    resolve_embedder_spec,
    resolve_path_signature,
    select_npu_onnx_provider,
    should_index_file_name,
    should_index_folder_name,
    should_index_name,
    weighted_similarity_score,
    get_collection,
)
from app.embedding_model import (
    DEFAULT_EMBEDDING_MODEL,
    install_default_embedding_model,
    load_embedding_model_name,
)
from app.settings import AppSettings


class FakeCollection:
    def __init__(self) -> None:
        self.deleted_paths = []

    def delete(self, where):
        file_path = where["file_path"]
        if isinstance(file_path, dict) and "$in" in file_path:
            self.deleted_paths.extend(file_path["$in"])
            return
        self.deleted_paths.append(file_path)


class FakeChromaCollection:
    def __init__(self, count_value: int = 0) -> None:
        self.count_value = count_value
        self.deleted_paths = []
        self.add_calls = []

    def count(self):
        return self.count_value

    def delete(self, where):
        self.deleted_paths.append(where["file_path"])

    def add(self, **kwargs):
        self.add_calls.append(kwargs)


class FakeVector(list):
    def tolist(self):
        return list(self)


class FakePersistentClient:
    created = 0
    get_collection_calls = 0

    def __init__(self, **kwargs) -> None:
        FakePersistentClient.created += 1

    def get_or_create_collection(self, name):
        FakePersistentClient.get_collection_calls += 1
        return {"name": name}


class FakeQueryCollection:
    def __init__(self, count_value: int) -> None:
        self.count_value = count_value
        self.requested_n_results = None

    def count(self):
        return self.count_value

    def query(self, *, query_embeddings, n_results, include):
        self.requested_n_results = n_results
        return {
            "documents": [["doc"] * n_results],
            "metadatas": [[{"file_path": "E:/docs/report.txt"}] * n_results],
            "distances": [[0.1] * n_results],
        }


class CoreHelpersTest(unittest.TestCase):
    def test_collect_indexable_paths_includes_files_and_nested_folders_for_name_index(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "keep.txt").write_text("hello", encoding="utf-8")
            (root / "archive.zip").write_bytes(b"\x00\x01")
            (root / "nested").mkdir()
            (root / "nested" / "slide.pptx").write_bytes(b"pptx")

            files = collect_indexable_paths(tmpdir)

            self.assertEqual(
                files,
                sorted(
                    [
                        str((root / "archive.zip").resolve()),
                        str((root / "keep.txt").resolve()),
                        str((root / "nested").resolve()),
                        str((root / "nested" / "slide.pptx").resolve()),
                    ]
                ),
            )

    def test_prune_stale_files_removes_missing_paths_from_manifest_and_collection(self):
        manifest = {
            "E:/docs/current.txt": {"sha256": "a"},
            "C:/old/docs/current.txt": {"sha256": "a"},
            "C:/old/docs/removed.pdf": {"sha256": "b"},
        }

        removed = prune_stale_files(manifest, ["E:/docs/current.txt"])

        self.assertEqual(removed, ["C:/old/docs/current.txt", "C:/old/docs/removed.pdf"])
        self.assertEqual(list(manifest.keys()), ["E:/docs/current.txt"])

    def test_delete_old_chunks_for_paths_deletes_from_all_collections(self):
        col1 = FakeCollection()
        col2 = FakeCollection()

        delete_old_chunks_for_paths([col1, col2], ["E:/docs/current.txt"])

        self.assertEqual(col1.deleted_paths, ["E:/docs/current.txt"])
        self.assertEqual(col2.deleted_paths, ["E:/docs/current.txt"])

    def test_query_collection_caps_n_results_to_collection_count(self):
        col = FakeQueryCollection(count_value=2)

        docs, metas, dists = query_collection(col, [0.1, 0.2], 5)

        self.assertEqual(col.requested_n_results, 2)
        self.assertEqual(len(docs), 2)
        self.assertEqual(len(metas), 2)
        self.assertEqual(len(dists), 2)

    def test_find_onnx_model_path_prefers_onnx_subfolder(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir) / "model"
            onnx_dir = model_dir / "onnx"
            onnx_dir.mkdir(parents=True)
            model_path = onnx_dir / "model.onnx"
            model_path.write_bytes(b"onnx")

            self.assertEqual(find_onnx_model_path(str(model_dir)), model_path.resolve())

    def test_select_npu_onnx_provider_uses_priority_order(self):
        providers = ["CPUExecutionProvider", "DmlExecutionProvider", "OpenVINOExecutionProvider"]

        self.assertEqual(select_npu_onnx_provider(providers), "OpenVINOExecutionProvider")

    def test_filename_to_embedding_text_contains_original_and_normalized_name(self):
        text = filename_to_embedding_text("E:/docs/K-9Project_team-C_v2.pptx")

        self.assertIn("K-9Project_team-C_v2.pptx", text)
        self.assertIn("K 9 Project team C v2", text)
        self.assertIn("pptx", text)

    def test_file_name_collection_name_appends_suffix(self):
        self.assertEqual(file_name_collection_name("office_index"), "office_index__file_names")

    def test_normalized_extension_lowercases_suffix(self):
        self.assertEqual(normalized_extension("E:/docs/REPORT.PDF"), ".pdf")

    def test_is_supported_file_uses_extension_allow_list(self):
        self.assertTrue(is_supported_file("E:/docs/REPORT.PDF"))
        self.assertFalse(is_supported_file("E:/docs/archive.zip"))
        self.assertFalse(is_supported_file("E:/docs/~$REPORT.PDF"))

    def test_should_index_file_name_allows_unsupported_extensions(self):
        self.assertTrue(should_index_file_name("E:/docs/archive.zip"))
        self.assertFalse(should_index_file_name("E:/docs/~$draft.pptx"))

    def test_is_temporary_office_file_detects_lock_file(self):
        self.assertTrue(is_temporary_office_file("E:/docs/~$draft.pptx"))
        self.assertFalse(is_temporary_office_file("E:/docs/draft.pptx"))

    def test_should_index_folder_name_allows_named_directories(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            nested = Path(tmpdir) / "nested"
            nested.mkdir()
            self.assertTrue(should_index_folder_name(str(nested)))

    def test_should_index_name_allows_folders_and_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            nested = Path(tmpdir) / "nested"
            nested.mkdir()
            file_path = Path(tmpdir) / "archive.zip"
            file_path.write_bytes(b"zip")
            self.assertTrue(should_index_name(str(nested)))
            self.assertTrue(should_index_name(str(file_path)))

    def test_path_signature_for_folder_is_name_based(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            folder_path = Path(tmpdir) / "ProjectA"
            folder_path.mkdir()
            self.assertEqual(path_signature(str(folder_path)), path_signature(str(folder_path)))

    def test_entry_type_recognizes_folder(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            folder_path = Path(tmpdir) / "ProjectA"
            folder_path.mkdir()
            self.assertEqual(entry_type(str(folder_path)), "folder")

    def test_manifest_entry_is_complete_detects_legacy_entries(self):
        self.assertFalse(manifest_entry_is_complete(None))
        self.assertFalse(manifest_entry_is_complete({"sha256": "abc"}))
        self.assertTrue(
            manifest_entry_is_complete(
                {
                    "sha256": "abc",
                    "entry_type": "file",
                    "content_indexed": True,
                }
            )
        )

    def test_manifest_entry_has_size_detects_size_field(self):
        self.assertFalse(manifest_entry_has_size(None))
        self.assertFalse(manifest_entry_has_size({"sha256": "abc"}))
        self.assertTrue(manifest_entry_has_size({"sha256": "abc", "size": 128}))

    def test_resolve_path_signature_reuses_manifest_sha_for_unchanged_supported_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "report.txt"
            file_path.write_text("hello", encoding="utf-8")
            prev = {
                "sha256": "cached-sha",
                "mtime": file_path.stat().st_mtime,
                "size": file_path.stat().st_size,
                "entry_type": "file",
                "content_indexed": True,
            }

            with patch("app.core.path_signature", side_effect=AssertionError("path_signature should not run")):
                sha, is_current = resolve_path_signature(
                    str(file_path),
                    prev,
                    content_supported=True,
                    mtime=file_path.stat().st_mtime,
                    size=file_path.stat().st_size,
                )

            self.assertEqual(sha, "cached-sha")
            self.assertTrue(is_current)

    def test_get_collection_reuses_cached_client_and_collection(self):
        core_module._chroma_client_cache.clear()
        core_module._collection_cache.clear()
        FakePersistentClient.created = 0
        FakePersistentClient.get_collection_calls = 0

        with patch("app.core.chromadb.PersistentClient", FakePersistentClient):
            col1 = get_collection("data/chroma_store", "office_index")
            col2 = get_collection("data/chroma_store", "office_index")

        self.assertEqual(FakePersistentClient.created, 1)
        self.assertEqual(FakePersistentClient.get_collection_calls, 1)
        self.assertEqual(col1, col2)
        core_module._chroma_client_cache.clear()
        core_module._collection_cache.clear()

    def test_resolve_embedder_spec_uses_cuda_for_gpu_when_available(self):
        with patch("app.core.detect_torch_cuda_available", return_value=True), \
            patch("app.core.detect_torch_npu_device", return_value=None), \
            patch("app.core.find_onnx_model_path", return_value=None):
            spec = resolve_embedder_spec("cuda", "E:/models/bge")

        self.assertEqual(spec.backend, "torch")
        self.assertEqual(spec.resolved_device, "gpu")
        self.assertEqual(spec.torch_device, "cuda")

    def test_resolve_embedder_spec_uses_onnx_for_npu_when_provider_available(self):
        onnx_path = Path("E:/models/bge/onnx/model.onnx")
        with patch("app.core.detect_torch_cuda_available", return_value=False), \
            patch("app.core.detect_torch_npu_device", return_value=None), \
            patch("app.core.find_onnx_model_path", return_value=onnx_path), \
            patch("app.core.get_available_onnx_providers", return_value=["CPUExecutionProvider", "QNNExecutionProvider"]):
            spec = resolve_embedder_spec("npu", "E:/models/bge")

        self.assertEqual(spec.backend, "onnx")
        self.assertEqual(spec.resolved_device, "npu")
        self.assertEqual(spec.onnx_provider, "QNNExecutionProvider")
        self.assertEqual(spec.onnx_model_path, str(onnx_path))

    def test_resolve_embedder_spec_falls_back_to_cpu_when_npu_unavailable(self):
        with patch("app.core.detect_torch_cuda_available", return_value=False), \
            patch("app.core.detect_torch_npu_device", return_value=None), \
            patch("app.core.find_onnx_model_path", return_value=None):
            spec = resolve_embedder_spec("npu", "E:/models/bge")

        self.assertEqual(spec.backend, "torch")
        self.assertEqual(spec.resolved_device, "cpu")
        self.assertEqual(spec.torch_device, "cpu")

    def test_weighted_similarity_score_downweights_folder_name_hits(self):
        self.assertEqual(weighted_similarity_score("content", 0.0), 1.0)
        self.assertEqual(weighted_similarity_score("file_name", 0.0), 0.85)
        self.assertEqual(weighted_similarity_score("folder_name", 0.0), 0.55)

    def test_merge_hits_prefers_content_over_folder_name_when_scores_are_close(self):
        file_map = {}

        merge_hits(
            file_map,
            ["営業資料"],
            [{"file_path": "E:/docs/営業資料", "kind": "folder_name", "entry_type": "folder"}],
            [0.0],
            default_kind="file_name",
        )
        merge_hits(
            file_map,
            ["営業資料の本文"],
            [{"file_path": "E:/docs/proposal.pdf", "kind": "content", "entry_type": "file"}],
            [0.2],
            default_kind="content",
        )

        ranked = sorted(file_map.items(), key=lambda kv: kv[1]["best_score"], reverse=True)

        self.assertEqual(ranked[0][0], "E:/docs/proposal.pdf")
        self.assertEqual(ranked[1][0], "E:/docs/営業資料")

    def test_index_folder_rebuilds_name_index_when_folders_are_missing_from_legacy_index(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            docs_dir = Path(tmpdir) / "docs"
            docs_dir.mkdir()
            nested_dir = docs_dir / "営業資料"
            nested_dir.mkdir()
            file_path = docs_dir / "report.txt"
            file_path.write_text("quarterly plan", encoding="utf-8")

            settings = AppSettings(
                docs_dir=str(docs_dir),
                chroma_dir=str(Path(tmpdir) / "chroma"),
                collection="test_collection",
            )
            content_col = FakeChromaCollection()
            name_col = FakeChromaCollection(count_value=1)
            saved_manifest = {}

            class FakeModel:
                def encode(self, texts, **kwargs):
                    return [FakeVector([float(i + 1)]) for i in range(len(texts))]

            old_manifest = {
                str(file_path.resolve()): {
                    "sha256": path_signature(str(file_path)),
                    "mtime": file_path.stat().st_mtime,
                    "ext": ".txt",
                    "chunks": 1,
                }
            }

            def fake_get_collection(chroma_dir, name):
                if name == settings.collection:
                    return content_col
                if name == file_name_collection_name(settings.collection):
                    return name_col
                raise AssertionError(name)

            def fake_save_manifest(chroma_dir, manifest):
                saved_manifest.update(manifest)

            with patch("app.core.get_collection", side_effect=fake_get_collection), \
                patch("app.core.load_manifest", return_value=old_manifest.copy()), \
                patch("app.core.save_manifest", side_effect=fake_save_manifest), \
                patch("app.core.load_embedding_model_name", return_value="fake-model"), \
                patch("app.core.resolve_device", return_value=("cpu", "CPU")), \
                patch("app.core.get_embedder", return_value=FakeModel()):
                index_folder(settings)

            self.assertEqual(len(name_col.add_calls), 1)
            added_metas = name_col.add_calls[0]["metadatas"]
            self.assertTrue(any(meta["kind"] == "folder_name" for meta in added_metas))
            self.assertIn(str(nested_dir.resolve()), saved_manifest)
            self.assertNotIn(str(docs_dir.resolve()), saved_manifest)
            self.assertEqual(saved_manifest[str(nested_dir.resolve())]["entry_type"], "folder")
            self.assertFalse(saved_manifest[str(nested_dir.resolve())]["content_indexed"])

    def test_index_folder_falls_back_to_name_only_when_text_extraction_fails(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            docs_dir = Path(tmpdir) / "docs"
            docs_dir.mkdir()
            file_path = docs_dir / "broken.pptx"
            file_path.write_bytes(b"not-a-real-pptx")

            settings = AppSettings(
                docs_dir=str(docs_dir),
                chroma_dir=str(Path(tmpdir) / "chroma"),
                collection="test_collection",
            )
            content_col = FakeChromaCollection()
            name_col = FakeChromaCollection()
            saved_manifest = {}

            class FakeModel:
                def encode(self, texts, **kwargs):
                    return [FakeVector([float(i + 1)]) for i in range(len(texts))]

            def fake_get_collection(chroma_dir, name):
                if name == settings.collection:
                    return content_col
                if name == file_name_collection_name(settings.collection):
                    return name_col
                raise AssertionError(name)

            def fake_save_manifest(chroma_dir, manifest):
                saved_manifest.update(manifest)

            with patch("app.core.get_collection", side_effect=fake_get_collection), \
                patch("app.core.load_manifest", return_value={}), \
                patch("app.core.save_manifest", side_effect=fake_save_manifest), \
                patch("app.core.load_embedding_model_name", return_value="fake-model"), \
                patch("app.core.resolve_embedder_spec", return_value=core_module.EmbedderSpec(backend="torch", resolved_device="cpu", torch_device="cpu", note="CPU")), \
                patch("app.core.get_embedder", return_value=FakeModel()), \
                patch("app.core.extract_text", side_effect=RuntimeError("broken")):
                indexed, skipped, chunks, removed, note = index_folder(settings)

            self.assertEqual(indexed, 1)
            self.assertEqual(chunks, 0)
            self.assertIn("抽出失敗 1 件", note)
            self.assertEqual(len(name_col.add_calls), 1)
            self.assertEqual(content_col.add_calls, [])
            self.assertFalse(saved_manifest[str(file_path.resolve())]["content_indexed"])

    def test_index_folder_reports_progress(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            docs_dir = Path(tmpdir) / "docs"
            docs_dir.mkdir()
            file_path = docs_dir / "note.txt"
            file_path.write_text("hello world", encoding="utf-8")

            settings = AppSettings(
                docs_dir=str(docs_dir),
                chroma_dir=str(Path(tmpdir) / "chroma"),
                collection="test_collection",
            )
            content_col = FakeChromaCollection()
            name_col = FakeChromaCollection()
            progress_events = []

            class FakeModel:
                def encode(self, texts, **kwargs):
                    return [FakeVector([float(i + 1)]) for i in range(len(texts))]

            def fake_get_collection(chroma_dir, name):
                if name == settings.collection:
                    return content_col
                if name == file_name_collection_name(settings.collection):
                    return name_col
                raise AssertionError(name)

            with patch("app.core.get_collection", side_effect=fake_get_collection), \
                patch("app.core.load_manifest", return_value={}), \
                patch("app.core.save_manifest"), \
                patch("app.core.load_embedding_model_name", return_value="fake-model"), \
                patch("app.core.resolve_embedder_spec", return_value=core_module.EmbedderSpec(backend="torch", resolved_device="cpu", torch_device="cpu", note="CPU")), \
                patch("app.core.get_embedder", return_value=FakeModel()):
                index_folder(settings, progress_callback=lambda current, total, path, stage: progress_events.append((current, total, stage)))

            self.assertTrue(progress_events)
            self.assertEqual(progress_events[0][2], "準備中")
            self.assertEqual(progress_events[-1][2], "完了")

    def test_load_embedding_model_name_returns_local_model_dir_when_model_exists(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir) / "embedding_model"
            model_dir.mkdir()
            (model_dir / "config.json").write_text("{}", encoding="utf-8")

            with patch("app.embedding_model.snapshot_download") as snapshot_download_mock:
                self.assertEqual(
                    load_embedding_model_name(model_dir),
                    str(model_dir.resolve()),
                )

            snapshot_download_mock.assert_not_called()

    def test_install_default_embedding_model_downloads_into_target_folder(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir) / "embedding_model"

            def fake_snapshot_download(*, repo_id, local_dir):
                self.assertEqual(repo_id, DEFAULT_EMBEDDING_MODEL)
                self.assertEqual(Path(local_dir), model_dir)
                (model_dir / "config.json").write_text("{}", encoding="utf-8")

            with patch("app.embedding_model.snapshot_download", side_effect=fake_snapshot_download) as snapshot_download_mock:
                installed_dir = install_default_embedding_model(model_dir)

            self.assertEqual(installed_dir, model_dir)
            self.assertTrue((model_dir / "config.json").exists())
            snapshot_download_mock.assert_called_once()

    def test_load_embedding_model_name_returns_installed_local_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir) / "embedding_model"

            def fake_snapshot_download(*, repo_id, local_dir):
                self.assertEqual(repo_id, DEFAULT_EMBEDDING_MODEL)
                self.assertEqual(Path(local_dir), model_dir)
                (model_dir / "config.json").write_text("{}", encoding="utf-8")

            with patch("app.embedding_model.snapshot_download", side_effect=fake_snapshot_download):
                self.assertEqual(load_embedding_model_name(model_dir), str(model_dir.resolve()))

    def test_app_settings_load_ignores_legacy_model_name_field(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            settings_file = Path(tmpdir) / "app_settings.json"
            settings_file.write_text(
                '{"docs_dir":"docs","model_name":"legacy-model","top_k_files":7}',
                encoding="utf-8",
            )

            settings = AppSettings.load(settings_file)

            self.assertEqual(settings.docs_dir, "docs")
            self.assertEqual(settings.top_k_files, 7)


if __name__ == "__main__":
    unittest.main()
