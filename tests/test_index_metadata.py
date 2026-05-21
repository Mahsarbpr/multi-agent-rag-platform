from rag_assistant.index_metadata import (
    calculate_file_hash,
    calculate_data_fingerprint,
    load_saved_fingerprint,
    save_fingerprint,
    data_changed,
)


def test_calculate_file_hash_changes_when_file_content_changes(tmp_path):
    file_path = tmp_path / "notes.txt"

    file_path.write_text("original content", encoding="utf-8")
    original_hash = calculate_file_hash(file_path)

    file_path.write_text("updated content", encoding="utf-8")
    updated_hash = calculate_file_hash(file_path)

    assert original_hash != updated_hash


def test_calculate_data_fingerprint_includes_only_pdf_and_txt_files(tmp_path):
    txt_file = tmp_path / "notes.txt"
    pdf_file = tmp_path / "paper.pdf"
    ignored_file = tmp_path / "image.png"

    txt_file.write_text("some notes", encoding="utf-8")
    pdf_file.write_bytes(b"fake pdf content")
    ignored_file.write_text("ignore me", encoding="utf-8")

    fingerprint = calculate_data_fingerprint(str(tmp_path))

    assert "notes.txt" in fingerprint
    assert "paper.pdf" in fingerprint
    assert "image.png" not in fingerprint


def test_save_and_load_fingerprint(tmp_path):
    metadata_path = tmp_path / "faiss_index" / "index_metadata.json"

    fingerprint = {
        "notes.txt": "abc123",
        "paper.pdf": "def456",
    }

    save_fingerprint(str(metadata_path), fingerprint)
    loaded_fingerprint = load_saved_fingerprint(str(metadata_path))

    assert loaded_fingerprint == fingerprint


def test_load_saved_fingerprint_returns_none_when_file_does_not_exist(tmp_path):
    metadata_path = tmp_path / "missing_metadata.json"

    result = load_saved_fingerprint(str(metadata_path))

    assert result is None


def test_data_changed_returns_true_when_no_saved_metadata_exists(tmp_path):
    data_folder = tmp_path / "data"
    data_folder.mkdir()

    file_path = data_folder / "notes.txt"
    file_path.write_text("hello", encoding="utf-8")

    metadata_path = tmp_path / "faiss_index" / "index_metadata.json"

    assert data_changed(str(data_folder), str(metadata_path)) is True


def test_data_changed_returns_false_when_fingerprint_matches(tmp_path):
    data_folder = tmp_path / "data"
    data_folder.mkdir()

    file_path = data_folder / "notes.txt"
    file_path.write_text("hello", encoding="utf-8")

    metadata_path = tmp_path / "faiss_index" / "index_metadata.json"

    fingerprint = calculate_data_fingerprint(str(data_folder))
    save_fingerprint(str(metadata_path), fingerprint)

    assert data_changed(str(data_folder), str(metadata_path)) is False


def test_data_changed_returns_true_when_file_is_modified(tmp_path):
    data_folder = tmp_path / "data"
    data_folder.mkdir()

    file_path = data_folder / "notes.txt"
    file_path.write_text("hello", encoding="utf-8")

    metadata_path = tmp_path / "faiss_index" / "index_metadata.json"

    fingerprint = calculate_data_fingerprint(str(data_folder))
    save_fingerprint(str(metadata_path), fingerprint)

    file_path.write_text("hello updated", encoding="utf-8")

    assert data_changed(str(data_folder), str(metadata_path)) is True