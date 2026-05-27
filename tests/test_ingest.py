from unittest.mock import patch, MagicMock

def test_ingest_pdf_adds_new_chunks(tmp_path):
    fake_pdf = str(tmp_path / "test.pdf")

    mock_doc = MagicMock()
    mock_doc.page_content = "This is test content for the RAG system."
    mock_doc.metadata = {"page": 0}

    mock_collection = MagicMock()
    mock_collection.get.return_value = {"ids": []}

    with patch("ingest.PyPDFLoader") as mock_loader, \
         patch("ingest.get_embedding", return_value=[0.1, 0.2, 0.3]), \
         patch("ingest.get_collection", return_value=mock_collection):

        mock_loader.return_value.load.return_value = [mock_doc]
        from ingest import ingest_pdf
        count = ingest_pdf(fake_pdf)

    assert count >= 1
    mock_collection.add.assert_called()

def test_ingest_pdf_skips_duplicate_chunks(tmp_path):
    fake_pdf = str(tmp_path / "test.pdf")

    mock_doc = MagicMock()
    mock_doc.page_content = "Duplicate content."
    mock_doc.metadata = {"page": 0}

    mock_collection = MagicMock()
    mock_collection.get.return_value = {"ids": ["existing-id"]}

    with patch("ingest.PyPDFLoader") as mock_loader, \
         patch("ingest.get_embedding", return_value=[0.1, 0.2, 0.3]), \
         patch("ingest.get_collection", return_value=mock_collection):

        mock_loader.return_value.load.return_value = [mock_doc]
        from ingest import ingest_pdf
        count = ingest_pdf(fake_pdf)

    assert count == 0
    mock_collection.add.assert_not_called()
