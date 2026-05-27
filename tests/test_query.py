from unittest.mock import patch, MagicMock

def test_search_returns_chunks():
    mock_collection = MagicMock()
    mock_collection.count.return_value = 2
    mock_collection.query.return_value = {"documents": [["chunk1", "chunk2"]]}

    with patch("query.get_embedding", return_value=[0.1, 0.2, 0.3]), \
         patch("query.get_collection", return_value=mock_collection):
        from query import search
        result = search("What is this about?")

    assert isinstance(result, list)
    assert result == ["chunk1", "chunk2"]

def test_search_returns_empty_when_no_documents():
    mock_collection = MagicMock()
    mock_collection.count.return_value = 0

    with patch("query.get_embedding", return_value=[0.1, 0.2, 0.3]), \
         patch("query.get_collection", return_value=mock_collection):
        from query import search
        result = search("What is this about?")

    assert result == []

def test_answer_returns_llm_response():
    mock_response = MagicMock()
    mock_response.json.return_value = {"response": "The answer is 42."}
    mock_response.raise_for_status = MagicMock()

    with patch("query.search", return_value=["relevant context here"]), \
         patch("query.requests.post", return_value=mock_response):
        from query import answer
        result = answer("What is the answer?")

    assert result == "The answer is 42."

def test_answer_returns_fallback_when_no_docs():
    with patch("query.search", return_value=[]):
        from query import answer
        result = answer("What is this about?")

    assert "No documents" in result
