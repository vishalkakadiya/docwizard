from unittest.mock import patch, MagicMock

def test_get_embedding_returns_list_of_floats():
    mock_response = MagicMock()
    mock_response.json.return_value = {"embedding": [0.1, 0.2, 0.3]}
    mock_response.raise_for_status = MagicMock()

    with patch("embeddings.requests.post", return_value=mock_response) as mock_post:
        from embeddings import get_embedding
        result = get_embedding("hello world")

    mock_post.assert_called_once()
    assert isinstance(result, list)
    assert result == [0.1, 0.2, 0.3]
