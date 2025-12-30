from unittest.mock import MagicMock, patch

import pytest

import server


@pytest.mark.asyncio
async def test_ensure_model_loaded_reuses_model():
    sentinel = object()
    with patch("server.load_model", MagicMock(return_value=sentinel)) as load_model:
        model = await server.ensure_model_loaded("distil-small.en")
        assert model is sentinel
        assert server.current_model_name == "distil-small.en"

        model_again = await server.ensure_model_loaded("distil-small.en")
        assert model_again is sentinel
        assert load_model.call_count == 1


@pytest.mark.asyncio
async def test_ensure_model_loaded_error():
    with patch("server.load_model", side_effect=RuntimeError("boom")):
        with pytest.raises(RuntimeError, match="boom"):
            await server.ensure_model_loaded("distil-small.en")
