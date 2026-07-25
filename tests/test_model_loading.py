from unittest.mock import MagicMock, patch

import pytest

from siren import models


@pytest.mark.asyncio
async def test_ensure_model_loaded_reuses_model():
    sentinel = object()
    with patch(
        "siren.models.load_backend", MagicMock(return_value=sentinel)
    ) as load_backend:
        backend = await models.ensure_model_loaded("distil-small.en")
        assert backend is sentinel
        assert models.current_model_name == "distil-small.en"

        backend_again = await models.ensure_model_loaded("distil-small.en")
        assert backend_again is sentinel
        assert load_backend.call_count == 1


@pytest.mark.asyncio
async def test_ensure_model_loaded_error():
    with patch("siren.models.load_backend", side_effect=RuntimeError("boom")):
        with pytest.raises(RuntimeError, match="boom"):
            await models.ensure_model_loaded("distil-small.en")
