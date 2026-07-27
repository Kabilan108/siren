import json
import logging
from unittest.mock import patch

from siren.logging_utils import log_event


def test_log_event_emits_one_compact_json_line() -> None:
    with patch("siren.logging_utils.logger.log") as logger_log:
        log_event(
            logging.ERROR,
            "transcribe_error",
            request_id="request-1",
            error="first line\nsecond line",
        )

    logger_log.assert_called_once()
    level, message = logger_log.call_args.args
    assert level == logging.ERROR
    assert "\n" not in message
    assert json.loads(message) == {
        "event": "transcribe_error",
        "request_id": "request-1",
        "error": "first line\nsecond line",
    }
