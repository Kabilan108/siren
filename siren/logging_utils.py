import json
import logging

logger = logging.getLogger("uvicorn")


def log_event(level: int, event: str, **fields: object) -> None:
    payload = {"event": event, **fields}
    logger.log(level, json.dumps(payload, default=str))
