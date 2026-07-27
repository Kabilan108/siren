import re

_PATH_PATTERN = re.compile(r"/[\w.@%+-]+(?:/[\w.@%+-]+)+")
_KNOWN_ERRORS = frozenset({"server_shutdown", "orphaned by server restart"})
_MAX_ERROR_CHARS = 200


def sanitize_job_error(error: str) -> str:
    """Collapse worker stderr/exception text to a client-safe summary.

    Full detail stays in job.json and the server log; API responses must not
    leak filesystem paths, build configuration, or tracebacks.
    """
    if error in _KNOWN_ERRORS:
        return error
    first_line = error.strip().splitlines()[0] if error.strip() else "job failed"
    redacted = _PATH_PATTERN.sub("<path>", first_line)
    if len(redacted) > _MAX_ERROR_CHARS:
        redacted = redacted[:_MAX_ERROR_CHARS] + "..."
    return redacted
