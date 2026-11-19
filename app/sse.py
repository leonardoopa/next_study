"""Server-Sent Events formatting."""

import json


def sse_event(event: str, data: dict | list) -> str:
    """One SSE event: an `event:` line, a `data:` line and a blank line that ends it.

    json.dumps escapes newlines inside strings, so the payload always fits on the single
    `data:` line (a raw newline would end the field and corrupt the event).
    """
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"
