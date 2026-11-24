"""HTML to plain text lines, with the standard library only.

The page is treated as untrusted data: it is parsed, never executed (scripts and styles
are dropped, not run).
"""

import re
from html.parser import HTMLParser

_BLOCK_TAGS = {
    "p", "div", "h1", "h2", "h3", "h4", "h5", "h6", "li", "tr", "table", "br", "blockquote",
}  # fmt: skip
_STRIKE_TAGS = {"strike", "s", "del"}


def decode_html(raw: bytes) -> str:
    """Decode the bytes without trusting the declared charset.

    Planalto pages declare utf-8 in <meta> but are sent in cp1252 (Windows Latin-1), so
    strict utf-8 fails. Try utf-8 first, then fall back to cp1252.
    """
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return raw.decode("cp1252", errors="replace")


class _LineParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.lines: list[str] = []
        self._buf: list[str] = []
        self._skip = 0  # inside <script> or <style>
        self._strike = 0  # inside <strike>: text that the law no longer contains

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style"):
            self._skip += 1
        elif tag in _STRIKE_TAGS:
            self._strike += 1
        if tag in _BLOCK_TAGS:
            self._flush()

    def handle_endtag(self, tag):
        if tag in ("script", "style"):
            self._skip = max(0, self._skip - 1)
        elif tag in _STRIKE_TAGS:
            self._strike = max(0, self._strike - 1)
        if tag in _BLOCK_TAGS:
            self._flush()

    def handle_data(self, data):
        if not self._skip and not self._strike:
            self._buf.append(data)

    def _flush(self) -> None:
        text = "".join(self._buf).replace("\xa0", " ")
        self._buf = []
        text = re.sub(r"\s+", " ", text).strip()
        if text:
            self.lines.append(text)

    def close(self) -> None:
        super().close()
        self._flush()


def html_to_lines(raw: bytes) -> list[str]:
    """One string per block of text. Struck-through (revoked) text is left out."""
    parser = _LineParser()
    parser.feed(decode_html(raw))
    parser.close()
    return parser.lines
