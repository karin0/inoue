from html import escape as html_escape  # noqa: F401

from telegram.constants import MessageLimit


def shorten(s: str | None, limit: int = 30) -> str:
    if s is None:
        return 'None'
    s = s.strip().replace('\n', ' ').replace('\r', ' ')
    if len(s) > limit:
        return s[:limit] + '...'
    return s


def truncate_text(s: str, limit: int = MessageLimit.MAX_TEXT_LENGTH) -> str:
    s = s.strip()
    if len(s) > limit:
        if limit < 12:
            return '[truncated]'[:limit] if limit > 0 else ''
        s = s[: limit - 12] + '\n[truncated]'
    return s


def _create_escape_trans(chars: str) -> dict[int, str]:
    return {ord(c): '\\' + c for c in chars}


# `telegram.helpers.escape_markdown` implements this with `re.sub`, which might
# be slower than us.
_MD2_TRANS = _create_escape_trans(r'\_*[]()~`>#+-=|{}.!')


def escape(s: str) -> str:
    return s.translate(_MD2_TRANS)
