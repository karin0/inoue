import re

from html import escape as html_escape

from .env import MAX_TEXT_LENGTH


def shorten(s: str | None, limit: int = 30) -> str:
    if s is None:
        return 'None'
    s = s.strip().replace('\n', ' ').replace('\r', ' ')
    if len(s) > limit:
        return s[:limit] + '...'
    return s


def truncate_text(s: str) -> str:
    s = s.strip()
    if len(s) > MAX_TEXT_LENGTH:
        s = s[: MAX_TEXT_LENGTH - 12] + '\n[truncated]'
    return s


def _create_escape_trans(chars: str) -> dict[int, str]:
    return {ord(c): '\\' + c for c in chars}


# `telegram.helpers.escape_markdown` implements this with `re.sub`, which might
# be slower than us.
_MD2_TRANS = _create_escape_trans(r'\_*[]()~`>#+-=|{}.!')


def escape(s: str) -> str:
    return s.translate(_MD2_TRANS)


def escape_pre(s: str) -> str:
    # `replace` is faster for small charset.
    return s.replace('\\', '\\\\').replace('`', '\\`')


type Content = tuple[str, str | None]


# This truncates the text by default, trimming any leading and trailing spaces.
def pre_block(s: str, *, do_truncate: bool = True) -> Content:
    if len(s) <= MAX_TEXT_LENGTH:
        return pre_block_raw(s), 'MarkdownV2'

    if do_truncate:
        s = truncate_text(s)
    return s, None


def pre_block_raw(s: str, *, lang: str = '') -> str:
    return f'```{lang}\n{escape_pre(s)}\n```'


reg_cleanup = re.compile(r'\n{3,}')
reg_cleanup_pre = re.compile(r'^```\n+(.+?)\n+```$', re.DOTALL | re.MULTILINE)


def repl_cleanup_pre(m: re.Match) -> str:
    return '```\n' + m.group(1).strip() + '\n```'


def cleanup_text(s: str) -> str:
    s = s.replace('\r', '\n')
    s = '\n'.join(map(str.rstrip, s.splitlines()))
    s = reg_cleanup.sub('\n\n', s)
    s = reg_cleanup_pre.sub(repl_cleanup_pre, s)
    return s.strip()


def _cleanup_md(s: str) -> str:
    s = s.replace('\r', '\n')
    s = '\n'.join(map(str.strip, s.splitlines()))
    s = reg_cleanup.sub('\n\n', s)
    s = reg_cleanup_pre.sub(repl_cleanup_pre, s)

    def hit(c: str) -> bool:
        return c.isdecimal() or c in '-*+`'

    # Collapse consecutive whitespaces which contains exactly one line break,
    # like in Markdown, unless it's followed by a digit or '-' to preserve lists.
    def repl(m: re.Match) -> str:
        t = m.group(0)
        if t.count('\n') == 1:
            p = m.start() - 1
            q = m.end()
            if not (p >= 0 and hit(s[p])) and not (q < len(s) and hit(s[q])):
                return ''
        return t

    return re.sub(r'\s+', repl, s).strip()


def cleanup_text_md(text: str) -> str:
    parts = text.split('```')
    n = len(parts)
    if n <= 2 or n & 1 == 0:
        return _cleanup_md(text)
    return '```'.join(
        ('\n' + part.strip() + '\n') if i & 1 else ('\n\n' + _cleanup_md(part) + '\n\n')
        for i, part in enumerate(parts)
    ).strip()
