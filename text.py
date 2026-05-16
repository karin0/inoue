import re

from env import MAX_TEXT_LENGTH
from bot import truncate_text


def escape_pre(s: str) -> str:
    # `replace` is faster for small charset.
    return s.replace('\\', '\\\\').replace('`', '\\`')


type Content = tuple[str, str | None]


def pre_block_raw(s: str, *, lang: str = '') -> str:
    return f'```{lang}\n{escape_pre(s)}```'


# This truncates the text by default, trimming any leading and trailing spaces.
def pre_block(s: str, *, do_truncate: bool = True) -> Content:
    if len(s) <= MAX_TEXT_LENGTH:
        return pre_block_raw(s), 'MarkdownV2'

    if do_truncate:
        s = truncate_text(s)
    return s, None


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
