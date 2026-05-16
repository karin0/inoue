import os
from typing import Iterable
from pathlib import Path

from bot import Responder, DocumentPayload, MessageArg, command
from text import pre_block


def parse_sort(arg: str) -> Iterable[int]:
    last = None
    for x in arg.split():
        if last and len(x) < len(last):
            # last = 10086, x = 89 -> out = 10089
            x = last[: len(last) - len(x)] + x
        last = x
        yield int(x)


@command(public=True)
def handle_sort(rs: Responder, arg: MessageArg):
    if not arg:
        return rs.reply_cached('Usage: /sort 114 514 1919 810 ...')
    res = '\n'.join(str(x) for x in sorted(parse_sort(arg)))
    return rs.reply_cached(*pre_block(res))


@command('fetch')
def reply_file(rs: Responder, path: MessageArg):
    if not path:
        return rs.reply_cached('Usage: /fetch <file path>')

    size = os.path.getsize(path)
    if size > 20 << 20:
        return rs.reply_cached(f'File too large: {size} bytes')

    filename = os.path.basename(path)
    if not os.path.splitext(filename)[1]:
        filename += '.txt'

    return rs.reply(path, media=DocumentPayload(Path(path), filename=filename))
