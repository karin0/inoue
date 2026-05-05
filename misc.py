import os
from typing import Iterable

from telegram import Message

from util import pre_block, reply_text
from dispatch import MessageArg, command


def parse_sort(arg: str) -> Iterable[int]:
    last = None
    for x in arg.split():
        if last and len(x) < len(last):
            # last = 10086, x = 89 -> out = 10089
            x = last[: len(last) - len(x)] + x
        last = x
        yield int(x)


@command(public=True)
async def handle_sort(msg: Message, arg: MessageArg):
    if not arg:
        return await msg.reply_text('Usage: /sort 114 514 1919 810 ...')
    res = '\n'.join(str(x) for x in sorted(parse_sort(arg)))
    await reply_text(msg, *pre_block(res))


@command('fetch')
async def reply_file(msg: Message, path: MessageArg) -> Message:
    if not path:
        return await msg.reply_text('Usage: /fetch <file path>', do_quote=True)

    size = os.path.getsize(path)
    if size > 20 << 20:
        return await msg.reply_text(f'File too large: {size} bytes', do_quote=True)

    filename = os.path.basename(path)
    if not os.path.splitext(filename)[1]:
        filename += '.txt'

    with open(path, 'rb') as fp:
        return await msg.reply_document(
            fp, filename=filename, caption=path, do_quote=True
        )
