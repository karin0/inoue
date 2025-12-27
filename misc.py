import os
from typing import Iterable

from telegram import Message, Update
from telegram.ext import ContextTypes

from util import (
    pre_block,
    reply_text,
    get_msg_arg,
)


def parse_sort(arg: str) -> Iterable[int]:
    last = None
    for x in arg.split():
        if last and len(x) < len(last):
            # last = 10086, x = 89 -> out = 10089
            x = last[: len(last) - len(x)] + x
        last = x
        yield int(x)


async def handle_sort(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    msg, arg = get_msg_arg(update)
    if not arg:
        return await msg.reply_text('Usage: /sort 114 514 1919 810 ...')
    res = '\n'.join(str(x) for x in sorted(parse_sort(arg)))
    await reply_text(msg, *pre_block(res))


async def handle_fetch(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await reply_file(*get_msg_arg(update))


async def reply_file(msg: Message, path: str) -> Message:
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
