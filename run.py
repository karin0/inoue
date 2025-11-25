import os
import asyncio
import codecs
import subprocess

from asyncio.subprocess import Process
from typing import Awaitable, Callable, TypeVar

from telegram import Message, Update
from telegram.ext import ContextTypes
from telegram.constants import ChatAction

from util import get_msg, get_msg_arg, MAX_TEXT_LENGTH

UPDATE_CWD = os.environ['UPDATE_CWD']


async def handle_run(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    msg, cmd = get_msg_arg(update)
    if cmd:
        await handle_cmd(update, cmd)
    else:
        await msg.reply_text('Provide a command to run.', do_quote=True)


async def handle_update(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await _handle_cmd(update, './run.sh', cwd=UPDATE_CWD)


async def handle_cmd(update: Update, cmd: str):
    return await _handle_cmd(update, 'bash', '-c', cmd)


async def _handle_cmd(update: Update, bin: str, *args, **kwargs):
    msg = get_msg(update)
    child = await asyncio.create_subprocess_exec(
        bin,
        *args,
        **kwargs,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    evt = asyncio.Event()

    async def action():
        await asyncio.sleep(0.3)
        while child and child.returncode is None:
            nonlocal evt
            await msg.reply_chat_action(ChatAction.TYPING)
            if evt:
                futs = (
                    asyncio.create_task(evt.wait()),
                    asyncio.create_task(asyncio.sleep(3)),
                )
                await asyncio.wait(futs, return_when=asyncio.FIRST_COMPLETED)
                if evt.is_set():
                    evt = None
            else:
                await asyncio.sleep(3)

    asyncio.create_task(action())

    try:
        return await __handle_cmd(msg, child, evt)
    finally:
        child = None


async def producer(q: asyncio.Queue, pipe: asyncio.StreamReader):
    while True:
        chunk = await pipe.read(4096)
        if not chunk:
            return q.put_nowait(None)
        q.put_nowait(chunk)


T = TypeVar('T')


async def consumer(
    q: asyncio.Queue, send: Callable[[str, T], Awaitable[Message]], arg: T
):
    msg: Message | None = None
    text = ''
    dec = codecs.getincrementaldecoder('utf-8')('replace')
    left = None

    while True:
        eof = False
        if left:
            new = left
            left = None
        else:
            t = await q.get()
            if t is None:
                break

            chunks = bytearray(t)
            await asyncio.sleep(0.01)
            try:
                while True:
                    t = q.get_nowait()
                    if t is None:
                        eof = True
                        break
                    chunks.extend(t)
            except asyncio.QueueEmpty:
                pass

            new = dec.decode(chunks, final=eof)

        if new:
            if len(new) > MAX_TEXT_LENGTH:
                msg = None
                text = new[:MAX_TEXT_LENGTH]
                left = new[MAX_TEXT_LENGTH:]
            else:
                text += new

                if len(text) > MAX_TEXT_LENGTH:
                    msg = None
                    text = new

            if msg:
                await msg.edit_text(text)
            else:
                msg = await send(text, arg)

        if eof:
            break


async def __handle_cmd(msg: Message, child: Process, evt: asyncio.Event):
    q = asyncio.Queue()
    asyncio.create_task(producer(q, child.stdout))

    q_err = asyncio.Queue()
    asyncio.create_task(producer(q_err, child.stderr))

    async def send(text, do_quote):
        nonlocal evt
        r = await msg.reply_text(text, do_quote=do_quote)
        if evt:
            evt.set()
            evt = None
        return r

    await asyncio.gather(consumer(q, send, True), consumer(q_err, send, False))
    r = await child.wait()
    if r or evt:
        await msg.reply_text(f'{child.pid} exited with {r}', do_quote=True)
