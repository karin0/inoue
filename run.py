import os
import re
import asyncio
import codecs
import subprocess

from asyncio.subprocess import Process
from typing import Awaitable, Callable, cast

from telegram import InlineKeyboardButton, InlineKeyboardMarkup
from telegram.constants import ChatAction

from log import log
from env import ME, MAX_TEXT_LENGTH
from bot import (
    create_task,
    Responder,
    EditHandle,
    command,
    MessageArg,
)
from misc import reply_file
from text import pre_block

UPDATE_CWD = os.environ['UPDATE_CWD']


@command
def handle_run(rs: Responder, cmd: MessageArg):
    if cmd:
        return handle_cmd(rs, cmd)
    return rs.reply_cached('Provide a command to run.')


@command
def handle_update(rs: Responder):
    return _handle_cmd(rs, './run.sh', cwd=UPDATE_CWD)


def handle_cmd(rs: Responder, cmd: str):
    return _handle_cmd(rs, 'bash', '-c', cmd)


async def _handle_cmd(rs: Responder, bin: str, *args, **kwargs):
    log.debug(f'Spawning: {bin} {args}')
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
            await rs.reply_chat_action(ChatAction.TYPING)
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

    create_task(action())

    try:
        return await __handle_cmd(rs, child, evt)
    finally:
        child = None


async def producer(q: asyncio.Queue, pipe: asyncio.StreamReader):
    while True:
        chunk = await pipe.read(4096)
        if not chunk:
            return q.put_nowait(None)
        q.put_nowait(chunk)


reg_file = re.compile(r'^' + re.escape(ME.upper()) + r'_SEND_FILE=(.+)$', re.M)

SEND_LIMIT = 5


async def consumer[T](
    q: asyncio.Queue,
    send: Callable[[tuple[str, str | None], T], Awaitable[EditHandle | None]],
    arg: T,
):
    msg: EditHandle | None = None
    text = ''
    dec = codecs.getincrementaldecoder('utf-8')('replace')
    left = None
    send_cnt: int | None = 0

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

            content = pre_block(text, do_truncate=False)
            if msg:
                await msg.edit_text(*content)
            elif send_cnt < SEND_LIMIT:
                msg = await send(content, arg)
                send_cnt += 1
            else:
                send_cnt = SEND_LIMIT + 1  # Indicate limit reached

            if m := reg_file.search(text):
                await send((m[1], 'file'), arg)

        if eof:
            break

    if send_cnt > SEND_LIMIT:
        await send((f'Output limit of {SEND_LIMIT} messages reached.', None), arg)
        if text:
            content = pre_block(text, do_truncate=False)
            await send(content, arg)


async def __handle_cmd(rs: Responder, child: Process, evt: asyncio.Event | None):
    if child.stdout is None or child.stderr is None:
        raise RuntimeError('stdout and stderr must be captured')

    q = asyncio.Queue()
    create_task(producer(q, child.stdout))

    q_err = asyncio.Queue()
    create_task(producer(q_err, child.stderr))

    final_msg: EditHandle | None = None

    async def send(content, do_quote):
        nonlocal evt, final_msg
        text, parse_mode = content
        if parse_mode == 'file':
            try:
                r = await reply_file(rs, text)
            except Exception as e:
                r = await rs.reply(
                    f'Failed to send file {text}: {type(e).__name__}: {e}',
                )
        else:
            r = await rs.reply(text, parse_mode)
            final_msg = r
        if evt:
            evt.set()
            evt = None
        return r

    try:
        await asyncio.gather(consumer(q, send, True), consumer(q_err, send, False))
    except Exception:
        log.exception('__handle_cmd: consumer failed')
        child.terminate()

    r = await child.wait()

    text = rs.get_text()
    data = 'relay_' + text
    if len(data) <= InlineKeyboardButton.MAX_CALLBACK_DATA:
        markup = InlineKeyboardMarkup.from_button(
            InlineKeyboardButton(text, callback_data=data)
        )
    else:
        markup = None

    if r or evt:
        await rs.reply(f'{child.pid} exited with {r}', reply_markup=markup)
    elif (m := cast(EditHandle | None, final_msg)) and markup:
        await m.edit_reply_markup(markup)
