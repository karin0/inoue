import asyncio
import codecs
import functools
import subprocess

from telegram import Message, Update
from telegram.ext import ContextTypes
from telegram.constants import ChatAction

from util import get_arg, MAX_TEXT_LENGTH


async def handle_run(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if cmd := get_arg():
        await handle_cmd(update, cmd)
    else:
        await update.message.reply_text('Provide a command to run.', do_quote=True)


async def handle_update(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await _handle_cmd(update, './run.sh', cwd='/data/app/gmact')


async def handle_cmd(update: Update, cmd: str):
    return await _handle_cmd(update, 'bash', '-c', cmd)


async def _handle_cmd(update: Update, bin, *args, **kwargs):
    child = await asyncio.create_subprocess_exec(
        bin,
        *args,
        **kwargs,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    async def action():
        await asyncio.sleep(0.2)
        if child.returncode is None:
            await update.effective_chat.send_chat_action(ChatAction.TYPING)

    asyncio.create_task(action())

    async def producer(q, pipe):
        async for line in pipe:
            q.put_nowait(line)
        q.put_nowait(None)

    async def consumer(q, send):
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
                        # print('got', chunks)
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
                    msg = await send(text)

            if eof:
                break

            asyncio.create_task(
                update.effective_chat.send_chat_action(ChatAction.TYPING)
            )

        return text

    q = asyncio.Queue()
    asyncio.create_task(producer(q, child.stdout))

    q_err = asyncio.Queue()
    asyncio.create_task(producer(q_err, child.stderr))

    r1, r2 = await asyncio.gather(
        consumer(q, functools.partial(update.message.reply_text, do_quote=True)),
        consumer(q_err, update.message.reply_text),
    )
    r = await child.wait()
    if r or not (r1 or r2):
        await update.message.reply_text(f'{child.pid} exited with {r}', do_quote=True)
