import os
import asyncio
import functools
from typing import Callable, Coroutine

from telegram import Update, Bot
from telegram.ext import (
    ApplicationBuilder,
    ContextTypes,
    MessageHandler,
    CommandHandler,
    CallbackQueryHandler,
    InlineQueryHandler,
    Application,
)

from util import log, shorten, USER_ID, CHAN_ID, init_util, set_msg
from motto import greeting
from receipt import render_receipt
from run import handle_run, handle_cmd, handle_update
from rg import handle_rg, handle_rg_callback, handle_start
from render import (
    handle_render,
    handle_doc,
    handle_render_inline_query,
    init_render,
    handle_render_callback,
    close_render,
)


def auth(
    func: Callable[[Update, ContextTypes.DEFAULT_TYPE], Coroutine],
) -> Callable[[Update, ContextTypes.DEFAULT_TYPE], Coroutine]:
    @functools.wraps(func)
    async def wrapper(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        if user := update.effective_user:
            src = f'{user.full_name} ({user.name} {user.id})'
            valid = user.id == USER_ID
        elif chat := update.effective_chat:
            src = f'{chat.type} ({chat.id})'
            valid = chat.id == CHAN_ID
        else:
            src = 'Unknown source'
            valid = False

        if msg := update.message:
            log.info('%s: msg %s', src, shorten(msg.text))
        elif msg := update.edited_message:
            log.info('%s: edited %s', src, shorten(msg.text))
        elif item := update.channel_post:
            log.info('%s: channel post %s', src, shorten(item.text))
        elif item := update.edited_channel_post:
            log.info('%s: edited post %s', src, shorten(item.text))
        elif item := update.callback_query:
            msg = update.callback_query.message
            log.info('%s: callback %s', src, shorten(item.data))
        elif item := update.inline_query:
            log.info('%s: inline %s', src, shorten(item.query))
        else:
            log.info('%s: unknown: %s', src, update)

        if not valid:
            log.warning('Drop unauthorized update from %s', src)
            return

        set_msg(msg)

        try:
            return await func(update, ctx)
        except Exception as e:
            log.exception('%s: %s: %s', func.__name__, type(e).__name__, e)

    return wrapper


async def handle_msg(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not update.message:
        if post := update.edited_channel_post or update.channel_post:
            return await handle_doc(post)
        return

    text = update.message.text
    if not (text and text.strip()):
        with open('out.ogg', 'rb') as f:
            return await update.message.reply_voice(f, do_quote=True)

    if text.startswith('/'):
        return await handle_cmd(update, text[1:].strip())

    if '\n' not in text:
        return await handle_rg(update, ctx)

    await update.message.reply_text(render_receipt(text), do_quote=True)


async def handle_callback(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    callback = update.callback_query
    data = callback.data
    fut = None
    if data:
        if data.startswith('rg_back_'):
            fut = handle_rg_callback(data)
        elif data[0] in ':-+':
            fut = handle_render_callback(update, ctx, data)
        else:
            log.warning('bad callback: %s', data)
    else:
        log.warning('empty callback')

    if fut:
        try:
            await asyncio.gather(fut, callback.answer())
        except Exception:
            await callback.answer()
            raise
    else:
        await callback.answer()


async def handle_inline_query(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    query = update.inline_query
    data = query.query.strip()
    if query and data:
        await handle_render_inline_query(query, data)


async def handle_greet(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(greeting(), do_quote=True)


commands = tuple(
    (f.__name__[f.__name__.index('_') + 1 :], f)
    for f in [
        handle_start,
        handle_greet,
        handle_run,
        handle_update,
        handle_rg,
        handle_render,
    ]
)


async def post_init(app: Application) -> None:
    bot: Bot = app.bot
    init_render('doc.db')
    init_util(bot)
    await bot.set_my_commands(tuple((s, s) for s, _ in commands))
    log.warning('Sendai initiated: ' + greeting())


async def post_stop(_: Application) -> None:
    close_render()
    log.info('Database closed.')


def main():
    app = (
        ApplicationBuilder()
        .token(os.environ['TELEGRAM_BOT_TOKEN'])
        .post_init(post_init)
        .post_stop(post_stop)
        .build()
    )

    for name, func in commands:
        app.add_handler(CommandHandler(name, auth(func)))

    app.add_handler(CallbackQueryHandler(auth(handle_callback)))
    app.add_handler(InlineQueryHandler(auth(handle_inline_query)))
    app.add_handler(MessageHandler(None, auth(handle_msg)))

    log.info('Starting Sendai...')
    app.run_polling()
    log.info('Sendai stopped.')


if __name__ == '__main__':
    main()
