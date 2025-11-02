import os
import functools
from typing import Callable, Coroutine

from telegram import Update, Bot
from telegram.ext import (
    ApplicationBuilder,
    ContextTypes,
    MessageHandler,
    CommandHandler,
    CallbackQueryHandler,
    Application,
)

from util import log, shorten
from calc import render_calc
from run import handle_run, handle_cmd, handle_update
from rg import handle_rg, handle_rg_callback, handle_start
from render import handle_render, handle_doc, init_render, handle_render_callback

USER_ID = int(os.environ['USER_ID'])
CHAN_ID = int(os.environ['CHAN_ID'])


def auth(
    func: Callable[[Update, ContextTypes.DEFAULT_TYPE], Coroutine],
) -> Callable[[Update, ContextTypes.DEFAULT_TYPE], Coroutine]:
    @functools.wraps(func)
    async def wrapper(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        user = update.effective_user
        if user:
            src = f'{user.full_name} ({user.name} {user.id})'
            valid = user.id == USER_ID
        else:
            chat = update.effective_chat
            if not chat:
                raise ValueError('Invalid message')
            src = f'Chat {chat.id} ({chat.type})'
            valid = chat.id == CHAN_ID

        if msg := update.message:
            log.info('%s: %s', src, shorten(msg.text))
        elif update.callback_query:
            log.info('%s: callback %s', src, shorten(update.callback_query.data))
        elif update.channel_post:
            log.info('%s: channel post %s', src, shorten(update.channel_post.text))
        elif update.edited_message:
            log.info('%s: edited %s', src, shorten(update.edited_message.text))
        elif update.edited_channel_post:
            log.info(
                '%s: edited post %s',
                src,
                shorten(update.edited_channel_post.text),
            )
        else:
            log.info('%s: unknown: %s', src, update)

        if not valid:
            log.info('Drop message from unknown chat')
            return

        try:
            return await func(update, ctx)
        except Exception as e:
            r = f'{type(e).__name__}: {str(e)}'
            log.exception('%s', r)
            if msg:
                await msg.reply_text(r, do_quote=True)

    return wrapper


@auth
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
    await update.message.reply_text(render_calc(text), do_quote=True)


async def handle_callback(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    callback = update.callback_query
    await callback.answer()

    data = callback.data
    if data.startswith('rg_back_'):
        await handle_rg_callback(data)
    elif data.startswith(':'):
        await handle_render_callback(update, ctx, data)


async def post_init(app: Application) -> None:
    bot: Bot = app.bot
    await bot.set_my_commands(
        (
            ('render', 'render'),
            ('start', 'start'),
            ('run', 'run'),
            ('update', 'update'),
            ('rg', 'rg'),
        )
    )
    await bot.send_message(USER_ID, 'Inoue bot started.')
    init_render('doc.db')


def main():
    token = os.environ['TELEGRAM_BOT_TOKEN']
    app = ApplicationBuilder().token(token).post_init(post_init).build()

    app.add_handler(CommandHandler('start', auth(handle_start)))
    app.add_handler(CommandHandler('run', auth(handle_run)))
    app.add_handler(CommandHandler('update', auth(handle_update)))
    app.add_handler(CommandHandler('rg', auth(handle_rg)))
    app.add_handler(CommandHandler('render', auth(handle_render)))

    app.add_handler(CallbackQueryHandler(auth(handle_callback)))
    app.add_handler(MessageHandler(None, handle_msg))

    log.info('Starting Inoue bot...')
    app.run_polling()
    log.info('Inoue Bot stopped.')


if __name__ == '__main__':
    main()
