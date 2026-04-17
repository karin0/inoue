import os
import sys
import asyncio
import logging
import functools
from typing import Callable, Coroutine

from telegram import Message, User, Update, Bot, MessageOriginChannel
from telegram.ext import (
    ApplicationBuilder,
    ContextTypes,
    MessageHandler,
    CommandHandler,
    CallbackQueryHandler,
    ChosenInlineResultHandler,
    InlineQueryHandler,
    Application,
)
from telegram.error import NetworkError
from telegram.constants import ChatID, ChatType

from util import (
    log,
    notify,
    pre_block,
    reply_text,
    shorten,
    USER_ID,
    CHAN_ID,
    GROUP_ID,
    GUEST_USER_IDS,
    IGNORE_CHAT_IDS,
    DB_FILE,
    ME,
    init_util,
    use_msg,
    get_msg,
    do_notify,
)
from db import db
from context import Sender, get_sender
from inoue import render_receipt
from rg import handle_rg, handle_rg_callback
from voice import try_handle_voice
from ytdlp import extract_url, handle_yt_inline_query, handle_yt_chosen_result
from render import (
    handle_render_doc,
    handle_render_callback,
    handle_render_group,
    handle_render_inline_query,
    CALLBACK_SPECIAL,
)
from commands import dispatch_cmd, set_commands, stats, commands, reply_usage


def auth(
    func: Callable[[Update, ContextTypes.DEFAULT_TYPE], Coroutine],
    *,
    permissive: bool = False,
) -> Callable[[Update, ContextTypes.DEFAULT_TYPE], Coroutine]:
    @functools.wraps(func)
    async def wrapper(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        effective_msg = update.effective_message
        src = None
        sender_id = None
        sender_name = None
        valid = False

        if sender := update.effective_sender:
            sender_id = sender.id
            if isinstance(sender, User):
                sender_name = sender.full_name
                src = f'{sender_name} ({sender.name} {sender_id})'
                valid = sender_id == USER_ID
            else:  # Chat
                sender_name = sender.title
                src = f'{sender_name} [{sender.type} {sender_id}]'

        if chat := update.effective_chat:
            if chat.id in IGNORE_CHAT_IDS:
                return

            if chat.id != sender_id:
                src2 = f'{chat.title} {{{chat.type} {chat.id}}}'
                src = f'{src} @ {src2}' if src else src2

            if not valid and permissive:
                # Only `handle_msg` accepts messages that are not from USER_ID.
                valid = chat.id == CHAN_ID or (
                    # The content must be from CHAN_ID to be trusted, even if
                    # auto-forwarded to GROUP_ID.
                    chat.id == GROUP_ID
                    and (msg := effective_msg)
                    and (from_user := msg.from_user)
                    and from_user.id == ChatID.SERVICE_CHAT
                    and msg.is_automatic_forward
                    and isinstance(origin := msg.forward_origin, MessageOriginChannel)
                    and origin.chat.id == CHAN_ID
                )

        if is_guest := not valid and sender_id in GUEST_USER_IDS:
            src = f'{src} (guest)'
            valid = permissive

        if sender_id:
            sender = Sender(sender_id, sender_name or '', is_guest)
        else:
            sender = None

        log.debug('Entering %s from %s: %s', func.__name__, src, update)

        item = callback = query = None
        if msg := update.message:
            log.info('%s: msg %s', src, shorten(msg.text))
        elif msg := update.edited_message:
            log.info('%s: edited %s', src, shorten(msg.text))
        elif item := update.channel_post:
            log.info('%s: channel post %s', src, shorten(item.text))
        elif item := update.edited_channel_post:
            log.info('%s: edited post %s', src, shorten(item.text))
        elif callback := update.callback_query:
            if isinstance(callback.message, Message):
                msg = callback.message
            log.info('%s: callback %s', src, callback.data)
        elif query := update.inline_query:
            log.info('%s: inline %s', src, query.query)
        elif chosen := update.chosen_inline_result:
            log.info('%s: chosen %s %s', src, chosen.result_id, chosen.query)
        else:
            log.info('%s: unknown: %s', src, update)

        if (item := msg or item) != effective_msg:
            log.warning('Message mismatch: %s vs %s', item, effective_msg)

        if not valid:
            log.warning(
                '%s: Drop unauthorized update from %s: %s\nSender: %s\nChat: %s',
                func.__name__,
                src,
                update,
                sender,
                chat,
            )
            if is_guest and msg:
                assert sender is not None
                await reply_usage(msg, sender)
            return

        with use_msg(msg, sender):
            try:
                return await func(update, ctx)
            except Exception as e:
                with notify.revocable():
                    # Can be edited to successful responses later after user edits
                    log.exception('%s: %s: %s', func.__name__, type(e).__name__, e)

    return wrapper


async def handle_msg(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    sender = get_sender()
    is_guest = bool(sender and sender.is_guest)
    if not is_guest and (post := update.edited_channel_post or update.channel_post):
        return await handle_render_doc(update, post)

    if not (msg := get_msg(update)):
        return

    # `render` handles Doc messages that are forwarded from CHAN_ID to its discussion group.
    if (
        isinstance(origin := msg.forward_origin, MessageOriginChannel)
        and msg.is_automatic_forward
        and msg.chat_id == GROUP_ID
        and origin.chat.id == CHAN_ID
    ):
        return await handle_render_group(msg, origin.message_id)

    if msg.chat.type != ChatType.PRIVATE:
        return

    if await try_handle_voice(update, ctx):
        return

    # ID Bot
    if origin:
        return await msg.reply_text(*pre_block(str(origin)), do_quote=True)

    if is_guest:
        assert sender is not None
        await reply_usage(msg, sender)
        return

    # Privileged operations are only allowed in private chats, even if it's from
    # USER_ID.
    if msg.chat_id != USER_ID:
        # This should not happen since non-private chats are already skipped.
        log.error('handle_msg: unauthorized update: %s', update)
        return

    if not ((text := msg.text) and text.strip()):
        if (
            msg.forum_topic_created
            or msg.forum_topic_edited
            or msg.forum_topic_closed
            or msg.forum_topic_reopened
        ):
            log.debug('Ignoring forum_topic: %s', msg)
            return

        with open('out.ogg', 'rb') as f:
            return await msg.reply_voice(f, do_quote=True)

    # Be more careful, since you won't be able to log in again within 10 minutes.
    if text == '/Please log out now/':
        await reply_text(msg, 'See you next time!')
        if await ctx.bot.log_out():
            log.info('log_out: success')
            sys.exit(1)
        else:
            log.error('log_out: failed')
        return

    if text.startswith('/'):
        return await dispatch_cmd(update, ctx, msg, text)

    if '\n' not in text:
        return await handle_rg(update, ctx)

    await reply_text(msg, *pre_block(render_receipt(text)))


async def handle_callback(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    callback = update.callback_query
    assert callback
    data = callback.data
    fut = None
    if data:
        if data.startswith('rg_'):
            fut = handle_rg_callback(data)
        elif data[0] in CALLBACK_SPECIAL:
            fut = handle_render_callback(update, ctx, data)
        elif data != 'noop':
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
    assert query
    if data := query.query.strip():
        if (parsed := extract_url(data)) is not None:
            await handle_yt_inline_query(query, parsed)
        else:
            await handle_render_inline_query(update, query, data)


async def handle_chosen_inline(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    result = update.chosen_inline_result
    assert result
    result_id = result.result_id
    if result_id.startswith('yt_'):
        query = result.query.strip()
        if result.inline_message_id and (parsed := extract_url(query)) is not None:
            await handle_yt_chosen_result(
                ctx.bot,
                result_id,
                parsed,
                result.inline_message_id,
            )
        else:
            log.warning('Invalid chosen inline result: %s', result)
    elif not result_id.startswith('noop'):
        log.error('Bad chosen inline result: %s', result)


async def post_init(app: Application) -> None:
    bot: Bot = app.bot
    init_util(bot)
    db.connect(DB_FILE)
    log.info('%s initiated: %s', ME, bot.bot)
    await asyncio.gather(
        set_commands(bot),
        do_notify(
            *stats(bot.bot, f'{ME} initiated'), quiet=log.isEnabledFor(logging.DEBUG)
        ),
    )


async def post_stop(_: Application) -> None:
    db.close()
    log.info('Database closed.')


async def handle_error(update, context: ContextTypes.DEFAULT_TYPE) -> None:
    e = context.error
    if isinstance(e, NetworkError):
        with notify.suppress():
            log.error('Network error in update %s: %s: %s', update, type(e).__name__, e)
    elif isinstance(e, Exception):
        log.error(
            'Exception in update %s: %s: %s', update, type(e).__name__, e, exc_info=e
        )
    else:
        log.error('Unknown error in update %s: %s', update, e)


def build_app():
    builder = (
        ApplicationBuilder()
        .token(os.environ['TELEGRAM_BOT_TOKEN'])
        .post_init(post_init)
        .post_stop(post_stop)
        .read_timeout(30)
        .write_timeout(30)
        .media_write_timeout(60)
    )

    # Looks like http://127.0.0.1:8081/
    if url := os.environ.get('LOCAL_SERVER'):
        log.info('Using local server: %s', url)
        builder = (
            builder.base_url(url + 'bot')
            .base_file_url(url + 'file/bot')
            .local_mode(True)
        )

    app = builder.build()
    app.add_error_handler(handle_error)

    rg = None
    for name, (func, permissive) in commands.items():
        f = auth(func, permissive=permissive)
        if name == 'rg':
            rg = f
        app.add_handler(CommandHandler(name, f))

    assert rg, 'handle_rg not found'

    for off in range(5):
        app.add_handler(CommandHandler(f'rg{off}', rg))

    app.add_handler(CallbackQueryHandler(auth(handle_callback, permissive=True)))
    app.add_handler(InlineQueryHandler(auth(handle_inline_query, permissive=True)))
    app.add_handler(
        ChosenInlineResultHandler(auth(handle_chosen_inline, permissive=True))
    )
    app.add_handler(MessageHandler(None, auth(handle_msg, permissive=True)))

    return app


def main():
    log.info('Starting %s...', ME)
    build_app().run_polling()
    log.info('%s stopped.', ME)


if __name__ == '__main__':
    main()
