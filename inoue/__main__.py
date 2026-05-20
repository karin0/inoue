import asyncio
import atexit
import contextlib
import os
import time

from telegram import Message, MessageOriginChannel, Update, User
from telegram.constants import ChatID
from telegram.ext import BaseHandler, ContextTypes

# aiohttp must be initialized after a event loop is set.
try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())


from bot import Responder, app, shorten

from . import media, misc, run  # noqa: F401
from .commands import handle_help
from .ctx import Sender, use_context
from .env import CHAN_ID, GROUP_ID, GUEST_USER_IDS, IGNORE_CHAT_IDS, LOCK_FILE, ME, USER_ID
from .handlers import (
    handle_callback_query,
    handle_chosen_inline,
    handle_guest,
    handle_inline_query,
    handle_msg,
    handle_post,
)
from .log import log, notify


async def handle_update(update: Update, _: ContextTypes.DEFAULT_TYPE):
    # ruff: noqa: E731
    t0 = time.perf_counter()
    rs = Responder.create(update)

    effective_msg = update.effective_message
    src = None
    sender_id = None
    sender_name = None
    valid = False

    if (sender := update.effective_sender) is not None:
        sender_id = sender.id
        if isinstance(sender, User):
            sender_name = sender.full_name
            src = f'{sender_name} ({sender.name} {sender_id})'
            valid = sender_id == USER_ID
        else:  # Chat
            sender_name = sender.title
            src = f'{sender_name} [{sender.type} {sender_id}]'

    if (chat := update.effective_chat) is not None:
        if chat.id in IGNORE_CHAT_IDS:
            return

        if chat.id != sender_id:
            src2 = f'{chat.title} {{{chat.type} {chat.id}}}'
            src = f'{src} @ {src2}' if src else src2

        if not valid:
            # Only `handle_msg` accepts messages that are not from USER_ID.
            valid = chat.id == CHAN_ID or (
                # The content must be from CHAN_ID to be trusted, even if
                # auto-forwarded to GROUP_ID.
                # Note that we may only receive such updates if we are an admin
                # or the privacy mode is disabled.
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
        valid = True

    sender = Sender(sender_id, sender_name or '', is_guest) if sender_id is not None else None

    log.trace('Update from %s: %s', src, update)

    def _rs():
        if rs is None:
            raise ValueError(f'No responder for update: {update}')
        return rs

    post = None
    if (msg := update.message) is not None:
        log.info('%s: msg %s', src, shorten(msg.text))
        func = lambda: handle_msg(_rs())
    elif (msg := update.edited_message) is not None:
        log.info('%s: edited %s', src, shorten(msg.text))
        func = lambda: handle_msg(_rs())
    elif (post := update.channel_post) is not None:
        log.info('%s: channel post %s', src, shorten(post.text))
        func = lambda: handle_post(post, sender)
    elif (post := update.edited_channel_post) is not None:
        log.info('%s: edited post %s', src, shorten(post.text))
        func = lambda: handle_post(post, sender)
    elif (callback := update.callback_query) is not None:
        if isinstance(callback.message, Message):
            msg = callback.message
        log.info('%s: callback %s', src, callback.data)
        func = lambda: handle_callback_query(_rs(), callback)
    elif (query := update.inline_query) is not None:
        log.info('%s: inline %s', src, query.query)
        func = lambda: handle_inline_query(query)
    elif (chosen := update.chosen_inline_result) is not None:
        log.info('%s: chosen %s %s', src, chosen.result_id, chosen.query)
        func = lambda: handle_chosen_inline(rs, chosen)
    elif (post := update.guest_message) is not None:
        log.info('%s: guest message %s', src, shorten(post.text))
        func = lambda: handle_guest(_rs())
    else:
        log.warning('%s: unhandled: %s', src, update)
        func = None

    if (item := msg or post) != effective_msg:
        log.warning('Message mismatch: %s vs %s', item, effective_msg)

    if not valid:
        log.warning(
            'Drop unauthorized update from %s: %s\nSender: %s\nChat: %s', src, update, sender, chat
        )
        return

    if func is None:
        return

    with use_context(update, msg and rs, sender):
        try:
            try:
                if (coro := func()) is not None:
                    await coro
            except (PermissionError, ValueError) as e:
                log.exception('Error: %s: %s\nFrom: %s', type(e).__name__, e, src)
                if rs is not None:
                    await handle_help(rs)
        except Exception as e:
            with notify.revocable():
                # Can be edited to successful responses later after user edits
                log.exception('handle_update: %s: %s\nFrom: %s', type(e).__name__, e, src)

    log.debug('Exiting after %.3f secs', time.perf_counter() - t0)


class Handler(BaseHandler):
    def check_update(self, update):
        return True


def init_app():
    app.add_handler(Handler(handle_update))


def get_lock():
    fd = os.open(LOCK_FILE, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
    with os.fdopen(fd, 'w') as fp:
        fp.write(str(os.getpid()))
    atexit.register(drop_lock)


def drop_lock():
    with contextlib.suppress(OSError):
        os.remove(LOCK_FILE)
    atexit.unregister(drop_lock)


def main():
    log.info('Starting %s...', ME)
    get_lock()
    init_app()
    app.run_polling()
    log.info('%s is exiting.', ME)
    drop_lock()


if __name__ == '__main__':
    main()
