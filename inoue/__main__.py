import os
import sys
import time
import atexit
import asyncio

from telegram import Message, Update, User, MessageOriginChannel
from telegram.ext import ContextTypes, BaseHandler
from telegram.constants import ChatID

# aiohttp must be initialized after a event loop is set.
try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

from bot import app, Responder, shorten

from .ctx import use_context, Sender
from .log import log, notify, trace
from .commands import reply_usage
from .env import (
    ME,
    USER_ID,
    CHAN_ID,
    GROUP_ID,
    GUEST_USER_IDS,
    IGNORE_CHAT_IDS,
    LOCK_FILE,
)
from .handlers import (
    handle_msg,
    handle_post,
    handle_callback_query,
    handle_inline_query,
    handle_chosen_inline,
    handle_guest,
)

from . import misc, media, run  # noqa: F401, E401


async def handle_update(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    # ruff: noqa: E731
    t0 = time.perf_counter()

    from .future import patch_update

    update = patch_update(update)

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

    if sender_id is not None:
        sender = Sender(sender_id, sender_name or '', is_guest)
    else:
        sender = None

    trace('Update from %s: %s', src, update)

    post = None
    if (msg := update.message) is not None:
        log.info('%s: msg %s', src, shorten(msg.text))
        func = lambda: handle_msg(msg)
    elif (msg := update.edited_message) is not None:
        log.info('%s: edited %s', src, shorten(msg.text))
        func = lambda: handle_msg(msg)
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
        func = lambda: handle_callback_query(callback)
    elif (query := update.inline_query) is not None:
        log.info('%s: inline %s', src, query.query)
        func = lambda: handle_inline_query(query)
    elif (chosen := update.chosen_inline_result) is not None:
        log.info('%s: chosen %s %s', src, chosen.result_id, chosen.query)
        func = lambda: handle_chosen_inline(chosen)
    elif (post := update.guest_message) is not None:
        log.info('%s: guest message %s', src, shorten(post.text))
        func = lambda: handle_guest(post)
    else:
        log.warning('%s: unhandled: %s', src, update)
        func = None

    if (item := msg or post) != effective_msg:
        log.warning('Message mismatch: %s vs %s', item, effective_msg)

    if not valid:
        log.warning(
            'Drop unauthorized update from %s: %s\nSender: %s\nChat: %s',
            src,
            update,
            sender,
            chat,
        )
        return

    if func is None:
        return

    with use_context(update, msg, sender):
        try:
            try:
                if (fut := func()) is not None:
                    await fut
            except (PermissionError, ValueError) as e:
                log.exception('Error: %s: %s\nFrom: %s', type(e).__name__, e, src)
                if msg is not None:
                    await reply_usage(Responder.create(msg))
        except Exception as e:
            with notify.revocable():
                # Can be edited to successful responses later after user edits
                log.exception(
                    'handle_update: %s: %s\nFrom: %s', type(e).__name__, e, src
                )

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
    try:
        os.remove(LOCK_FILE)
    except OSError as e:
        print(f'Failed to remove lock: {e}', file=sys.stderr)
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
