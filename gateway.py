import inspect
import functools
from typing import Callable, Sequence, TYPE_CHECKING

from telegram import Message, User, Update, MessageOriginChannel
from telegram.ext import (
    Application,
    ContextTypes,
    CommandHandler,
    BaseHandler,
)
from telegram.constants import ChatID

from util.log import log, trace, notify
from util.text import shorten
from util.ctx import use_context
from util.env import USER_ID, CHAN_ID, GROUP_ID, GUEST_USER_IDS, IGNORE_CHAT_IDS

from context import Sender

if TYPE_CHECKING:
    from dispatch import PTBHandler


def auth(func: PTBHandler, *, permissive: bool = False) -> PTBHandler:
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

        trace('Entering %s from %s: %s', func.__name__, src, update)

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
                from commands import reply_usage

                assert sender is not None
                await reply_usage(msg, sender)
            return

        with use_context(update, ctx, msg, sender):
            try:
                await func(update, ctx)
            except Exception as e:
                with notify.revocable():
                    # Can be edited to successful responses later after user edits
                    log.exception('%s: %s: %s', func.__name__, type(e).__name__, e)

    return wrapper


command_callbacks: dict[str, PTBHandler] = {}


def add_command_handler(
    app: Application,
    commands: Sequence[str],
    callback: PTBHandler,
    permissive: bool = False,
):
    callback = auth(callback, permissive=permissive)
    for command in commands:
        app.add_handler(CommandHandler(command, callback))
        command_callbacks[command] = callback


def add_handler[**P, T: BaseHandler](
    app: Application,
    typ: Callable[P, T],
    permissive: bool,
    *args: P.args,
    **kwargs: P.kwargs,
):
    if kwargs:
        raise TypeError(f'Unexpected kwargs: {kwargs}')

    callback: PTBHandler | None = None
    for arg in args:
        if inspect.iscoroutinefunction(arg):
            if callback is not None:
                raise ValueError(f'Multiple callbacks in args: {args}')
            callback = arg

    if callback is None:
        raise ValueError(f'No callback in args: {args}')

    new_callback = auth(callback, permissive=permissive)
    new_args = tuple(new_callback if arg is callback else arg for arg in args)
    if new_args == args:
        raise ValueError(f'No callback found in args: {args}')

    app.add_handler(typ(*new_args))  # type: ignore[arg-type]


get_command_callback = command_callbacks.get
