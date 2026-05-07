import asyncio

from typing import Awaitable, Callable, Sequence, Concatenate
from contextlib import contextmanager
from contextvars import ContextVar

from telegram import Bot, Message, MessageEntity, InlineKeyboardMarkup, Update
from telegram.ext import ContextTypes
from telegram.constants import ChatAction
from telegram.error import BadRequest

from db import db
from gateway import get_command_callback

from .log import log
from .text import truncate_text
from .ctx import use_text_override
from .env import USER_ID, CHAN_ID, GROUP_ID

bot: Bot | None = None


def get_msg_url(msg_id, chat_id=None) -> str:
    if chat_id is None:
        chat_id = CHAN_ID
    chat_id = str(chat_id).removeprefix('-100')
    return f'https://t.me/c/{chat_id}/{msg_id}'


def get_deep_link_url(arg: str) -> str:
    assert bot is not None
    return f'https://t.me/{bot.username}?start={arg}'


def encode_chat_id(m: Message, default: str = 'u') -> str:
    chat_id = m.chat_id
    if chat_id == USER_ID:
        return default
    if chat_id == CHAN_ID:
        return 'c'
    if chat_id == GROUP_ID:
        return 'g'
    return f'G{chat_id}'


async def try_send_text[**P, R](
    func: Callable[Concatenate[str, P], Awaitable[R]],
    text: str,
    *args: P.args,
    **kwargs: P.kwargs,
) -> R:
    '''`parse_mode` must be specified via `kwargs` to enable the fallback.'''
    if not ('parse_mode' in kwargs or 'entities' in kwargs):
        return await func(text, *args, **kwargs)

    try:
        return await func(text, *args, **kwargs)
    except BadRequest as e:
        if "Can't parse entities:" in (e_str := str(e)):
            log.warning(
                'try_send_text: falling back without entities: %s %s %s %s',
                e,
                func,
                args,
                kwargs,
            )
            new_text = truncate_text(f'{e_str}\n{text}')
            kwargs.pop('parse_mode', None)
            kwargs.pop('entities', None)
            return await func(new_text, *args, **kwargs)
        raise


reroute_capture: ContextVar[tuple[int, int, list[tuple[str, str | None]]] | None] = (
    ContextVar('reroute_capture', default=None)
)


# Note: the return value could be `None` if `allow_not_modified` is set.
async def reply_text(
    m: Message,
    text: str,
    parse_mode: str | None = None,
    reply_markup: InlineKeyboardMarkup | None = None,
    *,
    entities: Sequence[MessageEntity] | None = None,
    allow_not_modified: bool = False,
) -> Message | None:
    chat_kind = encode_chat_id(m)
    if chat_kind == 'c':
        log.warning('reply_text: channel chat: %s', m)
    key = f'{chat_kind}-{m.message_id}'

    # Can only be called when `key` is missing in the cache.
    async def _do_reply_text(save: bool = True):
        resp = await try_send_text(
            m.reply_text,
            text,
            parse_mode=parse_mode,
            reply_markup=reply_markup,
            entities=entities,
            do_quote=True,
        )

        if save:
            val = str(resp.message_id)
            db[key] = val
            log.debug('Sending new response: %s -> %s', key, val)
        return resp

    if (
        (reroute := reroute_capture.get()) is not None
        and reroute[0] == m.chat_id
        and reroute[1] == m.message_id
    ):
        log.info('reply_text: reroute_capture: %s', key)
        reroute[2].append((text, parse_mode))

        # Do not try to edit the reply, or we will mess up the response of the
        # capturing context (`/render`).
        return await _do_reply_text(save=False)

    if not (resp_msg_id := db.get(key)):
        return await _do_reply_text()

    resp_msg_id = int(resp_msg_id)
    log.debug('Editing cached response: %s -> %s', key, resp_msg_id)

    assert bot is not None
    try:
        resp = await try_send_text(
            bot.edit_message_text,
            text,
            m.chat.id,
            resp_msg_id,
            parse_mode=parse_mode,
            reply_markup=reply_markup,
            entities=entities,
        )
    except Exception as e:
        # Cache expired, remove it first for other coroutines.
        # We don't bypass 'Message is not modified' here, as the user side cannot
        # distinguish whether the message is being updated.
        # This behavior can be overridden by `allow_not_modified`.
        e = str(e)
        if 'Message is not modified' not in e:
            del db[key]
            fmt = 'Failed to edit response: %s -> %s: %s: %s'
            log.warning(fmt, key, resp_msg_id, type(e).__name__, e)
        elif allow_not_modified:
            log.info('Message not modified: %s -> %s', key, resp_msg_id)
            return None
        else:
            del db[key]

        return await _do_reply_text()

    assert isinstance(resp, Message)
    return resp


async def _keep_action(msg: Message, action: ChatAction):
    try:
        while True:
            await msg.reply_chat_action(action)
            await asyncio.sleep(4)
    except asyncio.CancelledError:
        pass


@contextmanager
def keep_chat_action(msg: Message, action: ChatAction):
    '''Re-send chat action every 4s so it stays visible during long operations.'''
    task = asyncio.create_task(_keep_action(msg, action))
    try:
        yield task
    finally:
        task.cancel()


async def reroute_cmd(
    update: Update, ctx: ContextTypes.DEFAULT_TYPE, text: str
) -> Sequence[tuple[str, str | None]] | None:
    if (msg := update.effective_message) is None:
        raise RuntimeError('No message')

    if reroute_capture.get():
        log.warning('reroute: already in reroute_cmd: %s', text)
        return None

    p = min(x for x in (text.find(' '), text.find('@'), len(text)) if x > 0)
    cmd = text[1:p]
    log.info('reroute: %s: %r', cmd, text)

    callback = get_command_callback(cmd)
    if callback is None:
        log.info('reroute: command not found: %s', cmd)
        return None

    log.info('reroute: %s: dispatching to %s', cmd, callback)
    buf = []
    token = reroute_capture.set((msg.chat_id, msg.message_id, buf))
    try:
        with use_text_override(text):
            await callback(update, ctx)
    finally:
        reroute_capture.reset(token)

    return buf
