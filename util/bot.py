import asyncio

from typing import Awaitable, Callable, Sequence, Concatenate, cast
from contextlib import contextmanager
from contextvars import ContextVar

from telegram import Message, InlineKeyboardMarkup, Update
from telegram.constants import ChatAction
from telegram.error import BadRequest

from db import db
from dispatch import get_command_handler, UpdateHandler

from .log import log
from .text import truncate_text
from .app import bot, create_task
from .ctx import get_text, use_text_override
from .env import USER_ID, CHAN_ID, GROUP_ID
from .proxy import InlineMessageProxy


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


async def try_send_text_or_not_modified[**P, R](
    func: Callable[Concatenate[str, P], Awaitable[R]],
    text: str,
    *args: P.args,
    **kwargs: P.kwargs,
) -> R | None:
    '''`parse_mode` must be specified via `kwargs` to enable the fallback.'''
    try:
        return await try_send_text(func, text, *args, **kwargs)
    except BadRequest as e:
        if 'Message is not modified' in str(e):
            log.info('try_send_text_allow_not_modified: message not modified')
            return None
        raise


reroute_capture: ContextVar[tuple[int, int, list[tuple[str, str | None]]] | None] = (
    ContextVar('reroute_capture', default=None)
)


# Note: the return value could be `None` if `allow_not_modified` is set.
async def reply_text(
    m: Message | InlineMessageProxy,
    text: str,
    parse_mode: str | None = None,
    reply_markup: InlineKeyboardMarkup | None = None,
    *,
    disable_web_page_preview: bool = False,
    allow_not_modified: bool = False,
) -> Message | None:
    if not isinstance(m, Message):
        # XXX: Skip for `InlineMessageProxy`, since we won't receive updates for
        # edited guest messages anyway.
        log.debug('reply_text: found proxy: %s', m)
        return cast(
            Message,
            await m.reply_text(text, parse_mode=parse_mode, reply_markup=reply_markup),
        )

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
            disable_web_page_preview=disable_web_page_preview,
            do_quote=True,
            allow_sending_without_reply=True,
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
            disable_web_page_preview=disable_web_page_preview,
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
    task = create_task(_keep_action(msg, action))
    try:
        yield task
    finally:
        task.cancel()


def _extract_cmd_handler(text: str) -> UpdateHandler | None:
    if text and text[0] == '/':
        p = min(
            x
            for x in (text.find(' '), text.find('\n'), text.find('@'), len(text))
            if x > 0
        )
        cmd = text[1:p]
        callback = get_command_handler(cmd)
        log.debug('route_cmd: %r -> %r %r %r', text, p, cmd, callback)
        if callback is not None:
            log.info('route_cmd: %s: dispatching to %s', cmd, callback)
            return callback
        log.debug('route_cmd: command not found: %s', cmd)


def route_cmd(update: Update, msg: Message) -> Awaitable | None:
    if (callback := _extract_cmd_handler(get_text(msg))) is not None:
        return callback(update)


async def reroute_cmd(
    update: Update, text: str
) -> Sequence[tuple[str, str | None]] | None:
    if (msg := update.effective_message) is None:
        raise RuntimeError('No message')

    if reroute_capture.get() is not None:
        log.warning('reroute: already in reroute_cmd: %s', update)
        return None

    if (callback := _extract_cmd_handler(text)) is not None:
        with use_text_override(text):
            buf = []
            token = reroute_capture.set((msg.chat_id, msg.message_id, buf))
            try:
                await callback(update)
            finally:
                reroute_capture.reset(token)
            return buf
