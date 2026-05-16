from typing import Awaitable, Callable, Sequence, Concatenate

from telegram import Message, Update
from telegram.error import BadRequest

from dispatch import get_command_handler, UpdateHandler

from .log import log
from .text import truncate_text
from .app import bot
from .ctx import get_msg, get_text, use_text_override
from .env import CHAN_ID
from .responder import reroute_capture


def get_msg_url(msg_id, chat_id=None) -> str:
    if chat_id is None:
        chat_id = CHAN_ID
    chat_id = str(chat_id).removeprefix('-100')
    return f'https://t.me/c/{chat_id}/{msg_id}'


def get_deep_link_url(arg: str) -> str:
    assert bot is not None
    return f'https://t.me/{bot.username}?start={arg}'


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


def reroute_cmd(
    update: Update, text: str
) -> Awaitable[Sequence[tuple[str, str | None]]] | None:
    '''
    This is deliberately made two stages to finish the command setup before
    going async. See `/voice`.
    '''
    if reroute_capture.get() is not None:
        log.warning('reroute: already in reroute_cmd: %s', update)
        return None

    msg = get_msg(update)
    if (callback := _extract_cmd_handler(text)) is not None:
        with use_text_override(text):
            return _reroute_cmd(msg, text, callback(update))


async def _reroute_cmd(
    msg: Message, text: str, coro: Awaitable
) -> Sequence[tuple[str, str | None]]:
    with use_text_override(text):
        buf = []
        token = reroute_capture.set((msg, buf))
        try:
            await coro
        finally:
            reroute_capture.reset(token)
        return buf
