from typing import Awaitable, Callable, Sequence, Concatenate

from telegram.error import BadRequest

from bot import bot, truncate_text, Responder

from .log import log
from .env import CHAN_ID


def get_msg_url(msg_id, chat_id=None) -> str:
    if chat_id is None:
        chat_id = CHAN_ID
    chat_id = str(chat_id).removeprefix('-100')
    return f'https://t.me/c/{chat_id}/{msg_id}'


def get_deep_link_url(arg: str) -> str:
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


async def reroute_cmd(
    rs: Responder, text: str
) -> Sequence[tuple[str, str | None]] | None:
    if rs.is_captured():
        log.warning('reroute: already in reroute_cmd: %s', rs)
        return None

    if (callback := rs.get_route()) is not None:
        buf = []
        old_text = rs.get_text()
        rs.set_text(text)
        try:
            with rs.capture(buf):
                await callback(rs)
        finally:
            rs.set_text(old_text)
        return buf
