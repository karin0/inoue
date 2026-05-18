from typing import Sequence

from bot import bot, Responder

from .log import log
from .env import CHAN_ID


def get_msg_url(msg_id, chat_id=None) -> str:
    if chat_id is None:
        chat_id = CHAN_ID
    chat_id = str(chat_id).removeprefix('-100')
    return f'https://t.me/c/{chat_id}/{msg_id}'


def get_deep_link_url(arg: str) -> str:
    return f'https://t.me/{bot.username}?start={arg}'


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
