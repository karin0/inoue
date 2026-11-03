import base64
import json
import os

from typing import TYPE_CHECKING, Any, NotRequired, TypedDict, cast

from claude_agent_sdk import create_sdk_mcp_server, tool
from telethon import TelegramClient
from telethon.tl.types import Channel, User
from telethon.utils import get_display_name

from bot import bot

from .env import USER_ID
from .text import visible

if TYPE_CHECKING:
    from collections.abc import Awaitable
    from datetime import datetime

    from claude_agent_sdk import McpSdkServerConfig
    from telethon.hints import EntityLike
    from telethon.tl.patched import Message

HISTORY_LIMIT = 200
# The Claude API rejects larger images.
IMAGE_LIMIT = 5 << 20
TEXT_LIMIT = 100_000

SERVER = 'telegram'
TOOLS = tuple(f'mcp__{SERVER}__{name}' for name in ('chat', 'media'))

type ToolResult = dict[str, Any]
type Fields = dict[str, object]


class ChatArgs(TypedDict):
    before_id: NotRequired[int]
    limit: NotRequired[int]


class MediaArgs(TypedDict):
    message_id: int


def _error(text: str) -> ToolResult:
    return {'content': [{'type': 'text', 'text': text}], 'is_error': True}


async def _download(m: Message) -> bytes:
    if (data := await m.download_media(file=bytes)) is None:
        raise RuntimeError(f'Telethon downloaded nothing from message {m.id}')
    return data


def stamp(t: datetime) -> str:
    # The local zone is the owner's, and the CLI states today's date in it too.
    return f'{t.astimezone():%Y-%m-%d %H:%M %z}'


def named(name: str | None, username: str | None, id: int | None) -> Fields:
    fields: Fields = {'name': visible(name or '?')}
    if username:
        fields['username'] = username
    if id == bot.bot.id:
        fields['role'] = 'you'
    elif id == USER_ID:
        fields['role'] = 'owner'
    return fields


def _describe(m: Message) -> Fields:
    sender = m.sender
    username = sender.username if isinstance(sender, User | Channel) else None
    fields: Fields = {
        'id': m.id,
        'from': named(get_display_name(sender), username, m.sender_id),
    }
    if m.date is not None:
        fields['date'] = stamp(m.date)
    if m.reply_to_msg_id is not None:
        fields['reply_to'] = m.reply_to_msg_id
    if (f := m.file) is not None:
        fields['media'] = {'type': f.mime_type or 'unknown', **({'name': f.name} if f.name else {})}
    if m.message:
        fields['text'] = visible(m.message)
    return fields


class Mtproto:
    '''Reads chats as `USER_ID` over MTProto, which sees every message of a chat
    where the Bot API shows a guest message and the message it replies to.'''

    __slots__ = ('client',)

    def __init__(self, session: str, api_id: int, api_hash: str):
        self.client = TelegramClient(session, api_id, api_hash)

    async def start(self):
        await self.client.connect()
        if not await self.client.is_user_authorized():
            raise RuntimeError('Telethon session is not authorized')

    async def stop(self):
        # `disconnect` returns a future when the loop is running, and its
        # signature leaves the return type out.
        await cast('Awaitable[None]', self.client.disconnect())

    async def _entity(self, chat_id: int) -> EntityLike:
        try:
            return await self.client.get_input_entity(chat_id)
        except ValueError:
            # Telethon resolves a bare id only from its cache of access hashes,
            # which fetching the dialogs fills.
            await self.client.get_dialogs()
            return await self.client.get_input_entity(chat_id)

    def server(self, chat_id: int) -> McpSdkServerConfig:
        client = self.client

        @tool(
            'chat',
            'Read messages of the current chat, oldest first, ending before `before_id` '
            'or at the latest message.',
            ChatArgs,
        )
        async def chat(args: ChatArgs) -> ToolResult:
            msgs = [
                m
                async for m in client.iter_messages(
                    await self._entity(chat_id),
                    limit=min(args.get('limit', 50), HISTORY_LIMIT),
                    offset_id=args.get('before_id', 0),
                )
            ]
            text = json.dumps([_describe(m) for m in reversed(msgs)], ensure_ascii=False)
            return {'content': [{'type': 'text', 'text': text}]}

        @tool(
            'media', 'Download an image or a text file of a message in the current chat.', MediaArgs
        )
        async def media(args: MediaArgs) -> ToolResult:
            # The signature names the raw `types.Message`, which Telethon replaces
            # with `patched.Message` on import.
            m = cast(
                'Message | None',
                await client.get_messages(await self._entity(chat_id), ids=args['message_id']),
            )
            if m is None or (f := m.file) is None:
                return _error('The message has no media.')
            mime = 'image/jpeg' if m.photo else f.mime_type or ''
            if mime.startswith('image/'):
                if (f.size or 0) > IMAGE_LIMIT:
                    return _error(f'The image has {f.size} bytes, over the limit of {IMAGE_LIMIT}.')
                data = base64.b64encode(await _download(m)).decode()
                return {'content': [{'type': 'image', 'data': data, 'mimeType': mime}]}
            if mime.startswith('text/') or mime == 'application/json':
                if (f.size or 0) > TEXT_LIMIT:
                    return _error(f'The file has {f.size} bytes, over the limit of {TEXT_LIMIT}.')
                text = (await _download(m)).decode(errors='replace')
                return {'content': [{'type': 'text', 'text': text}]}
            return _error(f'Unsupported media type: {mime or "unknown"}.')

        return create_sdk_mcp_server(SERVER, tools=[chat, media])


mtproto = (
    Mtproto(session, int(os.environ['TELETHON_API_ID']), os.environ['TELETHON_API_HASH'])
    if (session := os.environ.get('TELETHON_SESSION'))
    else None
)
