import asyncio
import itertools

from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock

from telegram import Message

from bot import EditHandle, Responder
from inoue.ctx import Sender, use_context
from inoue.env import USER_ID
from inoue.render_bridge import Bridge
from inoue.render_context import OverriddenDict
from inoue.render_ctx import RenderContext
from render_core import Value

if TYPE_CHECKING:
    from inoue.render_ctx import MessageSpec, UpdateCallback

type Sent = tuple[str | None, str | None, Any]

_message_ids = itertools.count(1)


class FakeEditHandle(EditHandle):
    __slots__ = ('_key', '_responder', 'edits')

    def __init__(self, key: str, responder: Responder | None) -> None:
        self._key = key
        self._responder = responder
        self.edits: list[Sent] = []

    async def edit(
        self,
        text=None,
        parse_mode=None,
        reply_markup=None,
        media=None,
        disable_web_page_preview=None,
        allow_not_modified=False,
    ) -> bool:
        self.edits.append((text, parse_mode, reply_markup))
        return True

    async def edit_text(
        self, text, parse_mode=None, reply_markup=None, *, disable_web_page_preview=None
    ) -> bool:
        return await self.edit(text, parse_mode, reply_markup)

    async def edit_reply_markup(self, reply_markup=None) -> bool:
        return await self.edit(reply_markup=reply_markup)

    def get_message_key(self) -> str:
        return self._key

    def as_responder(self) -> Responder | None:
        return self._responder


class FakeResponder(Responder):
    '''Records every reply, and answers each one with `handle`, a handle on
    the message this responder wraps.'''

    __slots__ = ('_msg', 'replies', 'handle')

    def __init__(self, text: str = '', *, message_id: int | None = None):
        super().__init__()
        self._msg = msg = MagicMock(name='message', spec=Message)
        msg.chat_id = 0
        msg.message_id = next(_message_ids) if message_id is None else message_id
        msg.message_thread_id = None
        msg.text = text
        msg.caption = None
        msg.sender_chat = None
        msg.from_user.id = USER_ID
        self.replies: list[Sent] = []
        self.handle = FakeEditHandle(self.get_message_key(), self)

    async def reply(
        self,
        text=None,
        parse_mode=None,
        reply_markup=None,
        *,
        cached=False,
        media=None,
        disable_web_page_preview=None,
        allow_not_modified=False,
    ) -> FakeEditHandle:
        self.replies.append((text, parse_mode, reply_markup))
        return self.handle

    def reply_copy(self, from_chat_id: int, message_id: int):
        raise NotImplementedError

    def reply_forward(self, from_chat_id: int, message_id: int):
        raise NotImplementedError

    async def reply_chat_action(self, action) -> bool:
        return True

    def get_message(self) -> Message:
        return self._msg


def make_data(data: dict[str, Value] | None = None) -> OverriddenDict:
    return OverriddenDict(data or {}, {})


def make_ctx(
    data: OverriddenDict | None = None,
    *,
    sender: Sender | None = None,
    doc_id: int | None = None,
    responder: FakeResponder | None = None,
) -> tuple[RenderContext, list[MessageSpec]]:
    '''A render context, editable through `responder` when one is given. The
    list collects every message spec the context sends through its update
    callback.'''
    seen: list[MessageSpec] = []
    update_callback: UpdateCallback | None = None
    if responder is not None:
        handle = responder.handle

        async def callback(spec: MessageSpec) -> FakeEditHandle:
            seen.append(spec)
            return handle

        update_callback = callback

    with use_context(MagicMock(name='update'), None, sender):
        ctx = RenderContext(
            make_data() if data is None else data,
            doc_id=doc_id,
            path='`x' if responder is not None else None,
            update_callback=update_callback,
            responder=responder,
        )
    return ctx, seen


def bridge_of(ctx: RenderContext) -> Bridge:
    bridge = ctx.data['os']
    assert isinstance(bridge, Bridge)
    return bridge


HOST = Sender(id=USER_ID, name='host', is_guest=False)
GUEST = Sender(id=USER_ID + 1, name='guest', is_guest=True)


async def settle() -> None:
    '''Run the loop until every task besides the caller has finished and the
    done callbacks of the last ones, which run a loop iteration later, have run
    too.'''
    idle = False
    while True:
        await asyncio.sleep(0)
        if tasks := asyncio.all_tasks() - {asyncio.current_task()}:
            idle = False
            _, pending = await asyncio.wait(tasks, timeout=5)
            assert not pending, pending
        elif idle:
            return
        else:
            idle = True
