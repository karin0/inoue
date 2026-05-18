from typing import TYPE_CHECKING
from telegram import Message, Update, InlineQueryResult

from .env import log

# PTB does not provide support for guest messages in API 10.0 yet.
# This provides a dirty workaround to make things work.


class UpdateProxy:
    __slots__ = ('_update', 'guest_message')

    def __init__(self, update: Update, guest_message: Message | None):
        self._update = update
        self.guest_message = guest_message

    def __getattr__(self, name):
        return getattr(self._update, name)

    def __repr__(self):
        return f'UpdateProxy({self._update}, {self.guest_message})'

    @property
    def effective_message(self):
        return self.guest_message or self._update.effective_message

    @property
    def effective_user(self):
        if (r := self._update.effective_user) is not None:
            return r

        if (m := self.guest_message) is not None:
            return m.from_user

        # XXX: PTB does not set `from_user` from channel posts.
        if (
            m := self._update.channel_post or self._update.edited_channel_post
        ) is not None:
            log.debug('UpdateProxy: use from_user from channel post: %s', m)
            return m.from_user

    @property
    def effective_sender(self):
        if (r := self._update.effective_sender) is not None:
            return r
        return self.effective_user

    @property
    def effective_chat(self):
        if (r := self._update.effective_chat) is not None:
            return r
        if (m := self.guest_message) is not None:
            return m.chat


if TYPE_CHECKING:

    class UpdateExt(Update):
        def __init__(self, update: Update, guest_message: Message | None): ...

        guest_message: Message | None

else:
    UpdateExt = UpdateProxy


def patch_update(update: Update) -> UpdateExt:
    if hasattr(update, 'guest_message'):
        log.warning('future: update has guest_message: %s', update)

    if (gm := update.api_kwargs.get('guest_message')) is not None:
        msg = Message.de_json(gm, bot=update.get_bot())
        log.debug('future: injected guest message: %s', msg)
        return UpdateExt(update, msg)

    return UpdateExt(update, None)


# A guest message mentions the bot in an alien chat, i.e. a group/channel we have
# not joined, or a private chat with another user.
# We can read their content, but cannot reply to them with normal messages, since
# we are not a participant of that chat.
# Instead, we have to answer them like handling inline queries, which creates an
# inline message.
async def answer_guest_query(msg: Message, result: InlineQueryResult) -> str:
    r = await msg.get_bot().do_api_request(
        'answerGuestQuery',
        {
            'guest_query_id': msg.api_kwargs['guest_query_id'],
            'result': result,
        },
    )
    log.debug('future: answerGuestQuery: %s', r)
    return r['inline_message_id']
