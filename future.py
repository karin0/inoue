from typing import cast
from telegram import Message, Update, InlineQueryResult

from log import log

# PTB does not provide support for guest messages in API 10.0 yet.
# This provides a dirty workaround to make things work.


class UpdateProxyWithGuestMessage:
    def __init__(self, update: Update, msg: Message):
        self._update = update
        self._message = msg

    def __repr__(self):
        return f'UpdateProxyWithGuestMessage({self._update}, {self._message})'

    __str__ = __repr__

    def __getattr__(self, item):
        return getattr(self._update, item)

    @property
    def guest_message(self):
        return self._message

    @property
    def effective_message(self):
        return self._message

    @property
    def effective_sender(self):
        return self._message.from_user

    @property
    def effective_user(self):
        return self._message.from_user

    @property
    def effective_chat(self):
        return self._message.chat


class UpdateProxyWithoutGuestMessage:
    def __init__(self, update: Update):
        self._update = update

    def __repr__(self):
        return f'UpdateProxyWithoutGuestMessage({self._update})'

    __str__ = __repr__

    def __getattr__(self, item):
        return getattr(self._update, item)

    @property
    def guest_message(self):
        return None


class UpdateEx(Update):
    @property
    def guest_message(self) -> Message | None: ...


def patch_update(update: Update) -> UpdateEx:
    if hasattr(update, 'guest_message'):
        log.debug('future: update has guest_message: %s', update)
        return cast(UpdateEx, update)

    if (gm := update.api_kwargs.get('guest_message')) is not None:
        msg = Message.de_json(gm, bot=update.get_bot())
        log.debug('future: injected guest message: %s', msg)
        return cast(UpdateEx, UpdateProxyWithGuestMessage(update, msg))

    return cast(UpdateEx, UpdateProxyWithoutGuestMessage(update))


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
