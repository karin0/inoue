# The Telegram layer

This package talks to Telegram and knows nothing about this particular bot. A
command function names a `Responder` and never names a chat id, a message id or
an API method, which is what lets the same function serve a person typing in a
chat, a document calling it through `Bridge.exec`, a button on an inline message
and a guest.

A command's output is described by three abstractions. `Responder` is where the
output goes. `EditHandle` is the position in that output being written.
`MediaPayload` is what the output is made of and where its bytes currently live.

A `Responder` is this project's file descriptor. `get_text` and `set_text` read
and rewrite argv, `reply` writes stdout, `capture` redirects it and `get_route`
is exec, which is all `reroute_cmd` needs to run a command from a document.

## Output mechanisms

Telegram offers three output mechanisms, and they differ on two axes.

| Mechanism | Writes | Address |
| --- | --- | --- |
| `send_message` and the `send_*` media methods | many, each a new message | learned after the first write, from the response |
| the `edit_message_*` methods with an `inline_message_id` | many, always the same slot | given before the first write |
| `answer_guest_query` | exactly one | none until the answer returns one |

Each implementation reconciles one mechanism with the single `reply()` call that
commands are written against.

`MessageResponder` sends, then records where it sent. `InlineResponder` keeps
every reply of the exchange in `_fragments` and hands the whole output to its
`_emitter` after each reply, since an inline message is one slot that cannot be
appended to. `InlineEmitter` holds the address and re-edits that slot.
`GuestEmitter` answers the guest query with the output so far.

## Guest exchanges

`GuestEmitter.emit` ends by replacing itself:

```python
r = await bot.answer_guest_query(gid, result)
...
rs._emitter = InlineEmitter(r.inline_message_id)
```

A guest exchange starts with no address, answers once, receives an
`inline_message_id` from that answer, and is editable from then on. The guest
case is the first state of the inline case.

This is why the two distinctions are drawn differently. Ordinary against inline
is fixed for the whole exchange and `Responder.create` decides it from the
update's shape, so it is a subtype. Guest against inline changes during the
exchange, so it is the `_emitter` field.

Media that has no `InputMedia` type can only reach an inline message through
that first answer, because every later write is an edit. A voice message is the
case that matters, so a command that produces one has to hold the guest answer
back until the media exists. It declares a `RequireDefer` parameter, and while
`can_defer()` holds, `dispatch.py` runs it under `defer_until`, which buffers
every emission and emits once when the handler returns.

## Editing a previous reply

A reply made with `cached=True`, which `reply_cached` passes, stores the reply's
message id under a key that identifies the *incoming* message. When the user
edits their command, the update arrives again, the same key is computed, and the
stored message is edited in place. The mapping lives in the KV table, so it
survives a restart.

Two things disable that path and force a fresh send. A reply markup that is not
an `InlineKeyboardMarkup` cannot be edited in, and neither can a media payload
that has no `InputMedia` type. The second is why sending a voice message always
starts a new message.

## Edit handles

A `Responder` is created per update and is meaningless once the update is
handled. A background task started during that update keeps writing long
afterwards, which is what `examples/redirect_hook.m` does when it chains
`edit_message(...).then(w)`. The position being written has to outlive the
responder, so it is a separate object that the caller keeps.

`InlineFragmentHandle` holds a responder and an index into `_fragments`, so a
handle can name a position in output that has not been sent yet. Editing it
rewrites one fragment and re-emits the whole message. This keeps the
reply-then-edit contract true where the API allows one answer.

`EditHandle.as_responder` converts back, so that output can become the
destination of the next output. `RenderContext._reply_to_rs` is the only caller:
a document that sends a new message must attach it to the message the document
rendered, because replying through the original responder would overwrite that
rendered output when the update is a callback on an inline message. A handle
that carries no `Message` converts to `None`, and the caller falls back.

## Media kinds

Where implementations differ in capability, the caller asks before acting, as
with `can_defer()` above and `payload_has_input` for media.

The split between `MediaPayload` and `MediaPayloadWithInput` follows which kinds
have an `InputMedia` class in python-telegram-bot. Photos, documents, videos and
audio have one. Voice has none in the Bot API, and stickers have one in the Bot
API that the library does not support yet. The comment block above
`PhotoPayload` in `payload.py` lists that and the other irregularities in the
media API, and it is the reason one interface cannot be generated over the six
kinds.

An inline result carries media by `file_id` only, so `stage()` sends locally
produced bytes to the staging chat and reads the `file_id` back off the
resulting message.
