import asyncio
import builtins
import os
import re

from typing import TYPE_CHECKING, cast

from telegram import (
    CallbackQuery,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    InlineQuery,
    InlineQueryResultArticle,
    InputTextMessageContent,
    Message,
)
from telegram.constants import KeyboardButtonStyle, ReactionEmoji

from bot import (
    CallbackData,
    EditHandle,
    MessageArg,
    Responder,
    bot,
    callback_query,
    command,
    escape,
    shorten,
    truncate_text,
)
from render_core import Value, to_str

from .ctx import get_context
from .db import db
from .env import CHAN_ID, MAX_TEXT_LENGTH, encode_id
from .log import do_notify, log
from .render_context import OverriddenDict, decode_value, encode_value
from .render_ctx import (
    BUTTON_KEY,
    DOC_OVERRIDE_DIR,
    ENV_PREFIX,
    FlattenSegment,
    MarkupState,
    MessageSpec,
    RenderContext,
    UpdateCallback,
    extract_media,
    get_doc,
    get_env,
    get_env_flag,
)
from .segments import Bold, Element, render_segment
from .text import pre_block
from .utils import get_msg_url

if TYPE_CHECKING:
    from collections.abc import Container, Mapping


# '/' is kept for compatibility, which was used for '-'.
BUTTON_SIGN = '!'
CALLBACK_SIGNS = '-+/' + BUTTON_SIGN
MEMORY_SIGN = '@'
PATH_SIGNS = ':#`'
CALLBACK_SPECIAL = frozenset(CALLBACK_SIGNS + MEMORY_SIGN + PATH_SIGNS)

SPECIAL_FLAG_ICONS = {'_pre': '📋', '_fold': '💌'}


def is_safe_key(key: str | None) -> bool:
    return bool(key) and all(c not in CALLBACK_SPECIAL for c in key)


def is_safe_mem(mem: str | None) -> bool:
    return bool(mem) and all(c not in PATH_SIGNS for c in mem)


def encode_flags(flags: dict[str, bool] | None) -> str:
    if not flags:
        return ''
    return ''.join((CALLBACK_SIGNS[v] + k) for k, v in flags.items())


MEMORY_KEY = '_mem'
SPECIAL_KEYS = (MEMORY_KEY, BUTTON_KEY)
BUTTON_PREFIX = ENV_PREFIX + 'btn.'
ICON_PREFIX = ENV_PREFIX + 'icon.'


def make_markup(
    path: str,
    ctx: Mapping[str, Value],
    current_state: MarkupState | None,
    doc_ids: dict[int, str] | None,
    ext_btn: tuple[str, str] | None,
) -> tuple[InlineKeyboardMarkup | None, str | None]:
    # query data:
    #   ('-'|'+' <flag-key>)*
    #   ['!' <btn-key>]
    #   ['@' <memory>]
    #   <path-header: ':'|'#'|'`'> <path-body>
    # where '-' means 0, '+' means 1

    if current_state is None:
        state = current_data = None
    else:
        state, current_data = current_state

    size = len(path.encode('utf-8'))
    if state:
        size += sum(len(k.encode('utf-8')) for k in state) + len(state)

    if size > InlineKeyboardButton.MAX_CALLBACK_DATA:
        return None, None

    memory = ''
    if (val := ctx.get(MEMORY_KEY)) is not None and is_safe_mem(payload := encode_value(val)):
        delta = len(payload.encode('utf-8')) + 1
        if size + delta <= InlineKeyboardButton.MAX_CALLBACK_DATA:
            memory = '@' + payload
            size += delta

    # `state`, `memory` and `ctx[BUTTON_KEY]` defines the current state, while
    # `flags` and `buttons` defines the potential next states.
    flags: dict[str, bool] = {}
    buttons = []
    for k, v in ctx.items():
        if len(btn_key := k.removeprefix(BUTTON_PREFIX)) != len(k):
            if (
                is_safe_key(btn_key)
                and (v and v != '0')
                and (
                    size + len(btn_key.encode('utf-8')) + 1
                    <= InlineKeyboardButton.MAX_CALLBACK_DATA
                )
            ):
                buttons.append(btn_key)
            continue

        if k.startswith(ENV_PREFIX) or k in SPECIAL_KEYS or not is_safe_key(k):
            continue

        if v == '0' or v == 0:  # This covers `False` as well.
            v = False
        elif v == '1' or v == 1:
            v = True
        else:
            continue

        if size + len(k.encode('utf-8')) + 1 <= InlineKeyboardButton.MAX_CALLBACK_DATA:
            flags[k] = v

    log.debug('make_markup: flags %s state %s memory %r', flags, state, memory)

    # The external state takes precedence even over `ctx.overrides`, but the existing
    # buttons should occur before in the order.
    if state:
        flags.update(state)

    if doc_ids and not get_env_flag(ctx, 'ref', True):
        doc_ids = None

    if current_data == path:
        current_data = None

    row: list[InlineKeyboardButton | None] = [None]

    def push_button(name: str, key: str):
        data = encode_flags(state) + BUTTON_SIGN + key + memory + path
        row.append(
            InlineKeyboardButton(
                name,
                callback_data=data,
                style=KeyboardButtonStyle.DANGER if key == '_cancel' else None,
            )
        )

    def push_flag(label: str, old: bool, v: bool):
        name = label + (':=' if old else '=') + '01'[v]
        data = encode_flags(state) + memory + path
        row.append(
            InlineKeyboardButton(
                name,
                callback_data=data,
                style=(
                    (KeyboardButtonStyle.SUCCESS if v else KeyboardButtonStyle.DANGER)
                    if old
                    else None
                ),
            )
        )

    if ext_btn is not None:
        push_button(*ext_btn)

    for k in buttons:
        label = str(icon) if (icon := get_env(ctx, 'icon.' + k)) else k
        push_button(label, k)

    if flags:
        if state is None:
            state = {}

        hide_flags = get_env_flag(ctx, 'hide_flags')
        for k, v in flags.items():
            old = k in state
            log.debug('make_markup: flag %s in=%s new=%s s=%s', k, old, v, state)
            state[k] = not v

            if (icon := get_env(ctx, 'icon.' + k)) is not None:
                if label := to_str(icon):
                    push_flag(label, old, v)
                else:
                    log.debug('make_markup: hiding flag %s due to empty icon', k)
            elif not hide_flags:
                label = SPECIAL_FLAG_ICONS.get(k, k)
                push_flag(label, old, v)
            else:
                log.debug('make_markup: hiding flag %s', k)

            if old:
                state[k] = v
            else:
                del state[k]

    log.debug('make_markup: final state %s (%s)', state, current_data)

    if current_data:
        row.append(
            InlineKeyboardButton(
                '🔄',
                callback_data=encode_flags(state) + memory + path,
                style=KeyboardButtonStyle.PRIMARY,
            )
        )

    if doc_ids:
        if len(doc_ids) == 1:
            doc_id = next(iter(doc_ids))
            row.append(
                InlineKeyboardButton('🔗', get_msg_url(doc_id), style=KeyboardButtonStyle.SUCCESS)
            )
        else:
            for doc_id, doc_name in doc_ids.items():
                row.append(
                    InlineKeyboardButton(
                        f'🔗{doc_name}', get_msg_url(doc_id), style=KeyboardButtonStyle.SUCCESS
                    )
                )

    if len(row) <= 1:
        return None, None

    row[0] = InlineKeyboardButton(
        ('⏪ ' if current_data else '🔄 ') + shorten(path[1:]),
        callback_data=path,
        style=(KeyboardButtonStyle.DANGER if current_data else KeyboardButtonStyle.PRIMARY),
    )
    row_ = cast(list[InlineKeyboardButton], row)

    row_limit_ = get_env(ctx, 'btns_per_row', 5)
    try:
        row_limit = int(row_limit_) if isinstance(row_limit_, (str, int, float)) else 5
    except ValueError, TypeError:
        row_limit = 5

    # Group 5 buttons per row
    rows = [row_[i : i + row_limit] for i in range(0, len(row), row_limit)]
    return InlineKeyboardMarkup(rows), current_data


def create_data(overrides: dict[str, Value]) -> OverriddenDict:
    source = None
    context = get_context()

    # Existing `overrides` are frozen and immutable in `Engine`, so this is safe.
    update = context.update
    if (user := update.effective_user) is not None:
        overrides['_user_id'] = user.id
        overrides['_user_name'] = source = user.full_name
    if (chat := update.effective_chat) is not None:
        overrides['_chat_id'] = chat.id
        if title := chat.title:
            overrides['_chat_title'] = title
            source = f'{title} @ {source}' if source else title
    if source:
        overrides['_source'] = source
    if (msg := update.effective_message) is not None:
        overrides['_msg_id'] = msg.message_id

    log.info('create_data: %s', overrides)

    return OverriddenDict({}, overrides)


@command(public=True)
def handle_render(msg: Message, rs: Responder, arg: MessageArg):
    target = msg.reply_to_message
    text = target and (target.text or target.caption or '').strip()

    if text:
        if arg:
            text = arg + '\n' + text
    elif arg:
        text = arg
    else:
        return rs.reply_cached('Specify text or reply to a message to render.')

    if doc_ref := is_doc_ref(text):
        path, row = doc_ref
        if path is None:
            return rs.reply_cached(f'No doc: {row}')
        assert isinstance(row, tuple)
        doc_id, text = row
    else:
        chat_prefix = encode_id(msg.chat_id, '')
        path = f'#{chat_prefix}{msg.message_id}'
        db['r-' + path] = text
        doc_id = None

    data = create_data({})
    ctx = RenderContext(
        data=data,
        doc_id=doc_id,
        path=path,
        responder=rs,
        update_callback=create_reply_callback(rs, data),
    )
    return ctx.render(text)


def create_reply_callback(rs: Responder, data: Mapping[str, Value]) -> UpdateCallback:
    def do_reply(spec: MessageSpec):
        return rs.reply_cached(*spec, media=extract_media(data), allow_not_modified=True)

    return do_reply


REG_DOC_REF = re.compile(r'[*:]\s*(\w+)\s*;')


def is_doc_ref(text: str) -> tuple[str, tuple[int | None, str]] | tuple[None, str] | None:
    '''Returns (path, (doc_id | None, text)), (None, doc_name), or None.'''
    if m := REG_DOC_REF.fullmatch(text):
        doc_name = m[1]
        row = get_doc(doc_name)
        log.info('handle_render: doc ref: %s %s', text, row)
        if row is None:
            return None, doc_name
        return ':' + doc_name, row


preview_cache: dict[int, tuple[RenderContext, str, FlattenSegment]] = {}


async def handle_render_group(rs: Responder, origin_id: int):
    msg_id = rs.get_message().message_id
    if cache := preview_cache.pop(origin_id, None):
        ctx, doc_name, result = cache
        log.info('Doc in group: %s -> %s %s', msg_id, origin_id, doc_name)

        ctx.set_path(':' + doc_name)
        ctx.set_update_callback(create_reply_callback(rs, ctx.data))
        await ctx.to_response(result)
    else:
        log.info('No preview cache in group: %s -> %s', msg_id, origin_id)


async def handle_render_inline_query(query: InlineQuery, text: str):
    if doc_ref := is_doc_ref(text):
        path, row = doc_ref
        if path is None:
            assert isinstance(row, str)
            msg = 'No doc: ' + row
            r = InlineQueryResultArticle(
                id='0', title=msg, input_message_content=InputTextMessageContent(msg)
            )
            await query.answer((r,))
            return
        assert isinstance(row, tuple)
        doc_id, text = row
    elif len(text) + 2 < InlineKeyboardButton.MAX_CALLBACK_DATA:
        path = '`' + text
        doc_id = None
    else:
        doc_id = path = None

    ctx = RenderContext(create_data({}), doc_id=doc_id, path=path)
    ctx.data['_env.footer'] = Element((Bold('via '), '@', bot.username, ' ', text))
    rendered = ctx.render_text(text)
    result, parse_mode, markup = await ctx.to_response(rendered)

    if rendered := render_segment(rendered).strip():
        p = 50
        left_chars = len(rendered) - p * 2
        title = (
            (f'{rendered[:p]} ...[{left_chars} chars]... \\{rendered[-p:]}')
            if left_chars > 0
            else rendered
        )
    else:
        title = '[empty]'

    r = InlineQueryResultArticle(
        id='noop',
        title=title,
        input_message_content=InputTextMessageContent(result, parse_mode=parse_mode),
        reply_markup=markup,
    )
    await query.answer((r,))


@callback_query(filter=lambda data: data[0] in CALLBACK_SPECIAL, public=True)
def handle_render_callback(callback: CallbackQuery, data: CallbackData, rs: Responder):
    flags = {}
    clicked_button = None

    j = 0
    sign = ''

    def take_until(i: int, delims: Container[str]) -> str:
        nonlocal j
        j = i + 1
        while j < len(data) and data[j] not in delims:
            j += 1
        return data[i + 1 : j]

    true = CALLBACK_SIGNS[True]
    while (i := j) < len(data) and (sign := data[i]) in CALLBACK_SIGNS:
        key = take_until(i, CALLBACK_SPECIAL)
        if sign == BUTTON_SIGN:
            log.debug('handle_render_callback: button %r', clicked_button)
            if clicked_button is not None:
                raise ValueError('handle_render_callback: multiple buttons: ' + data)
            clicked_button = key
        else:
            flags[key] = sign == true

    if sign == MEMORY_SIGN:
        memory = take_until(i, PATH_SIGNS)
        i = j
        log.debug('handle_render_callback: memory %r', memory)
    else:
        memory = None

    if len(path := data[i:]) < 2:
        raise ValueError(f'bad path in render callback: {data!r}, {path!r}')

    # Compatibility with old format
    if path[0] == ':' and path[1] in CALLBACK_SPECIAL:
        path = path[1:]

    doc_id = None
    match path[0]:
        case '`':
            text = path[1:]
        case '#':
            text = db.get('r-' + path)
            if text is None:
                raise ValueError('unknown msg in render callback: ' + path)
        case ':':
            row = get_doc(path[1:])
            if row is None:
                raise ValueError('unknown doc in render callback: ' + path)
            doc_id, text = row
        case _:
            raise ValueError('bad render callback: ' + data)

    inner = create_data(dict(flags))
    if clicked_button is not None:
        inner[BUTTON_KEY] = clicked_button
    if memory is not None:
        inner[MEMORY_KEY] = decode_value(memory)
    inner['_state'] = data
    handle = rs.as_edit_handle()
    ctx = RenderContext(
        data=inner,
        markup_state=(flags, data),
        doc_id=doc_id,
        path=path,
        as_caption=handle.as_caption if handle is not None else False,
        responder=rs,
        update_callback=(
            create_callback_query_callback(handle, callback, inner) if handle is not None else None
        ),
    )
    return ctx.render(text)


def create_callback_query_callback(
    handle: EditHandle, callback: CallbackQuery, data: Mapping[str, Value]
) -> UpdateCallback:
    answered = False

    async def edit_callback_message(spec: MessageSpec):
        nonlocal answered
        await handle.edit(*spec, media=extract_media(data), allow_not_modified=True)

        if (not answered) and (answer := get_env(data, 'answer')):
            answered = True
            answer = to_str(answer)
            if len(answer) > CallbackQuery.MAX_ANSWER_TEXT_LENGTH:
                answer = answer[: CallbackQuery.MAX_ANSWER_TEXT_LENGTH - 1] + '…'

            show_alert = get_env_flag(data, 'answer_alert')
            log.info('handle_render_callback: answer: %s, %s', answer, show_alert)
            await callback.answer(answer, show_alert=show_alert)

        return handle

    return edit_callback_message


def _report(out: list[str], action: str, id: int | None, name: str | None, text: str | None):
    parts = []
    if id is not None:
        parts.append(f'{id}:')

    if name is not None:
        parts.append(name)

    if text is not None:
        short = shorten(text)
        parts.append(f'({short})')
        log_info = ' '.join(parts)
        msg_info = escape(' '.join(parts[:-1])) + rf' \(`{escape(short)}`\)'
    else:
        log_info = msg_info = ' '.join(parts)

    log.info('%s %s', action, log_info)

    if id is not None:
        msg_info = f'[{msg_info}]({get_msg_url(id)})'

    out.append(f'{escape(action)} {msg_info}')


def cleanup_preview_cache(doc_id: int, val_id: int, name: str):
    if id(preview_cache.get(doc_id)) == val_id:
        log.warning('Unused preview cache for doc: %s %s', doc_id, name)
        del preview_cache[doc_id]


async def handle_render_doc(msg: Message):
    if not (text := msg.text) or not (text := text.strip()):
        return None

    id = msg.message_id
    ctx = RenderContext(create_data({}), doc_id=id)
    result = ctx.render_text(text)

    info = []
    if name := ctx.engine.doc_name:
        preview_cache[id] = t = (ctx, name, result)
        asyncio.get_event_loop().call_later(30, cleanup_preview_cache, id, builtins.id(t), name)

        old_by_id, old_by_name = db.save_doc(id, name, text)

        action = 'new doc:'
        if old_by_id:
            old_name, old_text = old_by_id
            if old_name == name:
                if old_text == text:
                    old_text = None
                _report(info, 'updated doc:', None, None, old_text)
            else:
                _report(info, 'renamed doc:', None, old_name, old_text)
            action = '->'

        if old_by_name:
            old_id, old_text = old_by_name
            if old_text == text:
                old_text = None
            _report(info, 'relinked doc:', old_id, None, old_text)
            assert old_id != id
            action = '->'

        _report(info, action, id, name, text)
        set_reaction = msg.set_reaction(ReactionEmoji.RED_HEART, True)
    elif old_row := db.delete_doc(id):
        _report(info, 'deleted doc:', id, *old_row)
        set_reaction = msg.set_reaction()
    else:
        return None

    res = truncate_text(('\n' if len(info) > 2 else ' ').join(info))
    await asyncio.gather(do_notify(res, 'MarkdownV2', quiet=True), set_reaction)
    return name


@command
def handle_ls(rs: Responder, arg: MessageArg):
    keywords = arg.split()

    if not (docs := tuple(db.find_docs(keywords))):
        return rs.reply_cached('No docs found.')

    lines = [rf'{len(docs)} docs:']
    for id, name, length in docs:
        line = rf'\- [*{escape(name)}*]({get_msg_url(id)}) \({id}, {length}\)'
        lines.append(line)

    return rs.reply_cached('\n'.join(lines), parse_mode='MarkdownV2')


@command
async def handle_rm(rs: Responder, arg: MessageArg):
    names = arg.split()
    if not names:
        return await rs.reply_cached('Usage: /rm <name1> [name2 ...]')
    db.delete_docs(names)
    return await rs.reply_cached(f'Deleted {len(names)} docs.')


@command
async def handle_submit(rs: Responder, arg: MessageArg):
    if not DOC_OVERRIDE_DIR:
        return await rs.reply_cached('DOC_OVERRIDE_DIR is unset.')

    if os.path.basename(arg) != arg:
        return await rs.reply_cached('Bad name.')

    file = os.path.join(DOC_OVERRIDE_DIR, arg + '.m')
    with open(file, encoding='utf-8') as fp:
        text = fp.read()

    length = len(text)
    if length > MAX_TEXT_LENGTH:
        return await rs.reply_cached(f'Too long: {length}')

    m = await bot.send_message(CHAN_ID, *pre_block(text))
    name = await handle_render_doc(m) or arg

    old_url = get_msg_url(r[0]) if (r := db.get_doc(name)) is not None else None

    dst = file + '.old'
    if os.path.exists(dst):
        n = 1
        while os.path.exists(dst := f'{file}.old.{n}'):
            n += 1

    os.rename(file, dst)
    log.info('Renamed %s -> %s', file, dst)

    info = f'{length} chars ({name})\nNew: {get_msg_url(m.message_id)}'
    if old_url is not None:
        info = f'{info}\nOld: {old_url}'

    await rs.reply_cached(info)
