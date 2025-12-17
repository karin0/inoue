import re
import asyncio
from typing import Mapping

from telegram import (
    InlineQuery,
    InlineQueryResultArticle,
    InputTextMessageContent,
    Update,
    Message,
    InlineKeyboardMarkup,
    InlineKeyboardButton,
)
from telegram.ext import ContextTypes
from telegram.error import BadRequest
from telegram.constants import ReactionEmoji

from util import (
    MAX_TEXT_LENGTH,
    log,
    get_msg_arg,
    get_msg_url,
    reply_text,
    try_send_text,
    shorten,
    truncate_text,
    escape,
    html_escape,
    pre_block,
    blockquote_block,
    do_notify,
    encode_chat_id,
)
from db import db
from render_core import Engine, Value

# '/' is kept for compatibility, which was used for '-'.
CALLBACK_SIGNS = '-+/'
ALL_CALLBACK_SIGNS = frozenset('-+/:#`')

SPECIAL_FLAG_ICONS = {
    '_pre': '📋',
    '_fold': '💌',
}


def encode_flags(flags: dict[str, bool]) -> str:
    return ''.join((CALLBACK_SIGNS[v] + k) for k, v in flags.items())


CTX_ENV_PREFIX = '_env.'


def get_ctx(
    ctx: Engine | Mapping[str, Value], key: str, default: Value | None = None
) -> Value | None:
    v = ctx.get(CTX_ENV_PREFIX + key)
    if v is None:
        v = ctx.get('_' + key)
        if v is None:
            return default
    return v


def get_ctx_flag(
    ctx: Engine | Mapping[str, Value], key: str, default: bool = False
) -> bool | Value:
    v = get_ctx(ctx, key)
    if v is None:
        return default
    return v and v != '0'


def make_markup(
    path: str | None, ctx: Engine | Mapping[str, Value], state: dict[str, bool] | None
) -> InlineKeyboardMarkup | None:
    if not path:
        return None

    # query data: ('-'|'+' <flag-key>)* <path-header: ':'|'#'|'`'> <path-body>
    # where '-' means 0, '+' means 1

    size = len(path.encode('utf-8'))
    if state:
        size += sum(len(k.encode('utf-8')) for k in state.keys()) + len(state)

    if size > InlineKeyboardButton.MAX_CALLBACK_DATA:
        return None

    flags = {}
    for k, v in ctx.items():
        if k.startswith(CTX_ENV_PREFIX):
            continue

        if v == '0' or v == 0:  # This covers `False` as well.
            v = False
        elif v == '1' or v == 1:
            v = True
        else:
            continue

        if size + len(k.encode('utf-8')) + 1 <= InlineKeyboardButton.MAX_CALLBACK_DATA:
            flags[k] = v

    log.debug('make_markup: flags %s state %s', flags, state)

    # The external state takes precedence even over `ctx.overrides`.
    if state:
        flags.update(state)

    if (doc_id := ctx.first_doc_id) is None and not flags:
        return None

    row = [
        InlineKeyboardButton(
            ('⏪ ' if state else '🔄 ') + shorten(path[1:]), callback_data=path
        )
    ]

    if flags:
        if state is None:
            state = {}

        def push_button(name: str):
            data = encode_flags(state) + path
            row.append(InlineKeyboardButton(name, callback_data=data))

        hide_flags = get_ctx_flag(ctx, 'hide_flags')
        for k, v in flags.items():
            if not v:
                # Hide conflicting options.
                if (
                    k == '_pre'
                    and get_ctx_flag(ctx, 'fold')
                    or k == '_fold'
                    and get_ctx_flag(ctx, 'pre', True)
                ):
                    log.debug('make_markup: hiding conflicting flag %s', k)
                    continue

            old = k in state
            log.debug('make_markup: flag %s in=%s new=%s s=%s', k, old, v, state)
            state[k] = not v

            if icon := get_ctx(ctx, 'icon.' + k):
                icon = str(icon)
                push_button(icon + (':=' if old else '=') + '01'[v])
            elif not hide_flags:
                icon = SPECIAL_FLAG_ICONS.get(k, k)
                push_button(icon + (':=' if old else '=') + '01'[v])
            else:
                log.debug('make_markup: hiding flag %s', k)

            if old:
                state[k] = v
            else:
                del state[k]

        log.debug('make_markup: final state %s', state)
        if state:
            push_button('🔄')

    if doc_id:
        row.append(InlineKeyboardButton('🔗', get_msg_url(doc_id)))

    row_limit = get_ctx(ctx, 'btns_per_row', 5)
    try:
        row_limit = int(row_limit)
    except ValueError:
        row_limit = 5

    # Group 5 buttons per row
    rows = [row[i : i + row_limit] for i in range(0, len(row), row_limit)]

    return InlineKeyboardMarkup(rows)


reg_cleanup = re.compile(r'\n{3,}')


def rendered_response(
    flags: dict[str, bool] | None,
    path: str | None,
    result: str,
    ctx: Engine,
) -> tuple[str, str | None, InlineKeyboardMarkup | None]:
    result = result or '[empty]'
    if errors := ctx.errors:
        result += f'\n\n---\n\n' + '\n'.join(errors)

    result = truncate_text(result)
    parse_mode = None

    if get_ctx_flag(ctx, 'cleanup', True):
        result = reg_cleanup.sub('\n\n', result)

    if not errors:
        if get_ctx_flag(ctx, 'md'):
            parse_mode = 'MarkdownV2'
            do_escape = escape
        elif get_ctx_flag(ctx, 'html'):
            parse_mode = 'HTML'
            do_escape = html_escape
        else:
            if get_ctx_flag(ctx, 'fold'):
                result, parse_mode = blockquote_block(result)
                do_escape = html_escape
            elif get_ctx_flag(ctx, 'pre', True):
                result, parse_mode = pre_block(result)
                do_escape = escape

        if parse_mode:
            if flags:
                suffix = do_escape(encode_flags(flags))
                if len(result) + len(suffix) <= MAX_TEXT_LENGTH:
                    result += suffix

            if footer := get_ctx(ctx, 'footer'):
                suffix = do_escape(footer)
                if flags:
                    suffix = ' ' + suffix
                if len(result) + len(suffix) <= MAX_TEXT_LENGTH:
                    result += suffix

    return result, parse_mode, make_markup(path, ctx, flags)


ENGINE_FUNCS = {'escape': escape, 'html_escape': html_escape}


def create_engine(
    update: Update,
    flags: dict[str, bool] | None = None,
    *,
    doc_id: int | None = None,
) -> Engine:
    overrides: dict[str, Value] = dict(flags) if flags is not None else {}
    source = None
    if user := update.effective_user:
        overrides['_user_id'] = str(user.id)
        overrides['_user_name'] = source = user.full_name
    if chat := update.effective_chat:
        overrides['_chat_id'] = str(chat.id)
        if title := chat.title:
            overrides['_chat_title'] = title
            source = f'{title} @ {source}' if source else title
    if source:
        overrides['_source'] = source
    if msg := update.effective_message:
        overrides['_msg_id'] = str(msg.message_id)
    log.debug('create_engine: overrides %s', overrides)
    return Engine(overrides=overrides, funcs=ENGINE_FUNCS, doc_id=doc_id)


def render_text(
    ctx: Engine,
    text: str,
    flags: dict[str, bool] | None = None,
    path: str | None = None,
) -> tuple[str, str | None, InlineKeyboardMarkup | None]:
    result = ctx.render(text)
    log.info('rendered %d -> %d', len(text), len(result))

    if path is None and len(text) + 2 < InlineKeyboardButton.MAX_CALLBACK_DATA:
        path = '`' + text

    return rendered_response(flags, path, result, ctx)


async def handle_render(update: Update, _ctx: ContextTypes.DEFAULT_TYPE):
    msg, arg = get_msg_arg(update)
    target = msg.reply_to_message

    if target:
        text = (target.text or target.caption).strip()
        if not text:
            return await reply_text(msg, 'No text to render.')

        if arg:
            # Include context set by arg
            text = arg + '\n' + text
    elif arg:
        text = arg
    else:
        return await reply_text(msg, 'Specify text or reply to a message to render.')

    chat_prefix = encode_chat_id(msg, '')
    path = f'#{chat_prefix}{msg.message_id}'
    db['r-' + path] = text
    res = render_text(create_engine(update), text, path=path)
    return await reply_text(msg, *res, allow_not_modified=True)


preview_cache = {}


async def handle_render_group(msg: Message, origin_id: int):
    if cache := preview_cache.pop(origin_id, None):
        doc_name, *args = cache
        log.info('Doc in group: %s -> %s %s', msg.id, origin_id, doc_name)
        args = rendered_response(None, ':' + doc_name, *args)
        await reply_text(msg, *args, allow_not_modified=True)

    if left := tuple((id, name) for id, (name, *_) in preview_cache.items()):
        log.warning('Preview cache not empty: %s', left)


async def handle_render_inline_query(update: Update, query: InlineQuery, text: str):
    result, parse_mode, markup = render_text(create_engine(update), text)
    r = InlineQueryResultArticle(
        id='1',
        title=f'Render: {len(result)}',
        input_message_content=InputTextMessageContent(result, parse_mode=parse_mode),
        reply_markup=markup,
    )
    await query.answer((r,))


async def handle_render_callback(
    update: Update, ctx: ContextTypes.DEFAULT_TYPE, data: str
):
    flags = {}
    j = 0
    true = CALLBACK_SIGNS[True]
    while (i := j) < len(data) and (sign := data[i]) in CALLBACK_SIGNS:
        j = i = i + 1
        while j < len(data) and data[j] not in ALL_CALLBACK_SIGNS:
            j += 1
        flags[data[i:j]] = sign == true

    if len(path := data[i:]) <= 2:
        raise ValueError('bad path in render callback: ' + data)

    # Compatibility with old format
    if path[0] == ':' and path[1] in ALL_CALLBACK_SIGNS:
        path = path[1:]

    doc_id = None
    match path[0]:
        case '`':
            data = path[1:]
        case '#':
            data = db.get('r-' + path)
            if data is None:
                # TODO: report when cache expired
                raise ValueError('unknown msg in render callback: ' + path)
        case ':':
            row = db.get_doc(path[1:])
            if row is None:
                # TODO: report when doc deleted
                raise ValueError('unknown doc in render callback: ' + path)
            doc_id, data = row
        case _:
            raise ValueError('bad render callback: ' + data)

    engine = create_engine(update, flags, doc_id=doc_id)
    result, parse_mode, markup = render_text(engine, data, flags, path)

    query = update.callback_query
    assert query is not None

    try:
        if query.inline_message_id:
            await try_send_text(
                ctx.bot.edit_message_text,
                result,
                inline_message_id=query.inline_message_id,
                reply_markup=markup,
                parse_mode=parse_mode,
            )
        else:
            await try_send_text(
                ctx.bot.edit_message_text,
                result,
                chat_id=update.effective_chat.id,
                message_id=query.message.message_id,
                reply_markup=markup,
                parse_mode=parse_mode,
            )
    except BadRequest as e:
        if 'Message is not modified' not in str(e):
            raise


def _report(
    out: list[str], action: str, id: int | None, name: str | None, text: str | None
):
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


async def handle_render_doc(update: Update, msg: Message):
    if not (text := msg.text) or not (text := text.strip()):
        return

    id = msg.message_id
    ctx = create_engine(update, doc_id=id)
    rendered = ctx.render(text)

    info = []
    if name := ctx.doc_name:
        preview_cache[id] = (name, rendered, ctx)
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
        reaction = (ReactionEmoji.RED_HEART, True)
    elif old_row := db.delete_doc(id):
        _report(info, 'deleted doc:', id, *old_row)
        reaction = ()
    else:
        return

    res = truncate_text(('\n' if len(info) > 2 else ' ').join(info))
    await asyncio.gather(
        do_notify(res, 'MarkdownV2', quiet=True),
        msg.set_reaction(*reaction),
    )
