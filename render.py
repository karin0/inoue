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
from render_sys import Syscall
from render_core import Engine, Value

# '/' is kept for compatibility, which was used for '-'.
CALLBACK_SIGNS = '-+/'
ALL_CALLBACK_SIGNS = frozenset('-+/:#`@')

SPECIAL_FLAG_ICONS = {
    '_pre': '📋',
    '_fold': '💌',
}


def is_safe_key(key: str) -> bool:
    return all(c not in ALL_CALLBACK_SIGNS for c in key)


def encode_flags(flags: dict[str, bool]) -> str:
    return ''.join((CALLBACK_SIGNS[v] + k) for k, v in flags.items())


CTX_ENV_PREFIX = '_env.'
MEMORY_KEY = 'mem'
MEMORY_KEY_RAW = '_mem'


def encode_type(val: Value) -> str:
    if isinstance(val, bool):
        return 'b' + '01'[val]
    if isinstance(val, int):
        return ('i' + str(val)) if val >= 0 else ('I' + str(-val))
    if isinstance(val, float):
        return ('f' + str(val)) if val >= 0 else ('F' + str(-val))
    return 's' + str(val)


def decode_type(val: str, typ: str) -> Value:
    match typ:
        case 'b':
            return val == '1'
        case 'i':
            return int(val)
        case 'I':
            return -int(val)
        case 'f':
            return float(val)
        case 'F':
            return -float(val)
    return val


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
    path: str | None,
    ctx: Engine,
    state: dict[str, bool] | None,
    doc_id: int | None = None,
) -> tuple[InlineKeyboardMarkup, str | None] | None:
    if not path:
        return None

    # query data: ('-'|'+' <flag-key>)* ['@' <memory>] <path-header: ':'|'#'|'`'> <path-body>
    # where '-' means 0, '+' means 1

    size = len(path.encode('utf-8'))
    if state:
        size += sum(len(k.encode('utf-8')) for k in state.keys()) + len(state)

    if size > InlineKeyboardButton.MAX_CALLBACK_DATA:
        return None

    memory = ''
    if (val := get_ctx(ctx, MEMORY_KEY)) is not None and is_safe_key(
        payload := encode_type(val)
    ):
        delta = len(payload.encode('utf-8')) + 1
        if size + delta <= InlineKeyboardButton.MAX_CALLBACK_DATA:
            memory = '@' + payload
            size += delta

    flags = {}
    for k, v in ctx.items():
        if k.startswith(CTX_ENV_PREFIX) or k == MEMORY_KEY_RAW or not is_safe_key(k):
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

    # The external state takes precedence even over `ctx.overrides`.
    if state:
        flags.update(state)

    has_state = state or memory
    may_have_state = has_state or flags

    if doc_id is not None and not get_ctx_flag(ctx, 'ref', True):
        doc_id = None

    if doc_id is None and not may_have_state:
        return None

    row = [
        InlineKeyboardButton(
            ('⏪ ' if state else '🔄 ') + shorten(path[1:]), callback_data=path
        )
    ]

    ret = None
    if may_have_state:
        if state is None:
            state = {}

        def push_button(name: str):
            data = encode_flags(state) + memory + path
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
        if has_state:
            ret = encode_flags(state) + memory
            row.append(InlineKeyboardButton('🔄', callback_data=ret + path))

    if doc_id:
        row.append(InlineKeyboardButton('🔗', get_msg_url(doc_id)))

    row_limit_ = get_ctx(ctx, 'btns_per_row', 5)
    try:
        row_limit = int(row_limit_)
    except (ValueError, TypeError):
        row_limit = 5

    # Group 5 buttons per row
    rows = [row[i : i + row_limit] for i in range(0, len(row), row_limit)]

    return InlineKeyboardMarkup(rows), ret


reg_cleanup = re.compile(r'\n{3,}')


ENGINE_FUNCS = {
    k: v
    for k in dir(Syscall)
    if not k.startswith('_') and callable(v := getattr(Syscall, k))
}


class RenderContext:
    def __init__(
        self,
        update: Update,
        flags: dict[str, bool] | None = None,
        doc_id: int | None = None,
    ):
        if flags is None:
            flags = {}
        self.flags = flags

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

        self._first_doc_id = None
        self._default_doc_id = doc_id

        def doc_loader(name: str) -> str | None:
            row = db.get_doc(name)
            if row is None:
                return None
            if self._first_doc_id is None:
                self._first_doc_id = row[0]
            return row[1]

        # Access to attributes with underscores should be forbidden in `simpleeval`,
        # so `os` is safe.
        self.engine = Engine(
            {'os': Syscall},
            overrides=overrides,
            doc_loader=doc_loader,
            funcs=ENGINE_FUNCS,
        )

    def render(
        self,
        text: str,
        path: str | None,
    ) -> tuple[str, str | None, InlineKeyboardMarkup | None]:
        result = self.engine.render(text)
        log.info('rendered %d -> %d', len(text), len(result))
        return self.to_response(path, result)

    def to_response(
        self,
        path: str | None,
        result: str,
    ) -> tuple[str, str | None, InlineKeyboardMarkup | None]:
        ctx = self.engine
        result = result or '[empty]'
        if errors := ctx.errors:
            result += f'\n\n---\n\n' + '\n'.join(errors)

        if Syscall._secret in result:
            log.error('dangerous output: %s', result.replace(Syscall._secret, '****'))
            raise ValueError('dangerous output')

        first_doc_id = self._first_doc_id or self._default_doc_id

        result = truncate_text(result)
        parse_mode = None
        if ret := make_markup(path, self.engine, self.flags, first_doc_id):
            markup, current_state = ret
        else:
            markup = current_state = None

        if get_ctx_flag(ctx, 'cleanup', True):
            result = reg_cleanup.sub('\n\n', result)

        if not errors:
            if get_ctx_flag(ctx, 'md'):
                parse_mode = 'MarkdownV2'
                do_escape = escape
            elif get_ctx_flag(ctx, 'html'):
                parse_mode = 'HTML'
                do_escape = html_escape
            elif get_ctx_flag(ctx, 'fold'):
                result, parse_mode = blockquote_block(result)
                do_escape = html_escape
            elif get_ctx_flag(ctx, 'pre', True):
                result, parse_mode = pre_block(result)
                do_escape = escape
            else:
                do_escape = lambda x: x[1 / 0]

            if parse_mode:
                if current_state:
                    suffix = do_escape(current_state)
                    if len(result) + len(suffix) <= MAX_TEXT_LENGTH:
                        result += suffix

                if footer := get_ctx(ctx, 'footer'):
                    suffix = do_escape(str(footer))
                    if current_state:
                        suffix = ' ' + suffix
                    if len(result) + len(suffix) <= MAX_TEXT_LENGTH:
                        result += suffix

        return result, parse_mode, markup


REG_DOC_REF = re.compile(r'[*:]\s*(\w+)\s*;')


async def handle_render(update: Update, _ctx: ContextTypes.DEFAULT_TYPE):
    msg, arg = get_msg_arg(update)
    target = msg.reply_to_message

    if target:
        text = (target.text or target.caption or '').strip()
        if not text:
            return await reply_text(msg, 'No text to render.')

        if arg:
            # Include context set by arg
            text = arg + '\n' + text
    elif arg:
        text = arg
    else:
        return await reply_text(msg, 'Specify text or reply to a message to render.')

    if doc_ref := is_doc_ref(text):
        path, row = doc_ref
        if path is None:
            return await reply_text(msg, f'No doc: {text}')
        doc_id, text = row
    else:
        chat_prefix = encode_chat_id(msg, '')
        path = f'#{chat_prefix}{msg.message_id}'
        db['r-' + path] = text
        doc_id = None

    ctx = RenderContext(update, doc_id=doc_id)
    res = ctx.render(text, path)
    return await reply_text(msg, *res, allow_not_modified=True)


def is_doc_ref(text: str) -> tuple[str, tuple[int, str]] | tuple[None, str] | None:
    if m := REG_DOC_REF.fullmatch(text):
        doc_name = m[1]
        row = db.get_doc(doc_name)
        log.info('handle_render: doc ref: %s %s', text, row)
        if row is None:
            return None, doc_name
        return ':' + doc_name, row


preview_cache = {}


async def handle_render_group(msg: Message, origin_id: int):
    if cache := preview_cache.pop(origin_id, None):
        ctx, doc_name, result = cache
        log.info('Doc in group: %s -> %s %s', msg.id, origin_id, doc_name)
        args = ctx.to_response(':' + doc_name, result)
        await reply_text(msg, *args, allow_not_modified=True)

    if left := tuple((id, name) for id, (name, *_) in preview_cache.items()):
        log.warning('Preview cache not empty: %s', left)


async def handle_render_inline_query(update: Update, query: InlineQuery, text: str):
    if doc_ref := is_doc_ref(text):
        path, row = doc_ref
        if path is None:
            msg = 'No doc: ' + row
            r = InlineQueryResultArticle(
                id='0',
                title=msg,
                input_message_content=InputTextMessageContent(msg),
            )
            return await query.answer((r,))
        doc_id, text = row
    elif len(text) + 2 < InlineKeyboardButton.MAX_CALLBACK_DATA:
        path = '`' + text
        doc_id = None
    else:
        doc_id = path = None

    ctx = RenderContext(update, doc_id=doc_id)
    result, parse_mode, markup = ctx.render(text, path)
    r = InlineQueryResultArticle(
        id='1',
        title=f'Render: {len(result)}',
        input_message_content=InputTextMessageContent(result, parse_mode=parse_mode),
        reply_markup=markup,
    )
    await query.answer((r,))


async def handle_render_callback(
    update: Update, ctx_: ContextTypes.DEFAULT_TYPE, data: str
):
    flags = {}
    j = 0
    true = CALLBACK_SIGNS[True]
    while (i := j) < len(data) and (sign := data[i]) in CALLBACK_SIGNS:
        j = i = i + 1
        while j < len(data) and data[j] not in ALL_CALLBACK_SIGNS:
            j += 1
        flags[data[i:j]] = sign == true

    if sign == '@':
        j = i = i + 1
        while i < len(data) and data[i] not in ALL_CALLBACK_SIGNS:
            i += 1
        memory = data[j:i]
        log.debug('handle_render_callback: memory %r', memory)
    else:
        memory = None

    if len(path := data[i:]) <= 2:
        raise ValueError('bad path in render callback: ' + data)

    # Compatibility with old format
    if path[0] == ':' and path[1] in ALL_CALLBACK_SIGNS:
        path = path[1:]

    doc_id = None
    match path[0]:
        case '`':
            text = path[1:]
        case '#':
            text = db.get('r-' + path)
            if text is None:
                # TODO: report when cache expired
                raise ValueError('unknown msg in render callback: ' + path)
        case ':':
            row = db.get_doc(path[1:])
            if row is None:
                # TODO: report when doc deleted
                raise ValueError('unknown doc in render callback: ' + path)
            doc_id, text = row
        case _:
            raise ValueError('bad render callback: ' + data)

    ctx = RenderContext(update, flags, doc_id=doc_id)
    if memory is not None:
        ctx.engine[MEMORY_KEY_RAW] = decode_type(memory[1:], memory[0])
    ctx.engine['_state'] = data
    result, parse_mode, markup = ctx.render(text, path)

    query = update.callback_query
    assert query is not None

    try:
        if query.inline_message_id:
            await try_send_text(
                ctx_.bot.edit_message_text,
                result,
                inline_message_id=query.inline_message_id,
                reply_markup=markup,
                parse_mode=parse_mode,
            )
        else:
            await try_send_text(
                ctx_.bot.edit_message_text,
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
    ctx = RenderContext(update, doc_id=id)
    result = ctx.engine.render(text)

    info = []
    if name := ctx.engine.doc_name:
        preview_cache[id] = (ctx, name, result)
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
