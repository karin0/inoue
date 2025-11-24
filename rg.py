import os
import asyncio
import dataclasses
import json
import mmap
import functools
from typing import Iterable, Sequence
from subprocess import DEVNULL, PIPE

from telegram import (
    MessageEntity,
    Update,
    Message,
    InlineKeyboardMarkup,
    InlineKeyboardButton,
)
from telegram.ext import ContextTypes
from telegram.constants import MessageEntityType

from util import (
    escape,
    get_msg_arg,
    truncate_text,
    reply_text,
    log,
    notify,
    BOT_NAME,
    MAX_TEXT_LENGTH,
)
from motto import greeting

type Segment = Sequence['Segment'] | str | 'Style' | 'Link'


@dataclasses.dataclass
class Style:
    inner: Segment
    type: MessageEntityType


@dataclasses.dataclass
class Link:
    inner: Segment
    url: str


def get_length(s: Segment) -> int:
    if isinstance(s, Style) or isinstance(s, Link):
        return get_length(s.inner)
    elif isinstance(s, str):
        return len(s)
    else:
        return sum(get_length(item) for item in s)


Bold = functools.partial(Style, type=MessageEntityType.BOLD)
Underline = functools.partial(Style, type=MessageEntityType.UNDERLINE)


class LengthExceeded(Exception):
    pass


class Formatter:
    def __init__(self):
        self.segments: list[Segment] = []
        self.length = 0
        self.saved_idx = None
        self.saved_length = None

    def append(self, seg: Segment):
        new_len = self.length + get_length(seg)
        if new_len > MAX_TEXT_LENGTH:
            raise LengthExceeded()
        self.segments.append(seg)
        self.length = new_len

    def __enter__(self):
        # This is only used in `RGFile.render_from`, so no need to be reentrant
        # with stacks.
        self.saved_idx = len(self.segments)
        self.saved_length = self.length
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type:
            self.segments[self.saved_idx :] = []
            self.length = self.saved_length
            self.saved_idx = self.saved_length = None
            if (t := MAX_TEXT_LENGTH - self.length) > 0:
                self.append('[truncated]'[:t])


# >>> 2026-01-02 15:04:05 (1)
SECTION_SEP = b'>>> 202'
SECTION_SEP_OFFSET = 2


@dataclasses.dataclass
class RGMatch:
    text: str
    line_number: int
    absolute_offset: int
    match: str
    # start: int
    # end: int

    def show(self, path: str) -> tuple[str, bool]:
        with open(path, 'rb') as fp:
            with mmap.mmap(fp.fileno(), 0, access=mmap.ACCESS_READ) as mm:
                off = self.absolute_offset

                # One character takes up to 4 bytes in UTF-8.
                SECTION_TEXT_LIMIT = MAX_TEXT_LENGTH << 2
                SECTION_FIND_LIMIT = SECTION_TEXT_LIMIT + SECTION_SEP_OFFSET

                bound = max(0, off - SECTION_FIND_LIMIT)
                section_start = mm.rfind(SECTION_SEP, bound, off)

                bound = (
                    off if section_start < 0 else section_start
                ) + SECTION_FIND_LIMIT

                section_end = mm.find(SECTION_SEP, off, bound)

                if section_end < 0:
                    section_end = min(mm.size(), bound)

                if hit := section_start >= 0:
                    section_start += SECTION_SEP_OFFSET
                else:
                    section_start = max(0, section_end - SECTION_TEXT_LIMIT)

                return (
                    mm[section_start:section_end]
                    .decode('utf-8', errors='replace')
                    .strip('�')
                    .strip(),
                    hit,
                )

    def render(self, segments: Formatter, i: int, j: int, k: int):
        kw = self.match
        s = self.text

        # p = len(s.encode('utf-8')[: self.start].decode('utf-8'))
        p = s.index(kw)
        q = p + len(kw)
        l = p - 30
        r = q + 30

        if l > 0:
            if l <= 15:
                pl = s[:l]
            else:
                pl = s[:15] + '...'
        else:
            pl = ''
            r -= l
            l = 0

        if r >= len(s):
            pr = ''
        else:
            pr = '...'

        segments.append(
            Link(
                [
                    str(self.line_number) + ':' + pl + s[l:p],
                    Underline(Bold(kw)),
                    s[q:r] + pr,
                ],
                url=f'{i}_{j}_{k}',
            )
        )
        segments.append('\n')


@dataclasses.dataclass
class RGFile:
    matches: list[RGMatch]
    path: str

    def render(
        self, segments: Formatter, i: int, j: int, match_offset: int
    ) -> Iterable[tuple[int, int]]:
        with segments:
            segments.append(Bold(self.path))
            segments.append('\n')
        for idx, m in enumerate(self.matches[match_offset:]):
            yield (j, idx + match_offset)
            with segments:
                m.render(segments, i, j, idx + match_offset)


@dataclasses.dataclass
class RGQuery:
    files: Sequence[RGFile]
    cwd: str
    match_cnt: int = 0
    page_num: int = 0
    page_offsets: list[tuple[int, int, int]] = dataclasses.field(default_factory=list)
    message: Message | None = None

    def render(
        self, segments: Formatter, i: int, file_offset: int, match_offset: int
    ) -> Iterable[tuple[int, int]]:
        assert file_offset >= 0 and match_offset >= 0
        assert file_offset < len(self.files)
        for idx, f in enumerate(self.files[file_offset:]):
            yield from f.render(
                segments, i, idx + file_offset, match_offset if idx == 0 else 0
            )
        yield (-1, -1)


CWD = os.environ['RG_CWD']
URL_BASE = 'https://t.me/' + BOT_NAME + '?start=rg_'
QUERIES: list[RGQuery] = []
QUERY_LIMIT = 10
QUERY_IDX = 0
MATCH_LIMIT = 300
PAGE_LIMIT = 10


def push_query(q: RGQuery):
    global QUERIES, QUERY_LIMIT, QUERY_IDX
    if len(QUERIES) >= QUERY_LIMIT:
        QUERIES[QUERY_IDX] = q
        r = QUERY_IDX
        QUERY_IDX += 1
        if QUERY_IDX >= QUERY_LIMIT:
            QUERY_IDX = 0
    else:
        r = len(QUERIES)
        QUERIES.append(q)
    return r


def render_segment(seg: Segment, out: bytearray, entities: list[MessageEntity]):
    the_offset = len(out)
    if isinstance(seg, Style):
        offset, length = render_segment(seg.inner, out, entities)
        entities.append(MessageEntity(type=seg.type, offset=offset, length=length))
    elif isinstance(seg, Link):
        offset, length = render_segment(seg.inner, out, entities)
        entities.append(
            MessageEntity(
                type=MessageEntityType.TEXT_LINK,
                offset=offset,
                length=length,
                url=URL_BASE + seg.url,
            )
        )
    elif isinstance(seg, str):
        out.extend(seg.encode('utf-16-le'))
    else:
        for item in seg:
            render_segment(item, out, entities)
    return the_offset >> 1, (len(out) - the_offset) >> 1


async def handle_start(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    msg, arg = get_msg_arg(update)
    if not (arg and arg.startswith('rg_')):
        return await msg.reply_text(greeting(), do_quote=True)

    _, i, j, k = arg.split('_')
    query = QUERIES[int(i)]
    file = query.files[int(j)]
    text, hit = file.matches[int(k)].show(os.path.join(query.cwd, file.path))
    text = truncate_text(text)
    parse_mode = None
    if len(text) + 7 <= MAX_TEXT_LENGTH and hit and (p := text.find('\n')) >= 0:
        header = text[:p].strip()
        body = text[p + 1 :].strip()
        new_text = escape(header) + '\n```' + escape(body) + '```'
        if len(new_text) <= MAX_TEXT_LENGTH:
            text = new_text
            parse_mode = 'MarkdownV2'

    await asyncio.gather(
        msg.delete(),
        query.message.edit_text(
            text,
            reply_markup=InlineKeyboardMarkup.from_button(
                InlineKeyboardButton(
                    text='Back',
                    callback_data=f'rg_back_{i}',
                )
            ),
            parse_mode=parse_mode,
        ),
    )


async def handle_rg_callback(data: str):
    _, cmd, *args = data.split('_')
    match cmd:
        case 'back':
            idx = int(args[0])
            query = QUERIES[idx]
            page_num = query.page_num
        case 'page':
            idx = int(args[0])
            page_num = int(args[1])
            query = QUERIES[idx]
            query.page_num = page_num
        case _:
            raise ValueError('bad rg callback: ' + data)

    if page_num == 0:
        offsets = (0, 0, 0)
    else:
        offsets = query.page_offsets[page_num - 1]

    text, entities, markup = render_query_menu(query, idx, *offsets)
    await query.message.edit_text(text, entities=entities, reply_markup=markup)


async def _run_rg(arg: str, cwd: str) -> RGQuery:
    cmd = ('rg', '-m', str(MATCH_LIMIT), '--sortr', 'path', '--json', arg)
    child = await asyncio.create_subprocess_exec(
        *cmd, stdin=DEVNULL, stdout=PIPE, stderr=DEVNULL, cwd=cwd
    )

    query = RGQuery(files=[], cwd=cwd)
    cnt = 0
    try:
        async for line in child.stdout:
            if line:
                line = json.loads(line)
                match line['type']:
                    case 'begin':
                        file = RGFile(matches=[], path=line['data']['path']['text'])
                        query.files.append(file)
                    case 'match':
                        data = line['data']
                        match = data['submatches'][0]
                        file.matches.append(
                            RGMatch(
                                text=data['lines']['text'].strip(),
                                line_number=data['line_number'],
                                absolute_offset=data['absolute_offset'],
                                match=match['match']['text'],
                                # start=match['start'],
                                # end=match['end'],
                            )
                        )
                        cnt += 1
                        if cnt >= MATCH_LIMIT:
                            break
    finally:
        # child.stdout.feed_eof()
        await child.stdout.read()
        if r := await child.wait():
            with notify.suppress():
                log.warning('rg exited with code %d', r)

    query.match_cnt = cnt
    return query


def render_query_menu(
    query: RGQuery,
    idx: int,
    file_offset: int,
    match_offset: int,
    total_offset: int,
) -> tuple[str, Sequence[MessageEntity]]:
    fmt = Formatter()
    cnt = -1
    try:
        for offset in query.render(fmt, idx, file_offset, match_offset):
            cnt += 1
            if cnt >= PAGE_LIMIT:
                break
    except LengthExceeded:
        pass

    total = total_offset + cnt
    page_num = query.page_num
    if len(query.page_offsets) == page_num:
        r = (*offset, total)
        log.info('rg: page %d for %d: %s', page_num, idx, r)
        query.page_offsets.append(r)

    if total < query.match_cnt or page_num > 0:
        row = []
        if page_num > 0:
            row.append(
                InlineKeyboardButton(
                    text='Prev',
                    callback_data=f'rg_page_{idx}_{page_num - 1}',
                )
            )
        else:
            row.append(
                InlineKeyboardButton(
                    text=' ',
                    callback_data='noop',
                )
            )

        row.append(
            InlineKeyboardButton(
                text=f'{total} / {query.match_cnt} ({page_num})',
                callback_data='noop',
            )
        )

        if total < query.match_cnt:
            row.append(
                InlineKeyboardButton(
                    text='Next', callback_data=f'rg_page_{idx}_{page_num + 1}'
                )
            )
        else:
            row.append(
                InlineKeyboardButton(
                    text=' ',
                    callback_data='noop',
                )
            )

        markup = InlineKeyboardMarkup.from_row(row)
    else:
        markup = None

    ba = bytearray()
    entities: list[MessageEntity] = []
    render_segment(fmt.segments, ba, entities)

    text = ba.decode('utf-16-le')
    if len(text) > MAX_TEXT_LENGTH:
        n = MAX_TEXT_LENGTH - 12
        text = text[:n] + '\n[truncated]'
        final_entities = []
        n = len(text.encode('utf-16-le')) >> 1
        for ent in entities:
            if ent.offset < n:
                if ent.length > n - ent.offset:
                    ent = MessageEntity(
                        type=ent.type,
                        offset=ent.offset,
                        length=n - ent.offset,
                        url=ent.url,
                    )
                final_entities.append(ent)
        entities = final_entities

    return text, entities, markup


async def handle_rg(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    msg, arg = get_msg_arg(update)
    if not arg:
        return await reply_text(msg, 'Provide a keyword.')

    query = await _run_rg(arg, CWD + ('' if msg.text.startswith('/rg') else '2'))

    if not query.files:
        return await reply_text(msg, 'No matches.')

    idx = push_query(query)
    text, entities, markup = render_query_menu(query, idx, 0, 0, 0)
    query.message = await reply_text(msg, text, entities=entities, reply_markup=markup)
