import os
import asyncio
import dataclasses
import json
import mmap
import functools
import itertools
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
    get_deep_link_url,
    truncate_text,
    reply_text,
    log,
    notify,
    MAX_TEXT_LENGTH,
)

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
            assert exc_type in (LengthExceeded, GeneratorExit)
            self.segments[self.saved_idx :] = []
            self.length = self.saved_length
            self.saved_idx = self.saved_length = None
            if exc_type is LengthExceeded and (t := MAX_TEXT_LENGTH - self.length) > 0:
                self.append('[truncated]'[:t])


# >>> 2026-01-02 15:04:05 (1)
SECTION_SEP = b'>>> 202'
SECTION_SEP_OFFSET = 2
SECTION_GAP = 24


@dataclasses.dataclass
class Section:
    start: int
    end: int
    hit: bool

    @staticmethod
    def discover(mm: mmap.mmap, off: int) -> 'Section | None':
        # One character takes up to 4 bytes in UTF-8.
        SECTION_TEXT_LIMIT = MAX_TEXT_LENGTH << 2
        SECTION_FIND_LIMIT = SECTION_TEXT_LIMIT + SECTION_SEP_OFFSET

        bound = max(0, off - SECTION_FIND_LIMIT)
        section_start = mm.rfind(SECTION_SEP, bound, off)

        bound = (off if section_start < 0 else section_start) + SECTION_FIND_LIMIT

        section_end = mm.find(SECTION_SEP, off, bound)

        if section_end < 0:
            section_end = min(mm.size(), bound)

        if hit := section_start >= 0:
            section_start += SECTION_SEP_OFFSET
        else:
            section_start = max(0, section_end - SECTION_TEXT_LIMIT)

        log.info('section: %d in %d-%d (hit=%s)', off, section_start, section_end, hit)
        if section_start < section_end:
            return Section(start=section_start, end=section_end, hit=hit)

    def decode(self, mm: mmap.mmap) -> str:
        return (
            mm[self.start : self.end]
            .decode('utf-8', errors='replace')
            .strip('�')
            .strip()
        )

    @property
    def next_offset(self) -> int:
        return max(0, self.start - SECTION_SEP_OFFSET - 1)

    @property
    def prev_offset(self) -> int:
        return self.end + SECTION_GAP


@dataclasses.dataclass
class RGMatch:
    text: str
    line_number: int
    absolute_offset: int
    match: str
    # start: int
    # end: int

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
                m.render(segments, i, j, idx + match_offset)
                segments.__enter__()  # Checkpoint before any potential LengthExceeded


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
QUERIES: list[RGQuery] = []
QUERY_LIMIT = 10
QUERY_IDX = 0
MATCH_LIMIT = 500
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
                url=get_deep_link_url('rg_' + seg.url),
            )
        )
    elif isinstance(seg, str):
        out.extend(seg.encode('utf-16-le'))
    else:
        for item in seg:
            render_segment(item, out, entities)
    return the_offset >> 1, (len(out) - the_offset) >> 1


async def handle_rg_start(msg: Message, arg: str):
    _, i, j, k = arg.split('_')
    await asyncio.gather(msg.delete(), do_show(i, j, k, None))


async def do_show(i: str | int, j: str | int, k: str | int | None, alt_off: int | None):
    query = QUERIES[int(i)]
    file = query.files[int(j)]

    if alt_off is not None:
        off = alt_off
    else:
        off = file.matches[int(k)].absolute_offset

    with open(os.path.join(query.cwd, file.path), 'rb') as fp:
        with mmap.mmap(fp.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            if not (sect := Section.discover(mm, off)):
                return await query.message.reply_text('Unable to show the section.')

            text = sect.decode(mm)

    text = truncate_text(text)
    parse_mode = None
    if len(text) + 9 <= MAX_TEXT_LENGTH and sect.hit and (p := text.find('\n')) >= 0:
        header = text[:p].strip()
        body = text[p + 1 :].strip()

        header = escape(header) + '\n```\n'
        p = body.find('\n---\n')
        if p >= 0:  # Two-section body
            body1 = body[:p].strip()
            body2 = body[p + 5 :].strip()
            new_text = header + escape(body1) + '\n```\n```\n' + escape(body2) + '\n```'
            if len(new_text) <= MAX_TEXT_LENGTH:
                text = new_text
                parse_mode = 'MarkdownV2'
            else:
                p = -1  # Mark as undone

        if p < 0:
            new_text = header + escape(body) + '\n```'
            if len(new_text) <= MAX_TEXT_LENGTH:
                text = new_text
                parse_mode = 'MarkdownV2'

    row = [
        InlineKeyboardButton(
            text='Prev',
            callback_data=f'rg_show_{i}_{j}_{sect.prev_offset}',
        ),
        InlineKeyboardButton(
            text='Back',
            callback_data=f'rg_back_{i}',
        ),
        InlineKeyboardButton(
            text='Next',
            callback_data=f'rg_show_{i}_{j}_{sect.next_offset}',
        ),
    ]

    await query.message.edit_text(
        text,
        reply_markup=InlineKeyboardMarkup.from_row(row),
        parse_mode=parse_mode,
    )


async def handle_rg_callback(data: str):
    _, cmd, *args = data.split('_')
    list_pages = False
    match cmd:
        case 'back':
            idx = int(args[0])
            query = QUERIES[idx]
            page_num = query.page_num
        case 'page':
            idx = int(args[0])
            page_num = int(args[1])
            query = QUERIES[idx]
        case 'list':
            idx = int(args[0])
            query = QUERIES[idx]
            page_num = query.page_num
            list_pages = True
        case 'show':
            idx = int(args[0])
            j = int(args[1])
            off = int(args[2])
            return await do_show(idx, j, None, off)
        case _:
            raise ValueError('bad rg callback: ' + data)

    text, entities, markup = render_query_menu(
        query, idx, page_num=page_num, list_pages=list_pages
    )
    await query.message.edit_text(text, entities=entities, reply_markup=markup)


async def _run_rg(arg: str, cwd: str) -> RGQuery:
    cmd = ('rg', '-Sm', str(MATCH_LIMIT), '--sortr', 'path', '--json', arg)
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


def render_page(
    query: RGQuery,
    idx: int,
    page_num: int = 0,  # must be an existing or next page ( `<= len(page_offsets)`)
) -> Formatter:
    if page_num == 0:
        offsets = (0, 0, 0)
    else:
        offsets = query.page_offsets[page_num - 1]
    *render_offset, total_offset = offsets

    fmt = Formatter()

    # This yields at least one item, and each item indicates the match that
    # *will* be rendered in the next iteration, except the final (-1, -1).
    #
    # This should hold for all the 3 cases of stopping: (-1, -1), `LengthExceeded`,
    # and `PAGE_LIMIT` exceeded.
    #
    # Assuming `PAGE_LIMIT` and `MAX_TEXT_LENGTH` are resonably large, we won't
    # end up with an empty page that blocks navigation, since the length of each
    # match is limited in `RGMatch.render`. Therefore, the final `page_offset`
    # will always point to (-1, -1, cnt).
    it = query.render(fmt, idx, *render_offset)
    it = itertools.islice(it, PAGE_LIMIT + 1)

    try:
        total = total_offset - 1
        for offset in it:
            total += 1
    except LengthExceeded:
        pass

    assert total >= total_offset

    offset = (*offset, total)
    if len(query.page_offsets) == page_num:
        log.info('rg: page %d for %d: %s', page_num, idx, offset)
        query.page_offsets.append(offset)
    else:
        assert query.page_offsets[page_num] == offset

    return fmt


def button(text: str, callback_data: str = 'noop') -> InlineKeyboardButton:
    return InlineKeyboardButton(text=text, callback_data=callback_data)


def render_query_menu(
    query: RGQuery,
    idx: int,
    page_num: int = 0,
    list_pages: bool = False,
) -> tuple[str, Sequence[MessageEntity]]:
    if page_num >= len(query.page_offsets):
        while True:
            next_pn = len(query.page_offsets)
            fmt = render_page(query, idx, next_pn)
            off0, off1, total = query.page_offsets[next_pn]
            if off0 < 0 or page_num == next_pn:
                page_num = next_pn
                break
    else:
        fmt = render_page(query, idx, page_num)
        off0, off1, total = query.page_offsets[page_num]

    if off0 < 0:
        assert (off0, off1, total) == (-1, -1, query.match_cnt)
    else:
        assert off1 >= 0

    query.page_num = page_num

    if list_pages:
        row = []
        for pn, (_, _, off) in enumerate(query.page_offsets):
            if pn == page_num:
                text = '📄'
            else:
                text = f'{off} ({pn})'
            row.append(button(text, f'rg_page_{idx}_{pn}'))

        if (left := query.match_cnt - off) > 0:
            # Not exhausted yet.
            guessed_max_page = pn + left // PAGE_LIMIT + 5
            for pn in range(pn + 1, guessed_max_page):
                row.append(button(f'? ({pn})', f'rg_page_{idx}_{pn}'))

        # Group by 5 buttons per row.
        rows = tuple(row[i : i + 5] for i in range(0, len(row), 5))
        markup = InlineKeyboardMarkup(rows)
    elif total < query.match_cnt or page_num > 0:
        row = []
        if page_num > 0:
            row.append(button('Prev', f'rg_page_{idx}_{page_num - 1}'))
        else:
            row.append(button(' '))

        row.append(
            button(f'{total} / {query.match_cnt} ({page_num})', f'rg_list_{idx}')
        )

        if total < query.match_cnt:
            row.append(button('Next', f'rg_page_{idx}_{page_num + 1}'))
        else:
            row.append(button(' '))

        markup = InlineKeyboardMarkup.from_row(row)
    else:
        markup = None

    ba = bytearray()
    entities: list[MessageEntity] = []
    render_segment(fmt.segments, ba, entities)

    text = ba.decode('utf-16-le')
    if len(text) > MAX_TEXT_LENGTH:
        log.error('rg: rendering exceeded max length unexpectedly: %d', len(text))
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

    text = msg.text.strip()
    bare = text.removeprefix(f'/rg')
    if bare != text:
        if bare:
            if (c := bare[0]).isdigit():
                off = '' if c == '0' else c
            else:
                off = '4'  # Simple for /rg foo
        else:
            raise ValueError(text)
    else:
        off = '2'  # Simple Miyagi for plain text

    query = await _run_rg(arg, CWD + off)

    if not query.files:
        return await reply_text(msg, 'No matches.')

    idx = push_query(query)
    text, entities, markup = render_query_menu(query, idx)
    query.message = await reply_text(msg, text, entities=entities, reply_markup=markup)
