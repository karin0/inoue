import os
import asyncio
import json
import mmap
import itertools
from typing import Iterable, Sequence
from subprocess import DEVNULL, PIPE
from dataclasses import dataclass, field

from telegram import (
    Update,
    Message,
    InlineKeyboardMarkup,
    InlineKeyboardButton,
)
from telegram.ext import ContextTypes

from util import (
    log,
    escape,
    get_msg_arg,
    get_deep_link_url,
    pre_block_raw,
    truncate_text,
    reply_text,
    MAX_TEXT_LENGTH,
)
from segments import Segment, Link, Bold, Underline, Formatter

# >>> 2026-01-02 15:04:05 (1)
SECTION_SEP = b'>>> 202'
SECTION_SEP_OFFSET = 2
SECTION_GAP = 24


@dataclass(frozen=True, slots=True, eq=False, match_args=False)
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


@dataclass(frozen=True, slots=True, eq=False, match_args=False)
class RGMatch:
    text: str
    line_number: int
    absolute_offset: int
    match: str
    # start: int
    # end: int

    def segment(self, i: int, j: int, k: int) -> Segment:
        kw = self.match
        s = self.text

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

        return (
            Link(
                [
                    str(self.line_number) + ':' + pl + s[l:p],
                    Underline(Bold(kw)),
                    s[q:r] + pr,
                ],
                url=get_deep_link_url(f'rg_{i}_{j}_{k}'),
            ),
            '\n',
        )


@dataclass(frozen=True, slots=True, eq=False, match_args=False)
class RGFile:
    matches: list[RGMatch]
    path: str

    def render(
        self, fmt: Formatter, i: int, j: int, match_offset: int
    ) -> Iterable[tuple[int, int]]:
        if not fmt.try_append((Bold(self.path), '\n')):
            yield (j, match_offset)
            return
        for idx, m in enumerate(self.matches[match_offset:]):
            k = idx + match_offset
            yield (j, k)
            if not fmt.try_append(m.segment(i, j, k)):
                return


@dataclass(slots=True, eq=False, match_args=False)
class RGQuery:
    files: Sequence[RGFile]
    cwd: str
    match_cnt: int = 0
    page_num: int = 0
    page_offsets: list[tuple[int, int, int]] = field(default_factory=list)
    message: Message | None = None

    def render(
        self, fmt: Formatter, i: int, file_offset: int, match_offset: int
    ) -> Iterable[tuple[int, int]]:
        assert file_offset >= 0 and match_offset >= 0
        assert file_offset < len(self.files)
        for idx, f in enumerate(self.files[file_offset:]):
            j = idx + file_offset
            mo = match_offset if idx == 0 else 0
            yield from f.render(fmt, i, j, mo)
            if fmt.full:
                return
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


async def handle_rg_start(msg: Message, arg: str):
    _, i, j, k = arg.split('_')
    await asyncio.gather(msg.delete(), do_show(i, j, k, None))


async def do_show(i: str | int, j: str | int, k: str | int | None, alt_off: int | None):
    query = QUERIES[int(i)]
    message = query.message
    assert message is not None
    file = query.files[int(j)]

    if alt_off is not None:
        off = alt_off
    else:
        assert k is not None
        off = file.matches[int(k)].absolute_offset

    with open(os.path.join(query.cwd, file.path), 'rb') as fp:
        with mmap.mmap(fp.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            if not (sect := Section.discover(mm, off)):
                return await message.reply_text('Unable to show the section.')

            text = sect.decode(mm)

    text = truncate_text(text)
    parse_mode = None
    if sect.hit and (p := text.find('\n')) >= 0:
        header = text[:p].strip()
        body = text[p + 1 :].strip()

        header = escape(header) + '\n'
        p = body.find('\n---\n')
        if p >= 0:  # Two-section body
            body1 = body[:p].strip()
            body2 = body[p + 5 :].strip()
            text = header + pre_block_raw(body1) + pre_block_raw(body2)
            parse_mode = 'MarkdownV2'
        else:
            text = header + pre_block_raw(body)
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

    await message.edit_text(
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

    text, markup = render_query_menu(
        query, idx, page_num=page_num, list_pages=list_pages
    )
    message = query.message
    assert message is not None
    await message.edit_text(text, parse_mode='HTML', reply_markup=markup)


async def _run_rg(arg: str, cwd: str) -> RGQuery:
    cmd = ('rg', '-Sm', str(MATCH_LIMIT), '--sortr', 'path', '--json', arg)
    child = await asyncio.create_subprocess_exec(
        *cmd, stdin=DEVNULL, stdout=PIPE, stderr=DEVNULL, cwd=cwd
    )

    files: list[RGFile] = []
    query = RGQuery(files=files, cwd=cwd)
    stdout = child.stdout
    assert stdout is not None
    cnt = 0
    try:
        async for line in stdout:
            if line:
                line = json.loads(line)
                match line['type']:
                    case 'begin':
                        files.append(
                            RGFile(matches=[], path=line['data']['path']['text'])
                        )
                    case 'match':
                        if not files:
                            continue
                        data = line['data']
                        match = data['submatches'][0]
                        files[-1].matches.append(
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
        await stdout.read()
        if r := await child.wait():
            log.info('rg exited with code %d', r)

    query.match_cnt = cnt
    return query


def render_page(
    query: RGQuery,
    idx: int,
    page_num: int = 0,  # must be an existing or next page ( `<= len(page_offsets)`)
) -> str:
    if page_num == 0:
        offsets = (0, 0, 0)
    else:
        offsets = query.page_offsets[page_num - 1]
    *render_offset, total_offset = offsets

    # Each yield precedes the try_append for that match. The last yielded
    # offset is therefore the first unrendered match (the resume point for
    # the next page), or (-1, -1) if all matches were rendered.
    #
    # This holds for all 3 cases of stopping: natural exhaustion (-1, -1),
    # formatter full, and `PAGE_LIMIT` exceeded (islice).
    #
    # Assuming `PAGE_LIMIT` and `MAX_TEXT_LENGTH` are reasonably large, we won't
    # end up with an empty page that blocks navigation, since the length of each
    # match is limited in `RGMatch.segment`. Therefore, the final `page_offset`
    # will always point to (-1, -1, cnt).
    offset: tuple[int, int] = (-1, -1)
    total = total_offset - 1

    fmt = Formatter(strict=True)
    for offset in itertools.islice(
        query.render(fmt, idx, *render_offset), PAGE_LIMIT + 1
    ):
        total += 1
    result = fmt.html()

    assert total >= total_offset

    page_offset = (offset[0], offset[1], total)
    if len(query.page_offsets) == page_num:
        log.info('rg: page %d for %d: %s', page_num, idx, page_offset)
        query.page_offsets.append(page_offset)
    else:
        assert query.page_offsets[page_num] == page_offset

    return result


def button(text: str, callback_data: str = 'noop') -> InlineKeyboardButton:
    return InlineKeyboardButton(text=text, callback_data=callback_data)


def render_query_menu(
    query: RGQuery,
    idx: int,
    page_num: int = 0,
    list_pages: bool = False,
) -> tuple[str, InlineKeyboardMarkup | None]:
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

        last_off = query.page_offsets[-1][2]
        last_pn = len(query.page_offsets) - 1
        if (left := query.match_cnt - last_off) > 0:
            # Not exhausted yet.
            guessed_max_page = last_pn + left // PAGE_LIMIT + 5
            for next_pn in range(last_pn + 1, guessed_max_page):
                row.append(button(f'? ({next_pn})', f'rg_page_{idx}_{next_pn}'))

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

    return fmt, markup


async def handle_rg(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    msg, arg = get_msg_arg(update)
    if not arg:
        return await reply_text(msg, 'Provide a keyword.')

    text = msg.text
    assert text
    text = text.strip()
    if len(bare := text.removeprefix('/rg')) != len(text):
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
    text, markup = render_query_menu(query, idx)
    query.message = await reply_text(msg, text, parse_mode='HTML', reply_markup=markup)
