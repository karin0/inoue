import os
import asyncio
import dataclasses
import json
import mmap
import functools
from typing import Sequence
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

from util import get_msg_arg, truncate_text, reply_text, log, BOT_NAME, MAX_TEXT_LENGTH
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


Bold = functools.partial(Style, type=MessageEntityType.BOLD)
Underline = functools.partial(Style, type=MessageEntityType.UNDERLINE)

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

    def show(self, path: str) -> str:
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

                if section_start < 0:
                    section_start = max(0, section_end - SECTION_TEXT_LIMIT)
                else:
                    section_start += SECTION_SEP_OFFSET

                return (
                    mm[section_start:section_end]
                    .decode('utf-8', errors='replace')
                    .strip('�')
                    .strip()
                )

    def render(self, segments: list[Segment], i: int, j: int, k: int):
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

    def render(self, segments: list[Segment], i: int, j: int):
        segments.append(Bold(self.path))
        segments.append('\n')
        for idx, m in enumerate(self.matches):
            m.render(segments, i, j, idx)


@dataclasses.dataclass
class RGQuery:
    files: Sequence[RGFile]
    cwd: str
    message: Message | None = None
    menu_text: str = ''
    menu_entities: Sequence[MessageEntity] = ()

    def render(self, segments: list[Segment], i: int):
        for idx, f in enumerate(self.files):
            f.render(segments, i, idx)


CWD = os.environ['RG_CWD']
URL_BASE = 'https://t.me/' + BOT_NAME + '?start=rg_'
QUERIES: list[RGQuery] = []
QUERY_LIMIT = 10
QUERY_IDX = 0
MATCH_LIMIT = 10


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
    text = file.matches[int(k)].show(os.path.join(query.cwd, file.path))
    text = truncate_text(text)

    await asyncio.gather(
        msg.delete(),
        query.message.edit_text(
            text,
            reply_markup=InlineKeyboardMarkup(
                inline_keyboard=[
                    [InlineKeyboardButton(text='Back', callback_data=f'rg_back_{i}')],
                ]
            ),
        ),
    )


async def handle_rg_callback(data: str):
    i = int(data[data.rindex('_') + 1 :])
    query = QUERIES[int(i)]
    await query.message.edit_text(
        query.menu_text,
        entities=query.menu_entities,
    )


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
            log.warning('rg exited with code %d', r)

    return query


async def handle_rg(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    msg, arg = get_msg_arg(update)
    if not arg:
        return await reply_text(msg, 'Provide a keyword.')

    query = await _run_rg(arg, CWD + ('' if msg.text.startswith('/rg') else '2'))

    if not query.files:
        return await reply_text(msg, 'No matches.')

    idx = push_query(query)
    segments = []
    query.render(segments, idx)

    ba = bytearray()
    entities: list[MessageEntity] = []
    render_segment(segments, ba, entities)

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

    query.menu_text = text
    query.menu_entities = entities
    query.message = await reply_text(msg, text, entities=entities)
