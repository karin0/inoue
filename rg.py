import asyncio
import subprocess
import dataclasses
import json
import functools
from typing import Sequence

from telegram import (
    MessageEntity,
    Update,
    Message,
    InlineKeyboardMarkup,
    InlineKeyboardButton,
)
from telegram.ext import ContextTypes
from telegram.constants import MessageEntityType

from util import get_arg, MAX_TEXT_LENGTH, truncate_text, BOT_NAME

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


@dataclasses.dataclass
class RGMatch:
    text: str
    line_number: int
    match: str
    # start: int
    # end: int

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
    message: Message | None = None
    menu_text: str = ''
    menu_entities: Sequence[MessageEntity] = ()

    def render(self, segments: list[Segment], i: int):
        for idx, f in enumerate(self.files):
            f.render(segments, i, idx)


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
    arg = get_arg()
    msg: Message = update.message
    if not (arg and arg.startswith('rg_')):
        return await msg.reply_text('Hello!', do_quote=True)

    _, i, j, k = arg.split('_')
    query = QUERIES[int(i)]
    text = query.files[int(j)].matches[int(k)].text
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


async def handle_rg(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    arg = get_arg()
    if not arg:
        return await update.message.reply_text('Provide a keyword.', do_quote=True)

    cwd_offset = '' if update.message.text.startswith('/rg') else '2'

    child = await asyncio.create_subprocess_exec(
        'rg',
        '-m',
        str(MATCH_LIMIT),
        '--sortr',
        'path',
        '--json',
        arg,
        cwd='/data/app/gmact/out' + cwd_offset,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )

    query = RGQuery(files=[])
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
        await child.wait()

    if not query.files:
        return await update.message.reply_text('No matches.', do_quote=True)

    idx = push_query(query)
    segments = []
    query.render(segments, idx)

    ba = bytearray()
    entities = []
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
    query.message = await update.message.reply_text(
        text, entities=entities, do_quote=True
    )
