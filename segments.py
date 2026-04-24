import functools
from typing import Callable, Sequence
from abc import ABC, abstractmethod

from render_core import Box
from util import log, escape, html_escape, escape_pre, MAX_TEXT_LENGTH

type Segment = Sequence[Segment] | str | Element


class Element(Box, ABC):
    def __init__(self, inner: Segment) -> None:
        super().__init__()
        self.inner = inner

    def __str__(self) -> str:
        return render_segment(self.inner)

    def __len__(self) -> int:
        return to_length(self.inner)

    @abstractmethod
    def html(self, out: list[str]) -> None: ...

    @abstractmethod
    def md(self, out: list[str]) -> None: ...


class Style(Element):
    def __init__(self, inner: Segment, tag: str, sym: str) -> None:
        super().__init__(inner)
        self.tag = tag
        self.sym = sym

    def __repr__(self) -> str:
        return f'Style[{self.tag}]({self.inner!r})'

    def html(self, out: list[str]) -> None:
        out.append(f'<{self.tag}>')
        to_html(self.inner, out)
        out.append(f'</{self.tag}>')

    def md(self, out: list[str]) -> None:
        out.append(self.sym)
        to_md(self.inner, out)
        out.append(self.sym)


class Pre(Element):
    def __init__(self, inner: Segment, lang: str = '') -> None:
        super().__init__(inner)
        self.lang = lang

    def __repr__(self) -> str:
        if self.lang:
            return f'Pre[{self.lang}]({self.inner!r})'
        return f'Pre({self.inner!r})'

    def html(self, out: list[str]) -> None:
        if self.lang:
            out.append(f'<pre lang="{html_escape(self.lang)}">')
        else:
            out.append('<pre>')
        to_html(self.inner, out)
        out.append('</pre>')

    def md(self, out: list[str]) -> None:
        out.append('```')
        if self.lang:
            out.append(escape(self.lang))
        out.append('\n')
        if isinstance(self.inner, str):
            out.append(escape_pre(self.inner).rstrip())
        else:
            to_md(self.inner, out)
        out.append('\n```')


Bold = functools.partial(Style, tag='b', sym='*')
Underline = functools.partial(Style, tag='u', sym='__')
Code = functools.partial(Style, tag='code', sym='`')


class Link(Element):
    def __init__(self, inner: Segment, url: str) -> None:
        super().__init__(inner)
        self.url = url

    def __repr__(self) -> str:
        return f'Link[{self.url}]({self.inner!r})'

    def __len__(self) -> int:
        return to_length(self.inner) + len(self.url)

    def html(self, out: list[str]) -> None:
        out.append(f'<a href="{html_escape(self.url)}">')
        to_html(self.inner, out)
        out.append('</a>')

    def md(self, out: list[str]) -> None:
        out.append('[')
        to_md(self.inner, out)
        out.append(f']({escape(self.url)})')


class BlockQuote(Element):
    def __init__(self, inner: Segment, expandable: bool = False) -> None:
        super().__init__(inner)
        self.expandable = expandable

    def __repr__(self) -> str:
        return f'BlockQuote{"[expandable]" if self.expandable else ""}({self.inner!r})'

    def html(self, out: list[str]) -> None:
        if self.expandable:
            out.append('<blockquote expandable>')
        else:
            out.append('<blockquote>')
        to_html(self.inner, out)
        out.append('</blockquote>')

    def md(self, out: list[str]) -> None:
        inner = []
        to_md(self.inner, inner)
        lines = ''.join(inner).rstrip().splitlines()
        if not lines:
            return

        if self.expandable:
            out.append('**')
            last = lines.pop()
            for line in lines:
                out.append(f'>{line}\n')
            out.append(f'>{last}||\n')
        else:
            for line in lines:
                out.append(f'>{line}\n')


class Time(Element):
    def __init__(self, inner: Segment, unix: int, format: str = '') -> None:
        super().__init__(inner)
        self.unix = unix
        self.format = format

    def __repr__(self) -> str:
        return f'Time[{self.unix}|{self.format}]({self.inner!r})'

    def __len__(self) -> int:
        return to_length(self.inner) + len(str(self.unix)) + len(self.format)

    def html(self, out: list[str]) -> None:
        out.append(f'<tg-time unix="{self.unix}" format="{self.format}">')
        to_html(self.inner, out)
        out.append('</tg-time>')

    def md(self, out: list[str]) -> None:
        out.append('![')
        to_md(self.inner, out)
        if self.format:
            out.append(f'](tg://time?unix={self.unix}&format={self.format})')
        else:
            out.append(f'](tg://time?unix={self.unix})')


class Raw(Element):
    def __init__(self, inner: str) -> None:
        super().__init__(inner)
        self.inner: str

    def __repr__(self) -> str:
        return f'Raw({self.inner!r})'

    def html(self, out: list[str]) -> None:
        out.append(self.inner)

    def md(self, out: list[str]) -> None:
        out.append(self.inner)


def to_length(s: Segment) -> int:
    if isinstance(s, (Element, str)):
        return len(s)
    return sum(to_length(item) for item in s)


def to_html(seg: Segment, out: list[str]) -> None:
    if isinstance(seg, Element):
        seg.html(out)
    elif isinstance(seg, str):
        out.append(html_escape(seg))
    else:
        for item in seg:
            to_html(item, out)


def to_md(seg: Segment, out: list[str]) -> None:
    if isinstance(seg, Element):
        seg.md(out)
    elif isinstance(seg, str):
        out.append(escape(seg))
    else:
        for item in seg:
            to_md(item, out)


def to_plain(seg: Segment, out: list[str]) -> None:
    if isinstance(seg, Element):
        to_plain(seg.inner, out)
    elif isinstance(seg, str):
        out.append(seg)
    else:
        for item in seg:
            to_plain(item, out)


def render_segment(
    seg: Segment, func: Callable[[Segment, list[str]], None] = to_plain
) -> str:
    out = []
    func(seg, out)
    return ''.join(out)


class Formatter:
    def __init__(self) -> None:
        self.segments: list[Segment] = []
        self.length = 0
        self.full = False

    def try_append(self, seg: Segment) -> bool:
        if not seg:
            return True

        inc_len = to_length(seg)
        new_len = self.length + inc_len
        if new_len > MAX_TEXT_LENGTH:
            self.full = True
            if (t := MAX_TEXT_LENGTH - self.length) > 0:
                log.info(f'segment truncated: {seg!r} ({self.length} + {inc_len})')
                self.segments.append('[truncated]'[:t])
                self.length += min(t, 11)
            return False
        self.segments.append(seg)
        self.length = new_len
        return True

    def try_extend(self, segs: Sequence[Segment]) -> bool:
        for seg in segs:
            if not self.try_append(seg):
                return False
        return True

    def try_push(self, *segs: Segment) -> bool:
        return self.try_extend(segs)

    def append(self, seg: Segment):
        if not self.try_append(seg):
            raise ValueError(f'segment overflow: {seg!r}')

    def html(self) -> str:
        return render_segment(self.segments, to_html)

    def md(self) -> str:
        return render_segment(self.segments, to_md)

    def plain(self) -> str:
        return render_segment(self.segments, to_plain)
