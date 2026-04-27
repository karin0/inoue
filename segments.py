import functools
from typing import Callable, Sequence
from abc import ABC, abstractmethod
from dataclasses import dataclass, replace

from render_core import Box
from util import log, escape, html_escape, escape_pre, MAX_TEXT_LENGTH

type Segment = Sequence[Segment] | str | Element


@dataclass(frozen=True)
class Element(Box, ABC):
    inner: Segment

    def __str__(self) -> str:
        return render_segment(self.inner)

    @abstractmethod
    def html(self, out: list[str]) -> None: ...

    @abstractmethod
    def md(self, out: list[str]) -> None: ...


@dataclass(frozen=True)
class Style(Element):
    tag: str
    sym: str

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


@dataclass(frozen=True)
class Pre(Element):
    lang: str = ''

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
        out.append('```')


# https://core.telegram.org/bots/api#markdownv2-style
Bold = functools.partial(Style, tag='b', sym='*')
Italic = functools.partial(Style, tag='i', sym='_')
Underline = functools.partial(Style, tag='u', sym='__')
Strikethrough = functools.partial(Style, tag='s', sym='~')
Code = functools.partial(Style, tag='code', sym='`')
Spoiler = functools.partial(Style, tag='tg-spoiler', sym='||')


@dataclass(frozen=True)
class Link(Element):
    url: str

    def __repr__(self) -> str:
        return f'Link[{self.url}]({self.inner!r})'

    def html(self, out: list[str]) -> None:
        out.append(f'<a href="{html_escape(self.url)}">')
        to_html(self.inner, out)
        out.append('</a>')

    def md(self, out: list[str]) -> None:
        out.append('[')
        to_md(self.inner, out)
        out.append(f']({escape(self.url)})')


@dataclass(frozen=True)
class BlockQuote(Element):
    expandable: bool = False

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


@dataclass(frozen=True)
class Time(Element):
    unix: int
    format: str = ''

    def __repr__(self) -> str:
        return f'Time[{self.unix}|{self.format}]({self.inner!r})'

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


@dataclass(frozen=True)
class Raw(Element):
    inner: str

    def __repr__(self) -> str:
        return f'Raw({self.inner!r})'

    def html(self, out: list[str]) -> None:
        out.append(self.inner)

    def md(self, out: list[str]) -> None:
        out.append(self.inner)


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


def get_renderer(parse_mode: str | None) -> Callable[[Segment, list[str]], None]:
    if parse_mode == 'MarkdownV2':
        return to_md
    elif parse_mode == 'HTML':
        return to_html
    else:
        return to_plain


def render_segment(
    seg: Segment, func: Callable[[Segment, list[str]], None] = to_plain
) -> str:
    out = []
    func(seg, out)
    return ''.join(out)


def _to_length(cache: dict[int, int], s: Element | Sequence[Segment]) -> int:
    '''Cache lengths for non-leaf segments.'''
    # Use `id` so that lists can be cached.
    # This means that the formatted lists must not be mutated in the same `Formatter` context.
    id_s = id(s)
    if (r := cache.get(id_s)) is not None:
        return r

    if isinstance(s, Element):
        # Telegram does not count tags/urls from the entities in the text length,
        # so we can always keep the outer style even truncated.
        r = _to_length(cache, s.inner)
    else:
        r = sum(
            len(item) if isinstance(item, str) else _to_length(cache, item)
            for item in s
        )

    cache[id_s] = r
    return r


class Formatter:
    limit = MAX_TEXT_LENGTH

    def __init__(
        self,
        *,
        strict: bool = False,
    ) -> None:
        self.segments: list[Segment] = []
        self.length = 0
        self.full = False
        self._best_effort = not strict
        self._lengths: dict[int, int] = {}

    def to_length(self, s: Segment) -> int:
        return len(s) if isinstance(s, str) else _to_length(self._lengths, s)

    def reserve(self, seg: Segment):
        '''Reserve its space, but not appending it.

        Note that this is unconditional, which can cause `length > limit`.
        '''
        self.length += self.to_length(seg)

    def _append_best_effort(self, seg: Segment, budget: int, out: list[Segment]):
        '''Precondition: to_length(seg) > budget > 0'''
        if isinstance(seg, str):
            item = seg[:budget]
            log.debug('_append_best_effort: %r (%d)', item, budget)
            out.append(item)
            self.length += len(item)
            return

        if isinstance(seg, Element):
            # Replace the inner into the outer `Element`.
            self._append_best_effort(seg.inner, budget, buf := [])
            if buf:
                seg = replace(seg, inner=buf[0] if len(buf) == 1 else buf)
                log.debug('_append_best_effort: %r (%d)', seg, budget)
                out.append(seg)
            return

        for item in seg:
            if (inc_len := self.to_length(item)) > budget:
                return self._append_best_effort(item, budget, out)

            log.debug('_append_best_effort: %r (%d)', item, budget)
            out.append(item)
            self.length += inc_len
            budget -= inc_len
            if budget <= 0:
                break

    def try_append(self, seg: Segment) -> bool:
        if not seg:
            return True

        if self.full:
            return False

        if self._best_effort and isinstance(seg, (list, tuple)):
            return self.try_extend(seg)

        inc_len = self.to_length(seg)
        new_len = self.length + inc_len
        if new_len <= self.limit:
            self.segments.append(seg)
            self.length = new_len
            return True

        self.full = True
        desc = repr(seg)
        if len(desc) > 100:
            desc = desc[:50] + '...' + desc[-47:]
        log.info(
            'segment truncated: %s (%d + %d = %d)', desc, self.length, inc_len, new_len
        )

        if self._best_effort and (budget := self.limit - self.length) > 0:
            self._append_best_effort(seg, budget, self.segments)

        return False

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
