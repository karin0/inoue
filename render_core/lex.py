from typing import Iterable, Iterator, Literal
from .context import trace, is_not_quiet

if not is_not_quiet:
    trace = lambda *_: None


type Code = tuple[Literal[True], str]
type Text = tuple[Literal[False], str | None]
type Chunk = Code | Text


class Chunker:
    r'''
    Chunks the input `text` into code blocks and text fragments.

    A code block (technically treated as `block_inner` in the grammar) is either:

    1. An outermost `{ ... }` block with balanced braces, or
    2. A (shortest) contiguous chunk of lines that:
        - ends with r';\s*$', and
        - has no unclosed `{}` inside.

    For 1., the outermost braces are stripped from the yielded chunk to match
    the `block_inner`. This means the scope modifiers before it ('@ [name]') is
    treated as texts (use naked blocks or nested code blocks to preserve them).

    For 2. (naked blocks), the chunk without the trailing whitespaces is yielded,
    followed by a `(False, None)` to indicate an "implicit" line break.

    Text fragments outside code blocks are yielded as-is.
    '''

    __slots__ = (
        'errors',
        '_text',
        '_block',
        '_escaping_indices',
    )

    def __init__(
        self,
        text: str,
        *,
        block: bool = False,
        errors: list[str] | None = None,
    ):
        self.errors: list[str] = [] if errors is None else errors
        self._text = text
        self._block = block
        self._escaping_indices: list[int] = []

    def _text_fragments(
        self, start: int, end: int
    ) -> Iterable[tuple[Literal[False], str]]:
        # Remove escaping `\` chars in text fragments.
        trace('text_fragments: %s to %s, %s', start, end, self._escaping_indices)
        self._escaping_indices.append(end)
        for i in self._escaping_indices:
            if start < i:
                yield False, self._text[start:i]
            else:
                assert start == i, (start, i, end)
            start = i + 1
        self._escaping_indices.clear()

    def _test_naked(self, naked_start: int, chunk: str) -> str | None:
        trace('test_naked: %r', chunk)
        stem = chunk.rstrip()
        if not stem.endswith(';'):
            return None
        escape_pos = naked_start + len(stem) - 2
        if escape_pos in self._escaping_indices:
            trace('test_naked: escaped at %s', escape_pos)
            return None
        trace('test_naked: Naked block: %r', chunk)
        return stem

    def __iter__(self) -> Iterator[Chunk]:
        text = self._text
        block = self._block
        errors = self.errors
        trace('Chunking text (block=%s): %s', block, text)

        block_starts: list[int] = []
        if block:
            # Pretend a `{` before the start.
            block_starts.append(-1)

        # Escaping state inside and outside code blocks.
        escape = False

        # Lexical states that are only valid inside blocks.
        # `comment` is 0 (none), 1 (line `// ... \n` or `# ...`), 2 (block `/* ... */`).
        comment = 0
        quote = None
        raw = False

        # Buffered chunks inside a block.
        buf: list[str] = []

        # The next position we should yield from.
        cursor = 0

        # Tracking the naked block candidate.
        _naked_start = 0
        _naked_buf: list[Chunk] = []

        for p, c in enumerate(text):
            if escape:
                escape = False
                continue

            if comment:
                if comment == 2:
                    # /* ... */ — ends only at a closing `*/`.
                    if c == '/' and p and text[p - 1] == '*':
                        comment = 0
                    cursor = p + 1
                    continue
                # Line comment — runs until '\n', which then falls through so
                # the rest of the loop can act on the newline (naked-block
                # detection, etc.).
                if c != '\n':
                    cursor = p + 1
                    continue
                comment = 0

            if quote:
                # The escaping char `\` in string literals is rehandled by the
                # parser later. We don't remove it from the fragment.
                # But in raw literals, `\` is NOT special.
                if not raw and c == '\\':
                    escape = True
                elif c == quote:
                    quote = None
                continue

            # Raw literals starting with '`' are NOT enclosed and consumes all the
            # rest of the statement, i.e., until the next statement boundary.
            # Statement boundaries ';' and '{' and '}' terminate raw literals,
            # but they are escaped in (and only in) quotes.
            # Escaping char `\` is NOT special in raw literals.
            if raw:
                if c not in ';{}':
                    if c == '"' or c == "'":
                        quote = c
                    continue
                raw = False
                chunk = text[cursor + 1 : p]
                trace('Raw literal: %s', chunk)
                buf.append('\'')
                buf.append(chunk.replace('\\', '\\\\').replace("'", r"\'"))
                buf.append('\'')
                cursor = p

            if c == '{':
                if not block_starts:
                    for fragment in self._text_fragments(cursor, p):
                        _naked_buf.append(fragment)
                    cursor = p

                block_starts.append(p)
                trace('Block starts: %s', block_starts)

            elif c == '}':
                if not block_starts or (block and len(block_starts) == 1):
                    # An plain '}' as text, when no block is open.
                    # But, when `block` is set, we cannot close the first (pretended) block here,
                    # or stuff afterwards would be "leaked" as text fragments and injected into
                    # our output.
                    # We leave it to cause a Lark parse error later.
                    continue
                trace('Block ends: %s', block_starts)
                block_starts.pop()
                if not block_starts:
                    buf.append(text[cursor:p])

                    # Remove the block start '{'.
                    # We cannot save `p+1` directly when '{' is found, in case of
                    # unclosed blocks that are recovered as final texts later.
                    # The '{' must exist there, as we cannot reach here if `block` is set.
                    chunk = ''.join(buf)[1:].strip()
                    buf.clear()
                    if chunk:
                        trace('Chunked block: %r', chunk)
                        _naked_buf.append((True, chunk))
                    cursor = p + 1

            elif block_starts:
                # Inside a block: track string literals, comments, raw literals.
                # `\n` is intentionally NOT handled here, so a naked block can
                # contain multi-line nested blocks.
                if c == "'" or c == '"':
                    quote = c
                elif c == '#':
                    comment = 1
                    if chunk := text[cursor:p]:
                        buf.append(chunk)
                elif c == '/' and p + 1 < len(text):
                    ch = text[p + 1]
                    if ch == '/':
                        comment = 1
                    elif ch == '*':
                        comment = 2
                    else:
                        continue
                    if chunk := text[cursor:p]:
                        buf.append(chunk)
                elif c == '`':
                    trace('Raw literal starts at %s', p)
                    raw = True
                    if chunk := text[cursor:p]:
                        buf.append(chunk)
                    cursor = p

            # Below: outside any block. We only handle escaping '\' chars
            # outside blocks (escaping '{', '}', ';', '\' in text fragments) —
            # inside blocks, '\' in quotes is rehandled by the parser later
            # and elsewhere has no outer-level meaning.
            elif c == '\\' and p + 1 < len(text) and text[p + 1] in '{};\\':
                escape = True
                self._escaping_indices.append(p)
            elif c == '\n':
                if (
                    stem := self._test_naked(_naked_start, text[_naked_start:p])
                ) is not None:
                    # We're about to bypass `_text_fragments` for this region.
                    # Clear `_escaping_indices` to maintain its invariant for
                    # the next text fragment.
                    _naked_buf.clear()
                    self._escaping_indices.clear()

                    yield False, text[cursor:_naked_start]
                    cursor = p + 1
                    yield from Chunker(stem, block=True, errors=errors)

                    # We are actually consuming the line break here, but yielding a `\n`
                    # each time would result in too many empty lines for consecutive
                    # inline naked blocks (`a = 1;\nb = 2;\n...`).
                    # `None` hints the implicit line break after the naked block here,
                    # so the caller can decide smartly.
                    yield False, None

                    _naked_start = p + 1
                    continue

                if _naked_buf:
                    yield from _naked_buf
                    _naked_buf.clear()

                yield from self._text_fragments(cursor, p + 1)
                cursor = _naked_start = p + 1

        # Flush final line buffer, pretending a EOF.
        if (
            not block
            and (stem := self._test_naked(_naked_start, text[_naked_start:]))
            is not None
        ):
            yield False, text[cursor:_naked_start]
            yield from Chunker(stem, block=True, errors=errors)
            return

        if _naked_buf:
            yield from _naked_buf

        if block:
            # Pretend a final `}` to close the "virtual" block opened at the start.
            assert block_starts
            if len(block_starts) > 1:
                # There are NO text fragments or unclosed blocks in `block` mode, as the
                # entire input is treated as a single block, where we are only expected
                # to preprocess comments and raw literals with the `buf`.
                # Trying to recover would leak text fragments inside the block, so we
                # also leave it to cause a Lark parse error later.
                msg = f'Unbalanced naked block starting at position {block_starts} {cursor}'
                errors.append(msg)
                trace('%s %s', msg, text)

            chunk = text[cursor:]
            buf.append(chunk)
            if chunk := ''.join(buf).strip():
                yield True, chunk
            return

        if block_starts:
            # Unclosed block! Reparse the inner of the first unclosed block to recover.
            msg = f'Unclosed block starting at position {block_starts} {cursor}'
            errors.append(msg)
            trace('%s', msg)

            p = block_starts[0]
            assert p >= 0  # `block` mode cannot reach here.
            yield False, text[p]

            rest = text[p + 1 :]
            trace('Recovering rest: %s', rest)
            yield from Chunker(rest, errors=errors)
            return

        # `buf` must be empty, since `block_starts` is empty.
        assert not buf

        # Flush final text fragments.
        yield from self._text_fragments(cursor, len(text))
