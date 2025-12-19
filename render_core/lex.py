from typing import Iterable, Literal
from .context import trace, is_not_quiet

lex_errors = []

if not is_not_quiet:
    trace = lambda *_: None


def lex(text: str, *, block: bool = False) -> Iterable[tuple[bool, str | None]]:
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
    trace('Lexing text (block=%s): %s', block, text)

    block_starts = []
    if block:
        # Pretend a `{` before the start.
        block_starts.append(-1)

    # Escaping state inside and outside code blocks.
    escape = False

    # Lexical states that are only valid inside blocks.
    comment: int = False
    quote = None
    raw = False

    # Buffered chunks inside a block.
    buf = []

    # The next position we should yield from.
    cursor = 0

    # Positions of escaping `\` chars in text fragments.
    escaping_indices = []

    # Fragments found but deferred to yield, until we confirm they are not
    # part of a naked block (via a line break in a text fragment).
    naked_buf = []
    naked_start = 0

    def text_fragments(start: int, end: int) -> Iterable[tuple[Literal[False], str]]:
        # Remove escaping `\` chars in text fragments.
        trace('text_fragments: %s to %s, %s', start, end, escaping_indices)
        escaping_indices.append(end)
        for i in escaping_indices:
            if start < i:
                chunk = text[start:i]
                yield False, chunk
            else:
                assert start == i, (start, i, end)
            start = i + 1
        escaping_indices.clear()

    def test_naked(chunk) -> str | None:
        trace('test_naked: %r', chunk)
        if (stem := chunk.rstrip()).endswith(';'):
            escape_pos = naked_start + len(stem) - 2
            if escape_pos in escaping_indices:
                trace('test_naked: escaped at %s', escape_pos)
            else:
                trace('test_naked: Naked block: %r\n  Dropped: %s', chunk, naked_buf)
                return stem

    for p, c in enumerate(text):
        if escape:
            escape = False
            continue

        if comment == 2:
            # /* ... */
            if c == '/' and p and text[p - 1] == '*':
                comment = False
            cursor = p + 1
            continue

        if comment:
            if c != '\n':
                cursor = p + 1
                continue

            # Skip the comment.
            comment = False

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
        if raw:
            if c != ';' and c != '{' and c != '}':
                # Statement boundaries ';' and '{' and '}' terminate raw literals,
                # but they are escaped in (and only in) quotes.
                # Escaping char `\` is NOT special in raw literals.
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

        match c:
            case '{':
                if not block_starts:
                    for fragment in text_fragments(cursor, p):
                        naked_buf.append(fragment)
                    cursor = p

                block_starts.append(p)
                trace('Block starts: %s', block_starts)

            case '}':
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
                    chunk = text[cursor:p]
                    buf.append(chunk)

                    # Remove the block start '{'.
                    # We cannot save `p+1` directly when '{' is found, in case of
                    # unclosed blocks that are recovered as final texts later.
                    # The '{' must exist there, as we cannot reach here if `block` is set.
                    chunk = ''.join(buf)[1:].strip()
                    buf.clear()
                    if chunk:
                        trace('Lexed block: %r', chunk)
                        naked_buf.append((True, chunk))
                    cursor = p + 1

            case _:
                if block_starts:
                    match c:
                        case "'" | '"':
                            quote = c

                        case '#':
                            comment = True
                            # Flush up before the comment.
                            if chunk := text[cursor:p]:
                                buf.append(chunk)

                        case '/':
                            if p + 1 < len(text):
                                ch = text[p + 1]
                                if ch == '/':
                                    comment = True
                                elif ch == '*':
                                    comment = 2
                                else:
                                    continue

                                # Flush up before the comment.
                                if chunk := text[cursor:p]:
                                    buf.append(chunk)

                        case '`':
                            trace('Raw literal starts at %s', p)
                            raw = True
                            # Flush up before the raw literal.
                            if chunk := text[cursor:p]:
                                buf.append(chunk)
                            cursor = p
                        # We do not handle '\n' inside blocks, so a naked block
                        # can contain multi-line blocks.

                # We only handles escaping '\' chars inside quotes (as such inside blocks)
                # or outside any blocks (to escape '{', '}', ';' and '\' in text fragments).
                # The latter ones here must be removed from the output fragments.
                elif c == '\\':
                    escape = True
                    escaping_indices.append(p)
                elif c == '\n':
                    if (stem := test_naked(text[naked_start:p])) is not None:
                        naked_buf.clear()
                        escaping_indices.clear()

                        yield False, text[cursor:naked_start]
                        cursor = p + 1
                        yield from lex(stem, block=True)

                        # We are actually consuming the line break here, but yielding a `\n`
                        # each time would result in too many empty lines for consecutive
                        # inline naked blocks (`a = 1;\nb = 2;\n...`).
                        # `None` hints the implicit line break after the naked block here,
                        # so the caller can decide smartly.
                        yield False, None

                        naked_start = p + 1
                        continue

                    if naked_buf:
                        yield from naked_buf
                        naked_buf.clear()

                    yield from text_fragments(cursor, p + 1)
                    cursor = naked_start = p + 1

    # Flush final line buffer, pretending a EOF.
    if not block and (stem := test_naked(text[naked_start:])) is not None:
        yield False, text[cursor:naked_start]
        yield from lex(stem, block=True)
        return

    if naked_buf:
        yield from naked_buf

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
            lex_errors.append(msg)
            trace('%s %s', msg, text)

        chunk = text[cursor:]
        buf.append(chunk)
        if chunk := ''.join(buf).strip():
            yield True, chunk
        return

    if block_starts:
        # Unclosed block! Reparse the inner of the first unclosed block to recover.
        msg = f'Unclosed block starting at position {block_starts} {cursor}'
        lex_errors.append(msg)
        trace('%s', msg)

        p = block_starts[0]
        assert p >= 0  # `block` mode cannot reach here.
        yield False, text[p]

        rest = text[p + 1 :]
        trace('Recovering rest: %s', rest)
        yield from lex(rest)
        return

    # `buf` must be empty, since `block_starts` is empty.
    assert not buf

    # Flush final text fragments.
    yield from text_fragments(cursor, len(text))
