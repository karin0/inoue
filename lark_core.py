import logging
from contextlib import contextmanager
from typing import Iterator, Type, TypeVar, cast

from simpleeval import simple_eval
from lark import Lark, Token, Tree
from lark.visitors import Interpreter
from lark.exceptions import LarkError

from db import db
from util import log, notify, shorten
from render_core import OverriddenDict, Value, MAX_DEPTH, to_str

is_debug = log.isEnabledFor(logging.DEBUG)

parser = Lark.open('dsl.lark', parser='lalr')

lex_errors = []


# First stage: find all "block lines" - lines that ends with `;`, which are treated
# like a `block_inner`.
# For the other lines, we lex them again to find the real blocks `{ ... }` inside.
def lex(text: str) -> Iterator[tuple[bool, str | None]]:
    lines = []
    for line in text.splitlines(keepends=True):
        if (stem := line.rstrip()).endswith(';'):
            if lines:
                yield from _lex(''.join(lines))
                lines.clear()
            # Only `_lex` handles comments.
            yield from _lex(stem, block=True)
            # We are actually consuming the line break here, but yielding a `\n`
            # each time would result in too many empty lines for consecutive block lines.
            # `None` hints the implicit line break after the naked block here.
            yield False, None
        else:
            lines.append(line)

    if lines:
        yield from _lex(''.join(lines))


# Thanks to the first stage, we no longer care about line endings.
def _lex(text: str, *, block: bool = False) -> Iterator[tuple[bool, str]]:
    block_starts = []
    if block:
        # Pretend a `{` before the start.
        block_starts.append(-1)

    buf = []  # buffered fragments inside code blocks
    quote = None
    raw = False
    comment = False
    escape = False
    escaping_indices = []
    cursor = 0  # number of chars processed

    for p, c in enumerate(text):
        if escape:
            escape = False
            continue

        if comment:
            if c != '\n' and c != ';' and c != '{' and c != '}':
                continue
            # Skip the comment.
            cursor = p
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

        # Raw literals starting with '`' are NOT enclosed.
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
            r = chunk.replace("'", r"\'")
            log.debug('Raw literal:%s\n->\n%s', chunk, r)
            buf.append('\'')
            buf.append(r)
            buf.append('\'')
            cursor = p

        match c:
            case '{':
                if not block_starts:
                    c = cursor
                    # Remove escaping `\` before `{`.
                    for i in escaping_indices:
                        assert c < i < p
                        yield False, text[c:i]
                        c = i + 1
                    escaping_indices.clear()
                    if c < p:
                        yield False, text[c:p]
                    cursor = p

                block_starts.append(p)

            case '}':
                if not block_starts:
                    continue
                block_starts.pop()
                if not block_starts:
                    chunk = text[cursor:p]
                    buf.append(chunk)
                    # Remove the block start `{`.
                    # We cannot save `p+1` directly when `{` is found, in case of
                    # unclosed blocks that are recovered as final texts later.
                    chunk = ''.join(buf)[1:].strip()
                    buf.clear()
                    if chunk:
                        yield True, chunk
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
                            cursor = p
                        case '`':
                            raw = True
                            # Flush up before the raw literal.
                            if chunk := text[cursor:p]:
                                buf.append(chunk)
                            cursor = p

                # We only handles escaping '\' chars inside quotes (which is inside blocks)
                # or outside any blocks (which escapes '{' and '}').
                # The latter ones here must be removed from the output fragments.
                elif c == '\\':
                    escape = True
                    escaping_indices.append(p)

    # Pretend a final `}` if `block` is set.
    if block and len(block_starts) == 1:
        chunk = text[cursor:]
        buf.append(chunk)
        if chunk := ''.join(buf).strip():
            yield True, chunk
        return

    if block_starts:
        # Unclosed block! Reparse the inner of the first unclosed block to recover.
        msg = f'Unclosed block starting at position {block_starts} {cursor}'
        lex_errors.append(msg)
        log.debug('%s', msg)

        p = block_starts[0]
        if p >= 0:
            yield False, text[p]
        yield from _lex(text[p + 1 :])
        return

    # `buf` must be empty, since `block_starts` is empty.
    assert not buf

    c = cursor
    for i in escaping_indices:
        assert i > c
        yield False, text[c:i]
        c = i + 1

    if chunk := text[c:]:
        yield False, chunk


T = TypeVar('T', bound=Tree | Token)


def narrow(x: Tree | Token, target_type: Type[T]) -> T:
    assert isinstance(x, target_type), x
    return cast(T, x)


# LALR cannot distinguish VAR from LITERAL, so we have to strip manually.
def _iden(tree: Tree | Token) -> str:
    return narrow(tree, Token).value.strip()


class Abort(Exception):
    pass


class RenderInterpreter(Interpreter):
    def __init__(
        self,
        overrides: dict[str, Value],
        *,
        this_doc: tuple[int | None, str] | None = None,
    ):
        super().__init__()
        self.ctx = OverriddenDict(overrides)
        self.this_doc = this_doc

        self.errors: list[str] = []
        self.doc_name: str | None = None
        self.first_doc_id: int | None = None

        self._output: list[str] = []
        self._tree: Tree | None = None
        self._depth: int = 0
        self._branch_depth: int = 0
        self._dirty: bool = False

    def _render(self, text: str) -> str:
        aborted = False
        for is_block, fragment in lex(text):
            if lex_errors:
                self.errors.extend(lex_errors)
                lex_errors.clear()

            if is_block:
                assert fragment is not None
                if aborted:
                    log.debug('Skipping aborted block: %r', fragment)
                    continue

                log.debug('Rendering block: %r', fragment)
                try:
                    tree = parser.parse(fragment)
                except LarkError as e:
                    self._error(f'parse: {fragment!r}: {type(e).__name__}: {e}')
                    continue

                if is_debug:
                    log.debug('Parsed tree: %s', tree.pretty())

                n = len(self._output)
                try:
                    self.visit(tree)
                except Abort:
                    with notify.suppress():
                        log.error('Parsing aborted at fragment: %s', shorten(fragment))
                    aborted = True
                    continue

                for s in self._output[n:]:
                    if s and not s.isspace():
                        self._dirty = True
                        break
            else:
                log.debug('Appending text fragment: %r', fragment)
                if fragment is None:
                    # Merge consecutive line breaks from naked block lines without outputs.
                    if self._dirty:
                        self._output.append('\n')
                        self._dirty = False
                else:
                    self._output.append(fragment)

        r = ''.join(self._output)
        log.debug('Rendered result: %r, Errors: %r', r, self.errors)
        return r

    def render(self, text: str) -> str:
        if not self.this_doc:
            self.this_doc = (None, text)
        # Internal document expansion does not trim spaces.
        return self._render(text).strip()

    def get_flag(self, key: str, default: bool = False) -> bool:
        val = self.ctx.get(key, default)
        return bool(val) and val != '0'

    def visit(self, tree: Tree):
        assert isinstance(tree, Tree)
        log.debug('Visit %s: %d %s', tree.data, len(tree.children), tree.children)
        self._tree = tree
        return super().visit(tree)

    def _tree_ctx(self) -> str:
        if self._tree is None:
            return 'unknown'
        return str(self._tree.pretty(indent_str='').replace('\n', ' '))

    def _error(self, msg: str):
        ctx = self._tree_ctx()
        log.debug('Error: %s: %s', msg, ctx)
        self.errors.append(f'{msg}: {ctx}')

        if len(self.errors) > 5:
            self.errors.append('too many errors, aborting')
            raise Abort()

    @contextmanager
    def _push(self):
        log.debug('Push: %d %s', self._depth, self._output)
        old_output = self._output
        old_branch_depth = self._branch_depth
        old_dirty = self._dirty
        self._depth += 1
        self._output = []
        self._branch_depth = 0
        self._dirty = False
        try:
            yield
        finally:
            self._depth -= 1
            log.debug('Pop: %d %s %s', self._depth, self._output, old_output)
            self._output = old_output
            self._branch_depth = old_branch_depth
            self._dirty = old_dirty

    def block_inner(self, tree: Tree):
        if ch := tree.children[0]:
            # Db doc name set: {name:}
            ch = narrow(ch, Tree)
            assert ch.data == 'doc_def', ch
            if self._depth == 0:
                if self.doc_name:
                    self._error('doc name redefined')
                else:
                    key = _iden(ch.children[0])
                    log.debug('Set doc name: %s', key)
                    self.doc_name = key

        if len(tree.children) > 1 and (ch := tree.children[-1]):
            if ch.data == 'stmt_list':
                self.stmt_list(ch)
            else:
                raise ValueError(f'Bad block_inner child: {ch}')

    def stmt_list(self, tree: Tree[Tree]):
        if (stmt := tree.children[0]) is None:
            return
        stmt = narrow(stmt, Tree)
        match stmt.data:
            case 'stmt':
                assert len(stmt.children) == 1
                ch = narrow(stmt.children[0], Tree)
                self.visit(ch)
            case 'stmt_list':
                self.stmt_list(stmt)
            case _:
                raise ValueError(f'Bad stmt_list child: {stmt}')

        if len(tree.children) > 1 and (rest := tree.children[1]):
            rest = narrow(rest, Tree)
            assert rest.data == 'stmt_list'
            self.stmt_list(rest)

    def stmt(self, tree: Tree):
        raise RuntimeError

    def expr(self, tree: Tree):
        raise RuntimeError

    def _get_var(self, key: str, *, as_str: bool = False) -> Value:
        if (val := self.ctx.get(key)) is None:
            self._error('undefined: ' + key)
            return ''
        log.debug('Got var: %s = %r', key, val)
        return to_str(val) if as_str else val

    def _expr(
        self, tree: Tree | None, *, as_str: bool = False, permissive: bool = False
    ) -> Value:
        '''
        Naked literals are treated as context vars when `permissive` is True.
        so we write {some_var}, {show_var ? 1 : 0} and {show_var ? $some_var : default literal}.
        '''
        if not tree:
            return ''
        assert len(tree.children) == 1
        ch = narrow(tree.children[0], Tree)
        match ch.data:
            case 'deref':
                return self._deref(ch, as_str=as_str)

            # Equality check: {key == val} (by string representation!)
            # `a==b` means `$a == 'b'`, and `a==$b` means `$a == $b`.
            case 'compare':
                left = self._expr(
                    narrow(ch.children[0], Tree), permissive=True, as_str=True
                )
                right = self._expr(narrow(ch.children[1], Tree), as_str=True)
                return '1' if left == right else '0'

            # Nested block: {{ ... }}
            case 'block_inner':
                if self._depth >= MAX_DEPTH:
                    self._error('block stack overflow')
                    return ''

                with self._push():
                    self.block_inner(ch)
                    return ''.join(self._output)

            # Python expression: {"1 + 1"}
            case 'dq_lit':
                expr = (
                    narrow(ch.children[0], Token)
                    .value[1:-1]
                    .strip()
                    .replace('\\"', '"')
                    .replace('\\\\', '\\')
                )
                try:
                    v = simple_eval(expr, names=self.ctx.clone())
                except Exception as e:
                    self._error(f'evaluate: {expr!r}: {type(e).__name__}: {e}')
                    v = ''
                else:
                    log.debug('evaluated script: %s -> %s', expr, v)
                    # Ensure a `Value`.
                    if v is None:
                        return ''

                # Not necessarily str!
                return to_str(v) if as_str else v

            # Literal: {'single quoted'}
            case 'sq_lit':
                return (
                    narrow(ch.children[0], Token)
                    .value[1:-1]
                    .replace("\\'", "'")
                    .replace('\\\\', '\\')
                )

            # Ambiguous naked literal / var: {some_name}
            # Treated as var only if `permissive` is True and contains no whitespace.
            case 'naked_lit':
                key = _iden(ch.children[0])
                if permissive and not any(c.isspace() for c in key):
                    return self._get_var(key, as_str=as_str)
                return key

            case _:
                raise ValueError(f'Bad literal: {tree}')

    def _deref(self, tree: Tree, *, as_str: bool = False) -> Value:
        op = narrow(tree.children[0], Token).value
        key = _iden(tree.children[1])

        # Context var get: {$name}
        if op == '$':
            return self._get_var(key, as_str=as_str)

        # Db doc get: {:name} or {*name}

        # Allow self-reference in `handle_render` and `handle_render_doc`,
        # where the doc is not saved yet.
        # Note that `doc_id` can be None in `handle_render`, where the
        # `doc_name` is transient and only used for recursion.
        if row := self.doc_name == key and self.this_doc or db.get_doc(key):
            doc_id, doc = row
        else:
            self._error('undefined doc: ' + key)
            return ''

        if self.first_doc_id is None:
            self.first_doc_id = doc_id

        # Recursion is allowed up to a limit.
        if self._depth >= MAX_DEPTH:
            self._error('stack overflow')
            return ''

        with self._push():
            log.debug('Rendering doc: %s %s', key, doc_id)
            return self._render(doc)

    def branch_or_assign_or_expr(self, tree: Tree):
        assert len(tree.children) <= 3
        branch = tree.children[1] if len(tree.children) > 1 else None
        ch = narrow(tree.children[0], Tree)

        if branch:
            # Conditional branch: {<cond> ? true-directive : false-directive} (false part optional)
            return self._branch(tree, ch, branch)

        match ch.data:
            case 'assign':
                # Real assignments.
                self._assign(ch)
            case 'expr':
                # Expression statement: {<expr>}
                # The value is directly appended to output.
                val = self._expr(ch, permissive=self._branch_depth == 0, as_str=True)
                assert isinstance(val, str), val
                self._output.append(val)

    def _branch(self, tree: Tree, cond: Tree, branch: Tree | Token):
        match cond.data:
            case 'expr':
                # {<expr> ? ...}
                val = self._expr(cond, permissive=True)
                test = val and val != '0'

            case 'assign':
                # Equality test: {key=<val> ? ...}
                right = cond.children[1]
                if not isinstance(right, Tree):
                    return self._error('condition equality must use =')
                if right.data != 'expr':
                    return self._error('condition equality cannot chain')
                left = cond.children[0]
                left_val = self._get_var(_iden(left), as_str=True)
                right_val = self._expr(right, as_str=True)
                test = left_val == right_val

            case _:
                raise ValueError(f'Bad branch cond: {cond}')

        if not test and not (branch := tree.children[2]):
            return

        to = narrow(branch, Tree)
        assert to.data == 'stmt_list'
        self._branch_depth += 1
        self.stmt_list(to)
        self._branch_depth -= 1

    def assign_part(self, tree: Tree):
        raise RuntimeError

    def assign(self, tree: Tree):
        raise RuntimeError

    def _assign_part(self, tree: Tree, out_keys: list[str]) -> Value:
        key = _iden(tree.children[0])
        out_keys.append(key)

        if (rest := tree.children[1]) is None:
            return ''

        match rest.data:
            case 'assign_part':
                return self._assign_part(rest, out_keys)
            case 'expr':
                return self._expr(rest)
            case _:
                raise ValueError(f'Bad assign rest: {rest}')

    def _assign(self, tree: Tree):
        rest = tree.children[-1]
        if isinstance(rest, Token):
            rest = None

        # Context set: {key=value}
        # {key1=key2=...=value} sets all keys to value
        if len(tree.children) < 2:
            # A special case: `iden = <empty expr>`.
            f = self.ctx.__setitem__
        elif isinstance((op := tree.children[1]), Token):
            match op.value:
                # Context set if empty: {key?=value}
                # {key1?=key2=...=value} sets all keys to value if empty
                case '?=':
                    f = self.ctx.setdefault

                # Context override: {key:=value}
                # a. Can never be overridden (except by markup buttons)
                # b. Does not interrupt the natural order of flags detected in markup
                # {key1:=key2=...=value} affects all keys
                case ':=':
                    f = self.ctx.setdefault_override

                case _:
                    raise ValueError(f'Bad assign op: {tree}')
        else:
            f = self.ctx.__setitem__

        keys = [_iden(tree.children[0])]

        if rest is None:
            val = ''
        else:
            match rest.data:
                case 'assign_part':
                    val = self._assign_part(rest, keys)
                case 'expr':
                    val = self._expr(rest)
                case _:
                    raise ValueError(f'Bad assign rest: {rest}')

        for key in keys:
            log.debug('Assigning: %s = %r', key, val)
            f(key, val)

    def _set_var(self, key: str, val: Value | None):
        if val is None:
            self.ctx.pop(key, None)
        else:
            self.ctx[key] = val

    # Context swap: {key1^key2}
    def swap(self, tree: Tree):
        key1 = _iden(tree.children[0])
        key2 = _iden(tree.children[1])
        val1 = self.ctx.get(key1)
        val2 = self.ctx.get(key2)
        self._set_var(key1, val2)
        self._set_var(key2, val1)

    # Context inplace replacement: {key|pat1/rep1|pat2/rep2|...}
    def repl(self, tree: Tree):
        key = _iden(tree.children[0])
        val = self._get_var(key, as_str=True)
        assert isinstance(val, str)
        for pair in tree.children[1:]:
            match len(pair.children):
                case 2:
                    pat, sub = pair.children
                    pat = self._expr(pat, as_str=True)
                    assert isinstance(pat, str)
                    sub = self._expr(sub, as_str=True)
                    assert isinstance(sub, str)
                case 1:
                    pat = pair.children[0]
                    pat = self._expr(pat, as_str=True)
                    assert isinstance(pat, str)
                    sub = ''
                case _:
                    continue
            val = val.replace(pat, sub)
        self.ctx[key] = val

    def unary_chain(self, tree: Tree):
        for ch in tree.children:
            ch = narrow(ch, Tree)
            match ch.data:
                case 'flag_set':
                    key = _iden(ch.children[0])
                    self.ctx.setdefault_override(key, '1')
                case 'flag_unset':
                    key = _iden(ch.children[0])
                    self.ctx.setdefault_override(key, '0')
                case 'deref':
                    r = self._deref(ch, as_str=True)
                    assert isinstance(r, str)
                    self._output.append(r)
                case _:
                    raise ValueError(f'Bad unary_chain: {ch}')
