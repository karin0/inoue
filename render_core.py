from contextlib import contextmanager
from typing import Callable, Iterable, Sequence, Type, TypeVar, cast

from lark import Lark, Token, Tree
from lark.visitors import Interpreter
from lark.exceptions import LarkError

from db import db
from util import log, notify, shorten
from render_context import (
    is_tracing,
    trace,
    Value,
    to_str,
    OverriddenDict,
    ScopedContext,
)

MAX_DEPTH = 20
MAX_GAS = 1000

lex_errors = []


# Find all "code blocks" and "block lines" - lines that ends with `;`, which are treated
# like a `block_inner`.
def lex(text: str, *, block: bool = False) -> Iterable[tuple[bool, str | None]]:
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

    line_start = 0
    line_buf = []

    yield_kind = None
    yield_buf = []

    def yield_text(start: int, end: int):
        if start >= end:
            return

        # Remove escaping `\` before `{`.
        for i in escaping_indices:
            assert start < i < end
            yield_buf.append(text[start:i])
            start = i + 1
        escaping_indices.clear()

        yield_buf.append(text[start:end])

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
            buf.append(chunk.replace("'", r"\'"))
            buf.append('\'')
            cursor = p

        yield_kind = None
        yield_buf.clear()

        match c:
            case '{':
                if not block_starts:
                    yield_kind = False
                    yield_text(cursor, p)
                    cursor = p

                block_starts.append(p)

            case '}':
                if not block_starts or (block and len(block_starts) == 1):
                    # An plain '}' as text, when no block is open.
                    # But, when `block` is set, we cannot close the first (pretended) block here,
                    # or stuff afterwards would be "leaked" as text fragments and injected into
                    # our output.
                    # We leave it to cause a Lark parse error later.
                    continue
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
                        yield_kind = True
                        yield_buf.append(chunk)
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
                        case '\n':
                            line_start = None
                            if line_buf:
                                yield from line_buf
                                line_buf.clear()

                # We only handles escaping '\' chars inside quotes (which is inside blocks)
                # or outside any blocks (which escapes '{' and '}').
                # The latter ones here must be removed from the output fragments.
                elif c == '\\':
                    escape = True
                    escaping_indices.append(p)
                elif c == '\n':
                    if line_start is not None:
                        line = text[line_start:p]
                        trace('Line: %r', line)

                        if (stem := line.rstrip()).endswith(';'):
                            # Naked block line inside inline text.
                            trace('Naked block line: %r\n  Dropped: %s', line, line_buf)
                            line_buf.clear()
                            escaping_indices.clear()

                            yield False, text[cursor:line_start]
                            cursor = p + 1
                            yield from lex(stem, block=True)

                            # We are actually consuming the line break here, but yielding a `\n`
                            # each time would result in too many empty lines for consecutive block lines.
                            # `None` hints the implicit line break after the naked block here.
                            yield False, None

                            line_start = p + 1
                            continue

                    if line_buf:
                        yield from line_buf
                        line_buf.clear()

                    yield_kind = False
                    yield_text(cursor, p + 1)
                    cursor = line_start = p + 1

        if yield_kind is not None:
            if any('\n' in part for part in yield_buf):
                for fragment in yield_buf:
                    yield yield_kind, fragment
            else:
                for fragment in yield_buf:
                    trace('Preserving inline fragment: %s %r', yield_kind, fragment)
                    line_buf.append((yield_kind, fragment))

    # Flush final line buffer, pretending a EOF.
    if not block and line_start is not None:
        line = text[line_start:]
        trace('Final Line: %r', line)

        if (stem := line.rstrip()).endswith(';'):
            trace('Final naked block line: %r\n  Dropped: %s', line, line_buf)
            yield False, text[cursor:line_start]
            yield from lex(stem, block=True)
            return

    if line_buf:
        yield from line_buf

    # Pretend a final `}` if `block` is set.
    if block:
        assert block_starts
        if len(block_starts) > 1:
            # There are NO text fragments or unclosed blocks in `block` mode, as the
            # entire input is treated as a single block, where we are only expected
            # to preprocess comments and raw literals with the `buf`.
            # Trying to recover would leak text fragments inside the block, so we
            # also leave it to cause a Lark parse error later.
            msg = f'Unbalanced line block starting at position {block_starts} {cursor}'
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
        if p >= 0:
            yield False, text[p]

        rest = text[p + 1 :]
        trace('Recovering rest: %s', rest)
        yield from lex(rest)
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


parser = Lark.open('dsl.lark', parser='lalr')

T = TypeVar('T', bound=Tree | Token)


def narrow(x: Tree | Token, target_type: Type[T]) -> T:
    assert isinstance(x, target_type), x
    return cast(T, x)


# LALR cannot distinguish VAR from LITERAL, so we have to strip manually.
def _iden(tree: Tree | Token) -> str:
    return narrow(tree, Token).value.strip()


class Abort(Exception):
    pass


class Exit(Exception):
    pass


class Engine(Interpreter):
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

        self._output: list[Value] = []
        self._root_output: list[str] = []
        self._depth: int = 0
        self._branch_depth: int = 0
        self._dirty: bool = False

        self._scope = ScopedContext(
            self.ctx,
            self._error,
            {
                '__file__': self._get_doc_func,
                'print': self._print_func,
                'exit': self._exit_func,
            },
        )
        self._tree: Tree | None = None
        self._aborted: bool = False
        self._gas: int = 0

    def _consume_gas(self):
        if self._gas >= MAX_GAS:
            self._error('out of gas')
            raise Abort()
        self._gas += 1

    def _put(self, text: str):
        if text:
            if not (self._dirty or text.isspace()):
                self._dirty = True
            self._output.append(text)

    # Keep the original type to preserve types in code_block as expression when possible.
    def _put_val(self, val: Value):
        if is_tracing and val != '':
            trace('Putting value: %r', val)
        if isinstance(val, str):
            self._put(val)
        else:
            self._dirty = True
            self._output.append(val)

    # The output implementation is responsible for filtering out any empty strings,
    # like in `_put` or `_unary_chain`.
    def _gather_output(self, out: list[Value], *, as_str: bool = False) -> Value:
        match len(out):
            case 0:
                return ''
            case 1:
                # Keep the original type for single output as possible.
                val = out[0]
                return to_str(val) if as_str else val
            case _:
                return ''.join(to_str(x) for x in out)

    def _render(self, text: str, strip: bool = False) -> str:
        for is_block, fragment in lex(text):
            if lex_errors:
                self.errors.extend(lex_errors)
                lex_errors.clear()

            if is_block:
                assert fragment is not None
                if self._aborted:
                    trace('Skipping aborted block: %r', fragment)
                    continue

                trace('Rendering block: %r', fragment)
                try:
                    tree = parser.parse(fragment)
                except LarkError as e:
                    self._error(f'parse: {fragment!r}: {type(e).__name__}: {e}')
                    continue

                if is_tracing:
                    trace('Parsed tree: %s', tree.pretty())

                try:
                    self.visit(tree)
                except Exit as e:
                    trace('Rendering exited at fragment: %s', shorten(fragment))
                    for val in e.args:
                        self._put(to_str(val))
                    continue
                except Abort:
                    with notify.suppress():
                        log.warning(
                            'Rendering aborted at fragment: %s', shorten(fragment)
                        )
                    self._aborted = True
                    continue
            else:
                trace('Appending text fragment: %r', fragment)
                if fragment is None:
                    # Merge consecutive line breaks from naked block lines without outputs.
                    if self._dirty:
                        self._output.append('\n')
                        self._dirty = False
                else:
                    self._put(fragment)

        r = self._gather_output(self._output, as_str=True)
        assert isinstance(r, str)
        trace('Rendered result: %r, Errors: %r', r, self.errors)

        if strip:
            return r.strip()

        r = r.lstrip()
        if len(s := r.rstrip()) != len(r) and s and r.find('\n', len(s)) >= 0:
            return s + '\n'

        return s

    def render(self, text: str) -> str:
        if not self.this_doc:
            self.this_doc = (None, text)
        # Internal document expansion does not trim spaces.
        r = self._render(text, strip=True)
        if is_tracing:
            trace('Final rendered result: %r', r)
            self.ctx.debug()
        return r

    def get_flag(self, key: str, default: bool = False) -> bool:
        val = self.ctx.get(key, default)
        return bool(val) and val != '0'

    def visit(self, tree: Tree):
        assert isinstance(tree, Tree)
        self._tree = tree

        if is_tracing:
            trace(
                '[%d %d] %s: %d %s',
                self._depth,
                self._gas,
                tree.data,
                len(tree.children),
                tree.children,
            )

        self._consume_gas()
        return super().visit(tree)

    # Override lark visitor to raise on unhandled nodes.
    def __getattr__(self, name: str):
        raise NotImplementedError(name)

    def start(self, tree: Tree):
        ch = narrow(tree.children[0], Tree)
        assert ch.data == 'block_inner'
        return self._block_inner(ch)

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
        trace('Push: %d %s', self._depth, self._output)
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
            trace('Pop: %d %s %s', self._depth, self._output, old_output)
            assert self._depth >= 0
            if self._depth == 0 and self._root_output:
                old_output.extend(self._root_output)
                self._root_output.clear()
            self._output = old_output
            self._branch_depth = old_branch_depth
            self._dirty = old_dirty

    def _print_func(self, *args):
        out = self._root_output if self._depth > 0 else self._output
        for val in args:
            out.append(to_str(val))

    def _exit_func(self, *args):
        raise Exit(*args)

    def _block_inner(self, tree: Tree):
        stmts = tree.children.pop()
        for ch in tree.children:
            if not isinstance(ch, Tree):
                continue
            match ch.data:
                # Db doc name set: {name:}
                case 'doc_def':
                    if self._depth == 0:
                        key = _iden(ch.children[0])
                        if self.doc_name:
                            self._error(f'doc name redefined: {self.doc_name} -> {key}')
                        else:
                            trace('Set doc name: %s', key)
                            self.doc_name = key
                # Db doc expand: {:doc} (same as {*doc} in unary expressions)
                case 'doc_ref':
                    iden = _iden(ch.children[-1])
                    self._put(self._doc_ref(iden))
                case _:
                    raise ValueError(f'Bad block_inner child: {ch}')

        stmts = narrow(stmts, Tree)
        assert stmts.data == 'stmt_list', stmts
        self._stmt_list(stmts)

    def _stmt_list(self, tree: Tree):
        # Flatten right-recursion.
        while True:
            if (stmt := tree.children[0]) is not None:
                self.visit(stmt)

            if (rest := tree.children[1]) is not None:
                rest = narrow(rest, Tree)
                assert rest.data == 'stmt_list'
                tree = rest
            else:
                break

    # Conditional branch: {<cond> ? true-directive : false-directive} (false part optional)
    def branch(self, tree: Tree):
        cond = narrow(tree.children[0], Tree)

        # Allow undefined vars as falsy in conditions.
        # Note that `$var ? 1 : 0` still emits errors when `var` is undefined.
        val = self._expr(cond, permissive=True, allow_undef=True)
        test = val and val != '0'

        # The associativity of `branch` is too greedy and consumes all statements
        # in a recursive `stmt_list`, like in `a ? b ; c; d` being parsed as
        # `a ? {b ; c ; d}`.
        # We keep only the first stmt here and execute the rest unconditionally.
        left = tree.children[1]
        right = tree.children[2]
        if tree.children[3] is None:
            if right is not None:
                right = narrow(right, Tree)
                assert right.data == 'stmt_list'
                right, rest = right.children
                visit = not test
            else:
                left = narrow(left, Tree)
                assert left.data == 'stmt_list'
                left, rest = left.children
                visit = test
        else:
            if (to := left if test else right) is not None:
                to = narrow(to, Tree)
                assert to.data == 'stmt_list'
                self._branch_depth += 1
                try:
                    self._stmt_list(to)
                finally:
                    self._branch_depth -= 1
            return

        if (to := left if test else right) is not None:
            to = narrow(to, Tree)
            self._branch_depth += 1
            try:
                if visit:
                    self.visit(to)
                else:
                    assert to.data == 'stmt_list'
                    self._stmt_list(to)
            finally:
                self._branch_depth -= 1

        if rest is not None:
            rest = narrow(rest, Tree)
            assert rest.data == 'stmt_list'
            self._stmt_list(rest)

    # Nested block: {{ ... }}
    # With scope: {@name {...}} (name optional, defaults to '_')
    def _code_block(self, tree: Tree):
        inner = narrow(tree.children[2], Tree)
        assert inner.data == 'block_inner'

        if (op := tree.children[0]) is not None:
            assert narrow(op, Token).type == 'SCOPE_OP'
            if (name := tree.children[1]) is None:
                name = ''
            elif isinstance(name, Tree):
                name = self._expr(name, as_str=True)
                assert isinstance(name, str), name
            else:
                name = _iden(name)

            self._scope.push(name)
            try:
                self._block_inner(inner)
            finally:
                self._scope.pop()
        else:
            self._block_inner(inner)

    def expr(self, tree: Tree):
        # Expression statement: {<expr>}
        # The value is directly appended to output.
        if isinstance(inner := tree.children[0], Tree):
            match inner.data:
                case 'assign_or_equal':
                    # Hijack the AOE "expression". Different semantics as statements.
                    return self._assign(inner)
                case 'unary_chain':
                    # Optimize: flatten the unary chain directly to output, without capturing.
                    return self._unary_chain(inner, put=self._put_val)
                case 'code_block':
                    # Optimize: flatten the nested block.
                    return self._code_block(inner)

        val = self._expr(tree, permissive=self._branch_depth == 0)
        self._put_val(val)

    def _evaluate(self, tree: Tree | Token | None, *, permissive: bool = False) -> str:
        if tree is None:
            return ''
        s = self._expr(narrow(tree, Tree), as_str=True, permissive=permissive)
        assert isinstance(s, str), s
        return s

    def _expr(
        self,
        tree: Tree,
        *,
        as_str: bool = False,
        permissive: bool = False,
        allow_undef: bool = False,
    ) -> Value:
        '''
        Naked literals are treated as context vars when `permissive` is True.
        so we write {some_var}, {show_var ? 1 : 0} and {show_var ? $some_var : literal}.
        '''
        if is_tracing:
            trace(
                '[%d] _expr: %s permissive=%s as_str=%s',
                self._gas,
                tree,
                permissive,
                as_str,
            )
        self._consume_gas()

        while True:
            assert len(tree.children) == 1
            if (ch := tree.children[0]) is None:
                # Empty parens: {( )}
                return ''
            tree = narrow(ch, Tree)
            if tree.data != 'expr':
                break
            # Flatten nested expressions: {( ( ... ) )}
            permissive = False

        ch = tree
        match ch.data:
            case 'unary_chain':
                out = []

                def put(val: Value):
                    if val != '':
                        out.append(val)

                self._unary_chain(ch, put=put)
                return self._gather_output(out, as_str=as_str)

            case 'assign_or_equal':
                return self._equal(ch)

            # Equality check: {key == val} (by string representation!)
            # `a==b` means `$a == 'b'`, and `a==$b` means `$a == $b`.
            case 'compare':
                left = self._evaluate(ch.children[0], permissive=True)
                right = self._evaluate(ch.children[1])
                return '1' if left == right else '0'

            # Nested block: {{ ... }}
            case 'code_block':
                if self._depth >= MAX_DEPTH:
                    self._error('block stack overflow')
                    raise Abort()

                with self._push():
                    self._code_block(ch)
                    return self._gather_output(self._output, as_str=as_str)

            case 'branch':
                if self._depth >= MAX_DEPTH:
                    self._error('block stack overflow')
                    raise Abort()

                with self._push():
                    self.branch(ch)
                    return self._gather_output(self._output, as_str=as_str)

            # Python expression: {"1 + 1"}
            # This should be non-mutating, i.e. side-effect free.
            case 'dq_lit':
                expr = (
                    narrow(ch.children[0], Token)
                    .value[1:-1]
                    .strip()
                    .replace('\\"', '"')
                    .replace('\\\\', '\\')
                )
                try:
                    val = self._scope.eval(expr)
                except Exit as e:
                    trace(f'evaluate: %r', e)
                    raise
                except Exception as e:
                    self._error(f'evaluate: {expr!r}: {type(e).__name__}: {e}')
                    return ''

                # Not necessarily str!
                return to_str(val) if as_str else val

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
                    if allow_undef:
                        _, val = self._scope.resolve_raw(key, as_str=as_str)
                        return '' if val is None else val
                    return self._scope.get(key, as_str=as_str)
                return key

            case _:
                raise ValueError(f'Bad expr: {tree.pretty()}')

    def _unary_chain(self, tree: Tree, *, put: Callable[[Value], None]):
        assert tree.children, tree
        for ch in tree.children:
            val = self._unary(narrow(ch, Tree))
            put(val)

    def _unary(self, tree: Tree, *, as_str: bool = False) -> Value:
        op = narrow(tree.children[0], Token).value
        key = _iden(tree.children[1])
        trace('Unary: %s %s', op, key)

        match op:
            case '+':
                self.ctx.setdefault_override(key, '1')
                return ''

            case '-':
                self.ctx.setdefault_override(key, '0')
                return ''

            # Variable read: {$key}
            case '$':
                return self._scope.get(key, as_str=as_str)

            # Db doc expand: {*doc} (same as {:doc})
            case '*':
                return self._doc_ref(key)

            case _:
                raise ValueError(f'Bad deref op: {op}')

    def _get_doc(self, key: str) -> tuple[int | None, str]:
        # Allow self-reference in `handle_render` and `handle_render_doc`,
        # where the doc is not saved yet.
        # Note that `doc_id` can be None in `handle_render`, where the
        # `doc_name` is transient and only used for recursion.
        if row := self.doc_name == key and self.this_doc or db.get_doc(key):
            return row
        self._error('undefined doc: ' + key)
        return None, ''

    # Used in dq_lit evaluation as `__file__` function.
    def _get_doc_func(self, key=None) -> str:
        if isinstance(key, str):
            return self._get_doc(key)[1]

        if isinstance(key, int):
            if self.first_doc_id is None:
                self._error('unknown doc')
                return ''
            if (doc := db.get_doc_by_id(self.first_doc_id)) is None:
                self._error(f'missing doc: {self.first_doc_id}')
                return ''
            return doc

        # Self-mapping when no argument is given.
        if self.this_doc:
            return self.this_doc[1]

        self._error('unknown doc')
        return ''

    def _doc_ref(self, key: str) -> str:
        doc_id, doc = self._get_doc(key)

        if self.first_doc_id is None:
            self.first_doc_id = doc_id

        # Recursion is allowed up to a limit.
        if self._depth >= MAX_DEPTH:
            self._error('stack overflow')
            raise Abort()

        with self._push():
            trace('Rendering doc: %s %s', key, doc_id)
            return self._render(doc)

    # Due to LALR limitations, AOE chains have very different semantics:
    # - In expression statements, they set context vars and return ''.
    # - Otherwise, they check whether all the assigned vars equal the final value, and return '1' or '0'.
    # But in both cases, the ASSIGN_OP besides the first one must be '='.
    def _resolve_aoe_chain(self, tree: Tree) -> tuple[Sequence[str], str, Tree | None]:
        keys = [_iden(tree.children[0])]
        op = narrow(tree.children[1], Token).value
        val = None

        while rest := tree.children[2]:
            rest = narrow(rest, Tree)
            assert rest.data == 'expr'
            tree = rest.children[0]
            if not (isinstance(tree, Tree) and tree.data == 'assign_or_equal'):
                val = rest
                break
            keys.append(_iden(tree.children[0]))
            if narrow(tree.children[1], Token).value != '=':
                self._error('bad assign op in chain: ' + op)

        return keys, op, val

    # Do real assignments here.
    def _assign(self, tree: Tree):
        keys, op, expr = self._resolve_aoe_chain(tree)
        trace('Assign: keys=%s op=%r expr=%s', keys, op, expr)

        match op:
            case '=':
                # Context set: {key=value}
                # {key1=key2=...=value} sets all keys to value
                f = self.ctx.__setitem__

            case ':=':
                # Context override: {key:=value}
                # a. Can never be overridden (except by markup buttons)
                # b. Does not interrupt the natural order of flags detected in markup
                # {key1:=key2=...=value} affects all keys
                f = self.ctx.setdefault_override

            case '?=':
                # Context set if empty (undefined or falsy): {key?=value}
                # {key1?=key2=...=value} assigns the value to the empty keys only
                # Note that the evaluation is deferred (conditional) in this case,
                # like a `OnceCell` initialization.
                val = '' if expr is None else None
                for key in keys:
                    key = self._scope.current_key(key)
                    old = self.ctx.touch(key)
                    if not old or old == '0':
                        if val is None:
                            val = self._expr(expr)  # type: ignore[assignment]
                        self.ctx[key] = val
                    else:
                        trace('"?=": Skip for non-empty key: %s = %r', key, old)
                if val is None:
                    trace('"?=": Skipped evaluation for %s', keys)
                return

            case _:
                raise ValueError(f'Bad assign op: {tree.pretty()}')

        val = self._expr(expr) if expr else ''
        for key in keys:
            # Cannot assign to outer scopes. No `global` or `nonlocal`.
            # However, outer vars can still be "mutated" via `swap` or `repl`.
            self._scope.set(key, val, f)

    # Equality test: {key1=key2=...=<expr>}, but not as expression statements.
    def _equal(self, tree: Tree) -> str:
        keys, op, expr = self._resolve_aoe_chain(tree)
        trace('Equal: keys=%s op=%r expr=%s', keys, op, expr)
        if op != '=':
            self._error('bad assign op in equality test: ' + op)

        final_val = self._expr(expr, as_str=True) if expr else ''
        for key in keys:
            val = self._scope.get(key, as_str=True)
            if val != final_val:
                return '0'

        return '1'

    # Context swap: {key1^key2}
    # To be "atomic", This allows modifying vars in outer scopes.
    def swap(self, tree: Tree):
        key1 = _iden(tree.children[0])
        key2 = _iden(tree.children[1])
        key1, val1 = self._scope.resolve_raw(key1)
        key2, val2 = self._scope.resolve_raw(key2)
        self._scope.set_or_del_raw(key1, val2)
        self._scope.set_or_del_raw(key2, val1)

    # Context inplace replacement: {key|pat1/rep1|pat2/rep2|...}
    def repl(self, tree: Tree):
        key = _iden(tree.children[0])
        key, val = self._scope.resolve_raw(key, '', as_str=True)
        assert isinstance(val, str), val
        for pair in tree.children[1:]:
            pat, sub = pair.children
            pat = self._evaluate(pat)
            sub = self._evaluate(sub)

            # Write back each time.
            self.ctx[key] = val = val.replace(pat, sub)
