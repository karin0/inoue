import os
import sys
import functools
import traceback
from contextlib import contextmanager
from typing import (
    Any,
    Callable,
    Mapping,
    Sequence,
    Type,
    TypeVar,
    Literal,
    cast,
    overload,
    override,
)

from lark import Lark, Token, Tree
from lark.visitors import Interpreter
from lark.exceptions import LarkError

from .lex import lex, lex_errors
from .context import (
    is_tracing,
    is_not_quiet,
    trace,
    log,
    Value,
    Box,
    Items,
    to_str,
    try_to_str,
    try_to_value,
    OverriddenDict,
    ScopedContext,
    ContextCallbacks,
)
from .tco import TCO, MaybeTCO, Tco, TCOContext

MAX_DEPTH = 20
MAX_GAS = 2000


# https://stackoverflow.com/a/47956089
def stack_size2a(size=2):
    from itertools import count

    """Get stack size for caller's frame."""
    frame = sys._getframe(size)

    for size in count(size):
        frame = frame.f_back
        if not frame:
            return size


def shorten(text: str, max_len: int = 64) -> str:
    if len(text) <= max_len:
        return text
    return text[: max_len // 2] + '...' + text[-(max_len // 2) :]


parser = Lark.open(
    os.path.join(os.path.dirname(__file__), 'dsl.lark'),
    parser='lalr',
    start='block_inner',
)


@functools.lru_cache
def parse_fragment(text: str) -> Tree:
    return parser.parse(text)


T = TypeVar('T', bound=Tree | Token)


if is_tracing:

    def narrow(  # pyright: ignore[reportRedeclaration]
        x: Tree | Token, target_type: Type[T]
    ) -> T:
        assert isinstance(x, target_type), x
        return cast(T, x)

else:

    def narrow(  # pyright: ignore[reportRedeclaration]
        x: Tree | Token, target_type: Type[T]
    ) -> T:
        return cast(T, x)


# Abort the entire rendering, until the "root" document.
class Abort(Exception):
    pass


# "exit()", stop rendering the current fragment only.
class Exit(Abort):
    pass


class SubDoc(Box):
    def __init__(self, tree: Tree, params: tuple[str, ...] | None):
        self.tree = tree
        self.params = params

    @override
    def __repr__(self) -> str:
        return f'{{{self.params} ↦ {Engine.debug_node(self.tree)}}}'


class Engine(Interpreter, ContextCallbacks):
    def __init__(
        self,
        ctx: Mapping[str, Value] | None = None,
        overrides: dict[str, Value] | None = None,
        doc_loader: Callable[[str], str | None] | None = None,
        *,
        funcs: Mapping[str, Callable[..., Value | None]] | None = None,
    ):
        super().__init__()
        if ctx is None:
            ctx = {}
        if overrides is None:
            overrides = {}
        self._ctx = OverriddenDict(ctx, overrides)
        self._doc_src = doc_loader or (lambda _: None)
        self._doc_text = ''

        self.errors: list[str] = []
        self.doc_name: str | None = None

        self._output: list[Value] = []
        self._root_output: list[str] = []
        self._depth: int = 0
        self._dirty: bool = False

        self._tco: TCOContext | None = None

        self._tree: Tree | None = None
        self._gas: int = 0

        eval_funcs = {
            '__file__': self._get_doc_func,
            'print': self._print_func,
            'exit': self._exit_func,
        }
        if funcs:
            eval_funcs.update(funcs)
        self._scope = ScopedContext(self._ctx, self, eval_funcs)

    @override
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
    def _put_val(self, val: Value | TCO) -> MaybeTCO:
        if is_tracing and val != '':
            trace('_put_val: %r', val)
        if val is Tco:
            # Manual "stack unwinding" for TCO.
            return Tco
        if isinstance(val, str):
            self._put(val)
        else:
            self._dirty = True
            self._output.append(val)

    @overload
    def _gather_output(
        self,
        out: list[Value],
        default: Literal[None],
        *,
        as_str: Literal[False] = False,
        trim: bool = False,
    ) -> Value | None: ...

    @overload
    def _gather_output(
        self,
        out: list[Value],
        default: Literal[None],
        *,
        as_str: Literal[True],
        trim: bool = False,
    ) -> str | None: ...

    @overload
    def _gather_output(
        self,
        out: list[Value],
        default: Value = '',
        *,
        as_str: Literal[False] = False,
        trim: bool = False,
    ) -> Value: ...

    @overload
    def _gather_output(
        self,
        out: list[Value],
        default: Value = '',
        *,
        as_str: Literal[True],
        trim: bool = False,
    ) -> str: ...

    # The output implementation is responsible for filtering out any empty strings,
    # like in `_put` or `_unary_chain`.
    def _gather_output(
        self,
        out: list[Value],
        default: Value | None = '',
        *,
        as_str: bool = False,
        trim: bool = False,
    ) -> Value | None:
        match len(out):
            case 0:
                return default
            case 1:
                # Keep the original type for single output as possible.
                val = out[0]
                if as_str:
                    val = to_str(val)
                    return self._trim_output(val) if trim else val
                elif trim and isinstance(val, str):
                    return self._trim_output(val)
                else:
                    return val
            case _:
                val = ''.join(to_str(x) for x in out)
                return self._trim_output(val) if trim else val

    def _trim_output(self, s: str) -> str:
        r = s.lstrip()
        s = r.rstrip()
        if len(s) != len(r) and s and r.find('\n', len(s)) >= 0:
            return s + '\n'
        return s

    def _render_tree(self, tree: Tree, fragment: str, root: bool):
        assert tree.data == 'block_inner', tree

        # Entry of the syntax: '{ ... }', or '...;' in a single line.
        # Works like a simpler `_code_block` without scope modifier.
        sub_doc_token = self._scan_block(tree, root=root)
        if sub_doc_token is not None:
            self._sub_doc(tree, sub_doc_token)
            return

        try:
            self._run_block(tree)
        except Exit:
            # "exit()" called: stop rendering, but only for *this fragment*.
            # Fragments afterwards and outer docs will continue rendering.
            trace('Rendering exited at fragment: %s', shorten(fragment))
        except Abort:
            # More serious "exit": stop the entire rendering, skipping all fragments,
            # this and all outer docs.
            if is_not_quiet:
                log.warning('Rendering aborted at fragment: %s', shorten(fragment))
            raise

    def _render(self, text: str, *, root: bool = False):
        for is_block, fragment in lex(text):
            if lex_errors:
                self.errors.extend(lex_errors)
                lex_errors.clear()

            if is_block:
                assert fragment is not None
                if is_not_quiet:
                    trace('Rendering block: %r', fragment)

                misses = parse_fragment.cache_info().misses
                try:
                    tree = parse_fragment(fragment)
                except LarkError as e:
                    self._error(f'parse: {fragment}: {type(e).__name__}: {e}')
                    continue

                new_misses = parse_fragment.cache_info().misses

                if is_tracing:
                    if new_misses > misses:
                        trace('Parsed tree (%s): %s', len(fragment), tree.pretty())
                        # trace('Parsed tree (%s)', len(fragment))
                    else:
                        trace('Cached tree (%s)', len(fragment))

                self._render_tree(tree, fragment, root)
            else:
                if is_not_quiet:
                    trace('Appending text fragment: %r', fragment)
                if fragment is None:
                    # Merge consecutive line breaks from naked block lines without outputs.
                    if self._dirty:
                        self._output.append('\n')
                        self._dirty = False
                else:
                    self._put(fragment)

        trace('Rendered result: %r', self._output)

    def render(self, text: str) -> str:
        self._doc_text = text = text.strip()
        # Internal document expansion does not trim spaces.
        try:
            self._render(text, root=True)
        except Abort:
            trace('Rendering aborted.')
        r = self._gather_output(self._output, as_str=True)
        if is_tracing:
            trace('Final rendered result: %r\nGas cost: %s', r, self._gas)
            self._ctx.debug()
        return r.strip()  # type: ignore

    # Flatten view without scopes to act as a MutableMapping. For external use only.
    def __getitem__(self, key: str) -> Value:
        return self._ctx[key]

    def __setitem__(self, key: str, value: Value) -> None:
        self._ctx[key] = value

    def get_flag(self, key: str, default: bool = False) -> bool:
        val = self._ctx.get(key, default)
        return bool(val) and val != '0'

    def get(self, key: str, default: Value | None = None) -> Value | None:
        return self._ctx.get(key, default)

    def items(self) -> Items:
        return self._ctx.items()

    def setdefault_override(self, key: str, value: Value) -> Value:
        return self._ctx.setdefault_override(key, value)

    MAX_DEBUG_DEPTH = 2 if is_not_quiet else 1

    @classmethod
    def debug_node(
        cls, node: Tree | Token | tuple[Tree, str] | None, *, depth: int = 0
    ) -> str:
        if node is None:
            return '/'
        if isinstance(node, Token):
            return f'{node.type}({node.value.strip()})'
        if isinstance(node, tuple):
            # Cached node from `_scan_code_block`.
            return f'Cached({node[0].data}, {node[1]})'
        assert isinstance(node, Tree), node
        nr = len(node.children)
        if nr > 1:
            if depth > cls.MAX_DEBUG_DEPTH:
                return f'{node.data}[{nr}]'
            nr = f'[{nr}]'
        else:
            nr = ''
        chs = ', '.join(cls.debug_node(ch, depth=depth + 1) for ch in node.children)
        return f'{node.data}{nr}({chs})'

    # This interface is only used for `stmt_list` and all types of `stmt` nodes.
    def visit(
        self, tree: Tree, *, direct_branch: bool = False, allow_tco: bool = False
    ) -> MaybeTCO:
        assert isinstance(tree, Tree)
        self._tree = tree

        self._consume_gas()
        self._trace(tree, direct_branch=direct_branch, allow_tco=allow_tco)

        match tree.data:
            case 'expr':
                return self.expr(tree, direct_branch=direct_branch, allow_tco=allow_tco)
            case 'stmt_list':
                return self.stmt_list(
                    tree, direct_branch=direct_branch, allow_tco=allow_tco
                )
            case 'branch':
                return self.branch(tree, allow_tco=allow_tco)
            case _:
                return super().visit(tree)

    # Override lark visitor to raise on unhandled nodes.
    @override
    def __getattr__(self, name: str):
        raise NotImplementedError(name)

    def _tree_ctx(self) -> str:
        if self._tree is None:
            return 'unknown'
        return self.debug_node(self._tree)

    @override
    def _error(self, msg: str):
        ctx = self._tree_ctx()
        log.debug('Error: %s: %s', msg, ctx)
        if is_tracing:
            trace('%s', ''.join(traceback.format_stack()))

        self.errors.append(f'{msg}: {ctx}')

        if len(self.errors) > 5:
            self.errors.append('too many errors, aborting')
            raise Abort()

    # Note that this is only used for output capturing before entering a block.
    # Scopes are completely separated from this mechanism.
    @contextmanager
    def _push(self):
        # Recursion is allowed up to a limit.
        old_output = self._output
        old_dirty = self._dirty
        sys_depth = stack_size2a()
        trace(
            '[%s] Push @ %s: %s (%s)',
            self._depth,
            self._scope.current(),
            old_output,
            sys_depth,
        )

        if self._depth >= MAX_DEPTH:
            self._error('stack overflow')
            raise Abort()

        self._depth += 1
        self._output = []
        self._dirty = False
        err = False
        try:
            yield
        except Exception:
            err = True
            raise
        finally:
            self._depth -= 1
            trace(
                '[%s] Pop @ %s: %s %s',
                self._depth,
                self._scope.current(),
                old_output,
                self._output,
            )
            assert self._depth >= 0
            if self._depth == 0 and self._root_output:
                old_output.extend(self._root_output)
                self._root_output.clear()
            if err:
                old_output.extend(self._output)
            self._output = old_output
            self._dirty = old_dirty

    def _print_func(self, *args):
        out = self._root_output if self._depth > 0 else self._output
        first = True
        for val in args:
            if first:
                first = False
            else:
                out.append(' ')
            out.append(try_to_str(val))
        out.append('\n')

    def _exit_func(self):
        raise Exit()

    # Nested block: {{ ... }}
    # With scope: {@name {...}} (name optional, defaults to '')
    # This wraps `_scan_block()` for a `code_block` for runtime caching.
    # To reduce stack usage, the caller is responsible for calling `_sub_doc()`
    # or `_run_block()`, based on whether a sub-doc token is returned.
    def _scan_code_block(self, tree: Tree) -> tuple[Tree, str | None]:
        inner = tree.children[1]
        if isinstance(inner, Tree):
            assert inner.data == 'block_inner'

            if (scope := tree.children[0]) is not None:
                scope = narrow(scope, Tree)
                assert scope.data == 'scope'
                if is_tracing:
                    trace('_code_block: scope: %s', self.debug_node(scope))

                if inner.children[0] is not None:
                    assert inner.children[0].data == 'scope'
                    self._error('double scope')

                # Embed our own scope into the 'block_inner'.
                # This need to be idempotent, since the tree is cached and may be reused.
                inner.children[0] = scope
                tree.children[0] = None

            sub_doc_token = self._scan_block(inner)

            # Cache the scan result.
            tree.children[1] = r = (inner, sub_doc_token)
            return r

        if is_tracing:
            trace('_code_block: cached: %s', inner[1])
        return inner

    # Create a boxed sub-doc value from `block_inner`, and store it in the current
    # scope if uncaptured.
    def _sub_doc(
        self, tree: Tree, sub_doc_token: str, *, captured: bool = False
    ) -> SubDoc:
        if sub_doc_token:
            if captured:
                params = tuple(r for s in sub_doc_token.split(',') if (r := s.strip()))
                sub_doc = SubDoc(tree, params)
                trace('Captured sub-doc: %s %s', sub_doc_token, sub_doc)
                return sub_doc

            key = sub_doc_token
            self._check_iden(key)
            sub_doc = SubDoc(tree, None)
            trace('Uncaptured sub-doc: %s %s', key, sub_doc)
            # Side-effect: the box is saved in the current scope.
            self._scope.set(key, sub_doc, self._ctx.__setitem__)
            return sub_doc

        if not captured:
            self._error('unnamed sub-doc must be captured')
        return SubDoc(tree, None)

    # Returns name of the sub-doc.
    def _scan_block(self, tree: Tree, *, root: bool = False) -> str | None:
        assert tree.data == 'block_inner', tree
        # Find any doc_def before entering statements to enable sub-docs.
        for i, ch in enumerate(tree.children):
            # Skip the modifier at 0 and the already processed doc_defs.
            if not (
                i and ch is not None and (ch := narrow(ch, Tree)).data == 'doc_def'
            ):
                continue

            op = narrow(ch.children[1], Token).value

            # Sub-document definition:
            # { name ↦ ... } (uncaptured, as statement)
            # { arg1, arg2, ... ↦ ... } (captured as expression)
            if op != ':':
                # Current 'block_inner' as a sub-document.
                # Scope from 'code_block' is already embedded.
                key = ch.children[0]
                trace('doc_def: sub_doc: %s %s', key, op)

                # The caller should skip evaluating the entire block (`_run_block`)
                # in this branch.
                return '' if key is None else narrow(key, Token).value.strip()

            # Doc name definition: {name:}
            key = ch.children[0]
            if key is None:
                self._error('empty doc definition')
                return

            key = self._iden(key)

            # Common doc definition is only allowed at the root level.
            # But we don't complain otherwise, since recursions also trigger this.
            if root:
                trace('doc_def: %s', key)
                if self.doc_name is not None:
                    self._error(f'doc name redefined: {self.doc_name} -> {key}')
                else:
                    self.doc_name = key
            else:
                trace('doc_def: ignored: %s', key)

            return

    def _set_tco(self, tree: Tree, env: dict[str, Any] | None = None) -> TCO:
        assert self._tco is None
        self._tco = (tree, self._scope.current(), env)
        return Tco

    def _run_block(self, tree: Tree, *, env: dict[str, Any] | None = None):
        '''
        Entrypoint of block execution.

        A block can provide semantics of a (dynamic) scope, a doc (name) definition, a
        sub-doc definition, any combination of them, or just a plain block of statements.

        A sub-doc may look like a "function", but is actually a `code_block` with
        deferred execution, and is completely *orthogonal* to the variable scoping
        mechanism, i.e. a sub-doc may or may not introduce a new scope.

        Maybe we can think of it as a "first-class macro" with runtime evaluation.
        '''

        # Trampoline for TCO, a.k.a. blockchain!
        ref_scope = None
        while True:
            if (scope_name := self._block_inner_resolve_scope(tree)) is not None:
                self._scope.push(scope_name)

            # Env must be set in the inner scope, after all pushing.
            self._block_inner_set_env(env)

            try:
                r = self._block_inner_exec(tree)
            finally:
                if scope_name is not None:
                    self._scope.pop()
                if ref_scope is not None:
                    # Pushed below.
                    last = self._scope.pop()
                    assert last == ref_scope

            if r is not Tco:
                return

            assert self._tco is not None
            # Replace our arguments.
            tree, ref_scope, env = self._tco
            self._tco = None

            if is_tracing:
                trace(
                    'TCO @ %s -> %s: %s',
                    self._scope.current(),
                    ref_scope,
                    self.debug_node(tree),
                )

            if ref_scope == self._scope.current():
                ref_scope = None
            else:
                self._scope.push_raw(ref_scope)

    # Find the scope modifier.
    def _block_inner_resolve_scope(self, tree: Tree) -> str | None:
        if (scope := tree.children[0]) is not None:
            scope = narrow(scope, Tree)
            assert scope.data == 'scope'
            if (scope := scope.children[0]) is None:
                return ''
            return self._lvalue(scope)

    # Enforce context in the inner scope.
    def _block_inner_set_env(self, env: dict[str, Any] | None = None):
        trace('_block_inner_set_env: %s', env)
        if env:
            for key, val in env.items():
                self._scope.set(key, try_to_value(val), self._ctx.__setitem__)

    # Execute!
    def _block_inner_exec(self, tree: Tree) -> MaybeTCO:
        stmt = narrow(tree.children[-1], Tree)
        if self._is_empty_stmt(stmt):
            stmt = None

        refs = []
        for ch in tree.children[1:-1]:
            ch = narrow(ch, Tree)
            match ch.data:
                # Db doc expand: {:doc} (same as {*doc} in unary expressions)
                case 'doc_ref':
                    key = self._iden(ch.children[-1])
                    refs.append(key)
                # Db doc get: handled before in `_scan_block()`.
                case 'doc_def':
                    continue
                case _:
                    # `block_inner` can contain at most one statement.
                    raise ValueError(f'Bad block_inner child: {tree.pretty()}')

        if stmt is None:
            if refs:
                last = refs.pop()
                for key in refs:
                    self._put_val(self._doc_ref(key))
                val = self._doc_ref(last, allow_tco=True)
                return self._put_val(val)
        else:
            for key in refs:
                self._put_val(self._doc_ref(key))
            return self.visit(stmt, allow_tco=True)

    def stmt_list(
        self, tree: Tree, *, direct_branch: bool = False, allow_tco: bool = False
    ) -> MaybeTCO:
        if not tree.children:
            return

        if allow_tco:
            for stmt in tree.children[:-1]:
                self.visit(narrow(stmt, Tree), direct_branch=direct_branch)
            last = narrow(tree.children[-1], Tree)
            return self.visit(last, direct_branch=direct_branch, allow_tco=True)
        else:
            for stmt in tree.children:
                self.visit(narrow(stmt, Tree), direct_branch=direct_branch)

    @staticmethod
    def _is_empty_stmt(tree: Tree) -> bool:
        return tree.data == 'stmt_list' and not tree.children

    # Conditional branch: {<cond> ? true-directive : false-directive} (false part optional)
    def branch(self, tree: Tree, *, allow_tco: bool = False) -> MaybeTCO:
        cond = narrow(tree.children[0], Tree)

        # Allow undefined vars as falsy in conditions.
        # This only applies to the direct 'unary_chain', AOE and 'naked_lit' as
        # the condition expression.
        val = self._expr(cond, permissive=True, allow_undef=True)
        test = val and val != '0'

        chs = tree.children
        left = chs[1]
        match len(chs):
            case 4:
                right = chs[2]
                final_marker = chs[3]
            case 3:
                right = chs[2]
                if isinstance(right, Token):
                    final_marker = right
                    right = None
                else:
                    final_marker = None
            case 2:
                right = final_marker = None
            case _:
                raise ValueError(f'Bad branch: {tree.pretty()}')

        # The associativity of `branch` is too greedy and consumes all statements
        # in a recursive `stmt_list`, like `a ? b ; c; d` being parsed as
        # `a ? {b ; c ; d}` by Lark.
        # We keep only the first stmt here and execute the rest unconditionally
        # to work around and make it more intuitive, behave like a "real" ternary
        # operator that accepts single expressions, i.e. `a ? b : c; d;` means
        # `(a ? b : c); d;`.
        # You can still explicitly specify the captured parts with braces or a
        # final marker, like in `a ? b : {c ; d}` or `a ? b : c ; d !`.
        rest = None
        if final_marker is None:
            split_left = right is None
            to_split = left if split_left else right
            if to_split is not None:
                to_split = narrow(to_split, Tree)
                if to_split.data == 'stmt_list' and (chs := to_split.children):
                    if len(chs) > 1:
                        rest = narrow(chs[1], Tree)
                        if self._is_empty_stmt(rest):
                            rest = None
                    if split_left:
                        left = chs[0]
                    else:
                        right = chs[0]

        if is_tracing:
            trace(
                'branch: test=%s val=%r tco=%s\nleft=%s\nright=%s\nrest=%s',
                test,
                val,
                allow_tco,
                self.debug_node(left),
                self.debug_node(right),
                self.debug_node(rest),
            )

        to = left if test else right
        if rest is None:
            if to is None:
                return
            return self.visit(narrow(to, Tree), direct_branch=True, allow_tco=allow_tco)

        if to is not None:
            self.visit(narrow(to, Tree), direct_branch=True)
        return self.visit(rest, allow_tco=allow_tco)

    def _trace(
        self, tree: Tree, *, direct_branch: bool = False, allow_tco: bool = False
    ):
        if is_tracing:
            st = [str(self._depth), str(self._gas)]
            if direct_branch:
                st.append('branch')
            if allow_tco:
                st.append('tco')
            trace('[%s] %s', ' '.join(st), self.debug_node(tree))

    @overload
    def expr(
        self,
        tree: Tree,
        *,
        direct_branch: bool = False,
        allow_tco: Literal[False] = False,
    ) -> None: ...

    @overload
    def expr(
        self, tree: Tree, *, direct_branch: bool = False, allow_tco: Literal[True]
    ) -> MaybeTCO: ...

    def expr(
        self, tree: Tree, *, direct_branch: bool = False, allow_tco: bool = False
    ) -> MaybeTCO:
        # Expression statement: {<expr>}
        # The value is directly appended to output.
        if isinstance(inner := tree.children[0], Tree):
            match inner.data:
                case 'assign_or_equal':
                    # Hijack the AOE "expression". Different semantics as statements.
                    return self._assign(inner)
                case 'unary_chain':
                    # Optimize: flatten the unary chain directly to output, without capturing.
                    tree = inner
                    self._trace(tree, allow_tco=allow_tco)
                    assert tree.children, tree
                    for ch in tree.children[:-1]:
                        val = self._unary(narrow(ch, Tree))
                        trace('unary_chain: %r', val)
                        self._put_val(val)

                    last = narrow(tree.children[-1], Tree)
                    val = self._unary(last, allow_tco=allow_tco)
                    return self._put_val(val)
                case 'code_block':
                    # Optimize: skip `self._push()` for direct output.
                    inner, sub_doc_token = self._scan_code_block(inner)
                    if sub_doc_token is None:
                        if allow_tco:
                            return self._set_tco(inner)
                        return self._run_block(inner)
                    return self._put_val(self._sub_doc(inner, sub_doc_token))
                case 'dq_lit':
                    # Optimize: chance to TCO.
                    val = self._dq_lit(inner, allow_tco=allow_tco)
                    return self._put_val(val)

        val = self._expr(tree, permissive=not direct_branch)
        self._put_val(val)

    def _evaluate(self, tree: Tree | Token | None, *, permissive: bool = False) -> str:
        if tree is None:
            return ''
        s = self._expr(narrow(tree, Tree), as_str=True, permissive=permissive)
        assert isinstance(s, str), s
        return s

    @overload
    def _expr(
        self,
        tree: Tree,
        *,
        as_str: Literal[True],
        permissive: bool = False,
        allow_undef: bool = False,
    ) -> str: ...

    @overload
    def _expr(
        self,
        tree: Tree,
        *,
        as_str: Literal[False] = False,
        permissive: bool = False,
        allow_undef: bool = False,
    ) -> Value: ...

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
            feats = (
                permissive and 'permissive',
                as_str and 'as_str',
                allow_undef and 'allow_undef',
            )
            if feats:
                feats = ' [' + ', '.join(f for f in feats if f) + ']'
            else:
                feats = ''
            trace(
                '[%d] _expr: %s%s',
                self._gas,
                self.debug_node(tree.children[0]),
                feats,
            )
        self._consume_gas()

        assert len(tree.children) == 1
        ch = tree.children[0]
        while True:
            if isinstance(ch, Tree):
                if ch.data == 'expr':
                    # Flatten nested expressions: {( ( ... ) )}
                    permissive = True
                    assert len(ch.children) == 1
                    ch = ch.children[0]
                    continue
                break

            # Ambiguous naked literal / var: {some_name}
            # Treated as var only if `permissive` is True and contains no whitespace.
            return self._naked_lit(
                ch,
                as_str=as_str,
                permissive=permissive,
                allow_undef=allow_undef,
            )

        match ch.data:
            case 'unary_chain':
                tree = ch
                assert tree.children, tree
                out = []
                for ch in tree.children:
                    val = self._unary(narrow(ch, Tree), allow_undef=allow_undef)
                    trace('unary_chain (captured): %r', val)
                    if val != '':
                        out.append(val)
                return self._gather_output(out, as_str=as_str)

            case 'assign_or_equal':
                return self._equal(ch, allow_undef=allow_undef)

            # Equality check: {key == val} (by string representation!)
            # `a==b` means `$a == 'b'`, and `a==$b` means `$a == $b`.
            case 'compare':
                left = self._evaluate(ch.children[0], permissive=True)
                right = self._evaluate(ch.children[1])
                return '1' if left == right else '0'

            # Nested block: {{ ... }}
            case 'code_block':
                inner, sub_doc_token = self._scan_code_block(ch)
                if sub_doc_token is not None:
                    return self._sub_doc(inner, sub_doc_token, captured=True)

                with self._push():
                    self._run_block(inner)
                    return self._gather_output(self._output, as_str=as_str)

            case 'subscript':
                key = self._subscript(ch)
                return self._scope.get(key, as_str=as_str, allow_undef=allow_undef)

            # Python expression: {"1 + 1"}
            # This should be non-mutating, i.e. side-effect free.
            case 'dq_lit':
                return self._dq_lit(ch, as_str=as_str)

            # Literal: {'single quoted'}
            case 'sq_lit':
                r = narrow(ch.children[0], Token).value[1:-1]
                r = r.encode('utf-8').decode('unicode_escape')
                return r

            case _:
                raise ValueError(f'Bad expr: {tree.pretty()}')

    def _naked_lit(
        self,
        token: Token,
        *,
        as_str: bool = False,
        permissive: bool = False,
        allow_undef: bool = False,
    ) -> Value:
        key = token.value.strip()
        if permissive and self._check_iden(key):
            return self._scope.get(key, as_str=as_str, allow_undef=allow_undef)
        return key

    @overload
    def _dq_lit(
        self, tree: Tree, *, as_str: Literal[True], allow_tco: Literal[False] = False
    ) -> str: ...

    @overload
    def _dq_lit(
        self, tree: Tree, *, as_str: Literal[True], allow_tco: Literal[True]
    ) -> str | TCO: ...

    @overload
    def _dq_lit(
        self, tree: Tree, *, as_str: bool = False, allow_tco: Literal[False] = False
    ) -> Value: ...

    @overload
    def _dq_lit(
        self, tree: Tree, *, as_str: bool = False, allow_tco: Literal[True]
    ) -> Value | TCO: ...

    def _dq_lit(
        self, tree: Tree, *, as_str: bool = False, allow_tco: bool = False
    ) -> Value | TCO:
        value: str = narrow(tree.children[0], Token).value
        expr = value[1:-1].strip().replace('\\"', '"').replace('\\\\', '\\')
        try:
            val = self._scope.eval(expr, allow_tco=allow_tco)
        except Abort as e:
            # This includes `Exit`.
            trace(f'_dq_lit: %r', e)
            raise
        except Exception as e:
            self._error(f'evaluate: {expr!r}: {type(e).__name__}: {e}')
            if is_tracing:
                s = traceback.format_exc()
                trace('%s', s)
            return ''

        if isinstance(val, tuple):
            assert allow_tco
            sub_doc = val[0]
            if not isinstance(sub_doc, SubDoc):
                self._error('bad tail-call')
                return ''
            return self._set_tco(sub_doc.tree, self._create_block_env(*val))

        # Not necessarily str!
        return to_str(val) if as_str else val

    def _check_iden(self, key: str) -> bool:
        if not key:
            self._error('empty identifier')
            return False

        if any(c.isspace() for c in key):
            self._error('bad identifier: ' + repr(key))
            return False

        return True

    # Lark didn't distinguish VAR from LITERAL, so we have to strip manually.
    def _iden(self, tree: Tree | Token) -> str:
        key = narrow(tree, Token).value.strip()
        self._check_iden(key)
        return key

    def _expr_iden(self, tree: Tree) -> str:
        # Paren expression as dynamic identifier:
        # {k=a; $(k)} means {$a}.
        assert tree.data == 'expr', tree
        key = self._expr(tree, permissive=True, as_str=True).strip()
        self._check_iden(key)
        return key

    def _subscript(self, tree: Tree) -> str:
        assert tree.data == 'subscript', tree
        key = self._lvalue(tree.children[0])
        val = self._expr(tree.children[1], permissive=True, as_str=True)
        assert isinstance(val, str), val
        key = key.strip() + '.' + val.strip()
        self._check_iden(key)
        return key

    def _lvalue(self, tree: Tree | Token) -> str:
        if isinstance(tree, Tree):
            if tree.data == 'expr':
                return self._expr_iden(tree)
            return self._subscript(tree)
        return self._iden(tree)

    def _get_by_raw_key(
        self, key: str, *, as_str: bool = False, allow_undef: bool = False
    ) -> Value:
        val = self._ctx.get(key)
        trace('_get_by_raw_key: %s (%s) = %r', key, len(key), val)
        if val is None:
            if not allow_undef:
                self._error('undefined: ' + key)
            return ''
        return to_str(val) if as_str else val

    @overload
    def _unary(
        self,
        tree: Tree,
        *,
        as_str: bool = False,
        allow_undef: bool = False,
        allow_tco: Literal[False] = False,
    ) -> Value: ...

    @overload
    def _unary(
        self,
        tree: Tree,
        *,
        as_str: bool = False,
        allow_undef: bool = False,
        allow_tco: Literal[True],
    ) -> Value | TCO: ...

    def _unary(
        self,
        tree: Tree,
        *,
        as_str: bool = False,
        allow_undef: bool = False,
        allow_tco: bool = False,
    ) -> Value | TCO:
        self._trace(tree, allow_tco=allow_tco)
        op = narrow(tree.children[0], Token).value
        name = tree.children[1]

        # `allow_undef` does not affect nested expressions in dynamic identifiers.
        key = self._lvalue(name)
        trace('_unary: op=%r key=%r', op, key)

        match op:
            # Flag set: {+name} or {-name}
            # This means {name:=1} or {name:=0}.
            case '+':
                self._ctx.setdefault_override(key, '1')
                return ''

            case '-':
                self._ctx.setdefault_override(key, '0')
                return ''

            # Variable read: {$name}
            # Resolve from the current or the nearest outer scopes with the name defined.
            case '$':
                return self._scope.get(key, as_str=as_str, allow_undef=allow_undef)

            # Db doc expand: {*doc} (same as {:doc})
            case '*':
                return self._doc_ref(key, allow_tco=allow_tco)

            # Variable read in the last scope: {::name}
            case '::':
                last_scope = self._scope.last()
                if last_scope is None:
                    self._error('no last scope for ::' + key)
                    return ''
                return self._get_by_raw_key(
                    last_scope + key, as_str=as_str, allow_undef=True
                )

            case _:
                raise ValueError(f'Bad deref op: {op}')

    def _get_doc(self, key: str) -> str | None:
        # Allow self-reference in when current doc is not saved yet.
        if self.doc_name == key:
            return self._doc_text
        doc = self._doc_src(key)
        if doc is None:
            self._error('no doc: ' + key)
        return doc

    # Used in dq_lit evaluation as `__file__` function.
    def _get_doc_func(self, key=None) -> str | None:
        if isinstance(key, str):
            return self._get_doc(key)
        return self._doc_text

    def _doc_ref(self, key: str, *, allow_tco: bool = False) -> Value | TCO:
        val = self._scope.get(key, allow_undef=True)
        if is_tracing:
            s = ' (tco)' if allow_tco else ''
            trace('_doc_ref%s: key=%s val=%r', s, key, val)

        if isinstance(val, SubDoc):
            tree = val.tree
            if is_tracing:
                trace(
                    '_doc_ref: Rendering sub-doc: %s %s',
                    key,
                    self.debug_node(tree),
                )
            if allow_tco:
                return self._set_tco(tree)

            with self._push():
                self._run_block(tree)
                return self._gather_output(self._output, trim=True)

        doc = self._get_doc(key)
        if doc is None:
            return ''

        trace('_doc_ref: Rendering doc: %s', key)
        with self._push():
            self._render(doc)
            return self._gather_output(self._output, trim=True)

    def _create_block_env(
        self, sub_doc: SubDoc, args: tuple, kwargs: dict[str, Any]
    ) -> dict[str, Any]:
        params = sub_doc.params
        if params:
            for param, arg in zip(params, args):
                kwargs.setdefault(param, arg)
            for i in range(len(args), len(params)):
                kwargs.setdefault(params[i], '')
            if len(args) > len(params):
                self._error(
                    f'box takes at most {len(params)} args {params}, got {len(args)}'
                )
        else:
            for i, arg in enumerate(args):
                kwargs.setdefault(str(i), arg)
        return kwargs

    @override
    def _call_box(self, sub_doc: Box, *args, **kwargs) -> Value | None:
        if not isinstance(sub_doc, SubDoc):
            self._error('bad box call')
            return

        tree = sub_doc.tree
        assert tree.data == 'block_inner', tree
        if is_tracing:
            trace(
                '_call_box: %s args=%s kwargs=%s', self.debug_node(tree), args, kwargs
            )

        # Recursive boxed calls are still limited by `MAX_DEPTH`.
        env = self._create_block_env(sub_doc, args, kwargs)
        with self._push():
            self._run_block(tree, env=env)
            return self._gather_output(self._output, default=None)

    # Due to LALR limitations, AOE chains have very different semantics:
    # - In expression statements, they set context vars and return ''.
    # - Otherwise, they check whether all the assigned vars equal the final value, and return '1' or '0'.
    # But in both cases, the ASSIGN_OP besides the first one must be '='.
    def _resolve_aoe_chain(self, tree: Tree) -> tuple[Sequence[str], str, Tree | None]:
        keys = [self._lvalue(tree.children[0])]
        op = narrow(tree.children[1], Token).value
        val = None

        while rest := tree.children[2]:
            rest = narrow(rest, Tree)
            assert rest.data == 'expr'
            tree = rest.children[0]
            if not (isinstance(tree, Tree) and tree.data == 'assign_or_equal'):
                val = rest
                break
            keys.append(self._lvalue(tree.children[0]))
            if narrow(tree.children[1], Token).value != '=':
                self._error('bad assign op in chain: ' + op)

        return keys, op, val

    # Do real assignments here.
    def _assign(self, tree: Tree):
        keys, op, expr = self._resolve_aoe_chain(tree)
        if is_tracing:
            trace('Assign: keys=%s op=%r expr=%s', keys, op, self.debug_node(expr))

        match op:
            case '=':
                # Context set: {key=value}
                # {key1=key2=...=value} sets all keys to value
                f = self._ctx.__setitem__

            case ':=':
                # Context override: {key:=value}
                # a. Can never be overridden (except by markup buttons)
                # b. Does not interrupt the natural order of flags detected in markup
                # {key1:=key2=...=value} affects all keys
                f = self._ctx.setdefault_override

            case '?=':
                # Context set if undefined or empty: {name?=value}
                # {name1?=name2=...=value} assigns the value to the undefined/empty
                # names only. Only '' (empty string) is considered empty.
                #
                # Mote that the evaluation is short-circuit (conditional) here,
                # like a `OnceCell` initialization.
                #
                # Also, be careful that the original value can be resolved from outer
                # scopes, in which case a key is still created in the current scope
                # to hold the outer value, even if the provided expression is not
                # evaluated!
                #
                # This is to ensure consistent behavior with other assignment ops
                # to ensure the var exists in the current scope after the assignment.
                val = '' if expr is None else None
                for key in keys:
                    raw_key, old = self._scope.resolve_raw(key)
                    key = self._scope.current_key(key)
                    if old is None or old == '':
                        if val is None:
                            val = self._expr(expr)  # type: ignore[assignment]
                        new = val
                    else:
                        new = old

                    trace(
                        'Assigning: "?=": key: %s = %r (%s was %r)',
                        key,
                        new,
                        raw_key,
                        old,
                    )

                    # Write back even if unchanged. This has two effects:
                    # 1. Ensures the var in the inner scope with the outer value.
                    # 2. Notifies the underlying `OverriddenDict` to touch the key.
                    #    See `OverriddenDict._setitem__`.
                    self._ctx[key] = new

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
    def _equal(self, tree: Tree, allow_undef: bool = False) -> str:
        keys, op, expr = self._resolve_aoe_chain(tree)
        if is_tracing:
            trace('Equal: keys=%s op=%r expr=%s', keys, op, self.debug_node(expr))
        if op != '=':
            self._error('bad assign op in equality test: ' + op)

        final_val = self._expr(expr, as_str=True) if expr else ''
        for key in keys:
            val = self._scope.get(key, as_str=True, allow_undef=allow_undef)
            if val != final_val:
                return '0'

        return '1'

    # Context swap: {key1^key2}
    # To be "atomic", This allows modifying vars in outer scopes.
    def swap(self, tree: Tree):
        key1 = self._lvalue(tree.children[0])
        key2 = self._lvalue(tree.children[1])
        key1, val1 = self._scope.resolve_raw(key1)
        key2, val2 = self._scope.resolve_raw(key2)
        self._scope.set_or_del_raw(key1, val2)
        self._scope.set_or_del_raw(key2, val1)

    # Context inplace replacement: {key|pat1/rep1|pat2/rep2|...}
    def repl(self, tree: Tree):
        key = self._iden(tree.children[0])
        key, val = self._scope.resolve_raw(key, '', as_str=True)
        assert isinstance(val, str), val
        for pair in tree.children[1:]:
            pat, sub = pair.children
            pat = self._evaluate(pat)
            sub = self._evaluate(sub)

            # Write back each time.
            self._ctx[key] = val = val.replace(pat, sub)
