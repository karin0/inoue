# The render language

A small template language whose output is a piece of text. A source document is
ordinary text with code embedded in it, and rendering walks the document from
top to bottom, emitting text fragments as they are and replacing each code block
with whatever that block produces.

`Engine.render(text)` returns the result as a string and `Engine.render_value`
keeps the value's type. `python -m render_core.cli <file>` renders a file, with
positional arguments bound to the names `0`, `1` and so on. `Engine.errors`
collects every error the run produced; rendering continues past most of them.

`examples/redirect_hook.m` at the repository root is a worked document that uses
most of what follows.

## Text and code

`lex.py` splits a document into text fragments and code chunks before anything
is parsed. A chunk becomes code in one of two ways.

A balanced `{ ... }` block is code. The outer braces are stripped before the
inside is parsed, so a scope modifier written before the outer brace stays
text. Write `{@name { ... }}` or `{ @name; ... }` to attach a scope.

A run of lines is code when it ends with `;` and leaves no brace unclosed. This
is a naked block, and it may span several physical lines when a nested block
inside it does.

```
a = 1;
b = 2;
Sum is {a + b}.
```

A line holding only code and whitespace, whether naked or braced, drops the
whitespace and ends with an implicit line break in place of its own. That break
survives only when something non-blank has been written since the previous one,
so structural lines leave no blank gap.

Outside a block, a backslash escapes `{`, `}`, `;` and itself, and the backslash
is removed from the text.

## Lexical structure

Comments run only inside blocks. `//` and `#` reach the end of the line, and
`/* ... */` ends at its closing marker.

A single-quoted literal is a string, and backslash escapes inside it are
resolved the way Python resolves them, so `'\n'` is a line break.

A double-quoted literal is a Python expression, evaluated as described under
embedded Python below. `"1 + 1"` yields 2.

A backtick starts a raw literal that runs to the next `;`, `{` or `}`. A
backslash carries no meaning inside it, and a quoted section inside it protects
those three terminators. This is the way to write text that would otherwise need
escaping at every character.

Anything else that is neither punctuation nor an operator is a naked literal. It
may contain spaces, and its surrounding whitespace is stripped. A naked literal
that holds no whitespace and names a valid identifier is a variable in an
expression statement, a condition, a subscript or parentheses, and reading it
there reports an error when it is undefined. Inside the body of a branch it
stands for its own text, as does any naked literal holding whitespace, so
`{c = 1; c ? Hello : Bye}` renders `Hello` and `{Hello World}` renders
`Hello World`. A naked literal of digits without a leading zero becomes an
integer.

Identifiers exclude whitespace, and a bare decimal identifier is refused so that
`$0` has to be written with the sigil. Names carry `.` as an ordinary character,
which is how scope prefixes and subscripts share one flat namespace.

## Values

A value is a string, an integer, a float, a bool, a complex number, a bytes
object, or a `Box` that the host supplies. Converting to text turns a bool into
`1` or `0` and decodes bytes as UTF-8 with replacement.

A block that produced nothing yields the empty string, a block that produced one
value yields that value with its type intact, and a block that produced several
yields a `Fragment` holding them in order. Keeping the single-value type is what
lets `{ a = {b = 1; b}; a + 1 }` arrive at 2 rather than concatenating text.

## Variables and scopes

Every variable lives in one flat mapping. A scope is a prefix on that mapping,
so entering the scope `s` makes the name `a` read and write the key `s.a`.
`{@name; ... }` and `{@name { ... }}` enter a named scope, `@` alone enters an
anonymous one, and a read that misses walks outward through the enclosing scopes
until it finds the name or runs out.

A write lands in the current scope, except that `+name` and `-name` write the
name without any scope prefix, so a flag is one global name whatever scope sets
it. There is no other way to assign into an enclosing scope, though `^` and `|`
below reach the variable a lookup would have found.

| Read | |
| --- | --- |
| `$name` | the value, reporting an error when the name is undefined outside a condition |
| `$$name` | the value, or the empty string when undefined |
| `::name` | the value from the immediately enclosing scope |
| `name` | the value in a position that allows a variable, the literal text otherwise |
| `$(expr)` | the value of the name that `expr` spells out |
| `a[expr]` | the value of the name `a.` followed by what `expr` spells out |

| Write | |
| --- | --- |
| `name = value` | assign in the current scope |
| `name := value` | assign through the host, described below |
| `name ?= value` | assign when the name is undefined or holds the empty string |
| `+name` and `-name` | set the global name to 1 and 0 through the host, described below |
| `++name` and `--name` | add and subtract 1 to a name of the current scope, yielding the new value in an expression |
| `a ^ b` | exchange two values, each resolved the way a read resolves it |

`a = b = value` assigns to both names, and every operator after the first in
such a chain has to be `=`.

`:=`, `+name` and `-name` hand the write to `setitem_with(key, value, op)` on
the mapping the host passed to `Engine`, so the host decides what they do beyond
assigning. The base `Context` assigns, and `python -m render_core.cli` keeps
that. A host may instead freeze the key, so the first value written this way
outlasts every later assignment.

`?=` evaluates its expression only when at least one name needs it. It writes to
the current scope in every case, so a name that resolved to an enclosing scope
gets a local copy holding that outer value.

An `=` chain in a condition or in parentheses tests equality and yields `1` or
`0`, so `{a = 2; a = 1 ? y : n}` renders `n` and `{a = 2; x = (a = 1)}` leaves
`a` at 2. As a statement it assigns, also when it is the only statement of a
block, as in `{a = 1}` or `x = {a = 1}`. `?=` and `:=` assign in both positions
and yield the first name's value.

`name | pat / sub | pat2 / sub2` replaces text in place, applying each pair in
turn and writing the result back after each one. Writing `\` in place of `/`
makes the pair a regular expression, and a second `\` introduces flags drawn from
`a`, `i`, `m`, `s`, `u` and `x`. A regular expression that runs longer than 0.1
seconds is abandoned with an error.

## Conditions

A value is false when Python considers it false or when it is the string `0`,
and true otherwise, so the string `0.0` is true.

`cond ? then : else` chooses between two statement lists, and `:` with an empty
right side is allowed. The parser is greedy here, so a bare `cond ? a; b; c;`
would swallow everything; the interpreter keeps only the first statement of the
last branch under the condition and runs the rest unconditionally. Write braces
around a branch, or close the whole thing with `!`, to say otherwise:

```
{ c = 1; p='P'; q='Q'; c ? b : $p; $q }       renders bQ
{ c = 1; p='P'; q='Q'; c ? b : {$p; $q} }     renders b
{ c = 1; p='P'; q='Q'; c ? b : $p; $q ! }     renders b
```

A condition yields no value of its own, so a conditional used where a value is
expected is wrapped in a block. `{ c = 1; x = (c ? y : z) }` is a parse error,
and `{ c = 1; x = {c ? y : z} }` is the way to write it.

A condition may also span text rather than statements, which is how a document
selects between passages. A block whose source ends in `?` opens a clause, `{:}`
switches to the other side, and `{!}` closes it. A block ending in `?:` opens
the clause inverted.

```
{ show ? }
This paragraph appears when show is true.
{ : }
This one appears otherwise.
{ ! }
```

## Documents and sub-documents

`{name:}` at the top level of a document declares its name, which is how the
host stores and finds it. A document may declare one name.

`{:name}` and `{*name}` render another document and yield its output, and
`{**name}` renders it straight into the current output instead of gathering it.
`{:name}` ignores sub-documents of the same name, while `{*name}` prefers one.

A sub-document is a block held as a value.

| Form | |
| --- | --- |
| `{name ↦ ...}` | as a statement, binds the block to `name` in the current scope |
| `{a, b ↦ ...}` | as an expression, yields the block with `a` and `b` as parameters |
| `⇒` in place of `↦` | additionally captures the scope where the block was written |

Calling one binds arguments to its parameter names, fills any parameter left
over with the empty string, and reports an error when there are too many
arguments. A block written with no parameter list binds its arguments to the
names `0`, `1` and so on. Parameters land in whatever scope the block runs in,
so a block that wants them local opens with `@`, as in `{@; x ↦ ...}`. A block
written with `↦` runs in the caller's scope, and one written with `⇒` runs with
its defining scope restored, which is what makes it usable as a callback after
the surrounding render has returned.

Sub-documents and scopes are independent. A sub-document introduces a scope only
when its own block asks for one with `@`.

Expanding a sub-document with `*` in the last position of a block replaces the
current block rather than nesting inside it, so a document can recurse to any
depth without consuming stack. `examples/redirect_hook.m` uses this for its
polling loop.

## Embedded Python

A double-quoted literal is a Python expression, and so is a bare run of terms
joined by operators. Names resolve through the scope chain, and calls reach the
builtins below along with anything the host registered.

In a bare run of terms `/`, `|` and `^` are a parse error naming the character,
since the grammar uses them for replacement pairs, replacement separators and
swap, `//` starts a comment, and `in` and `is` are read as part of a naked
literal. `&&`, `||` and `~` stand in for `and`, `or` and `not`. A double-quoted
literal is plain Python and accepts all of these.
Comparing a string with a value of another type converts that value to text
first, in both forms, so `"1 == '1'"` is true and `"2 < '10'"` is false.

A `{ ... }` block may appear inside a Python expression and contributes its
rendered value. A Python keyword appearing as a naked literal is read as a
variable of that name, so a document may use `for` or `class` as an ordinary
name.

## Builtins

`print` writes to the root output, bypassing whatever block is being gathered.
`exit` stops the current document and leaves the enclosing ones running.
`output` returns and clears the current block's output so far. `eval` renders a
value as a document. `prefix` returns the current scope prefix. `__file__`
returns a document's source. `__name__` returns the name the root document
declares. `__this__` returns the name of the innermost document being rendered,
which is the name it was loaded by, or the declared name at the root. A
sub-document call changes neither, and `__this__` is empty in text rendered by
`eval`. `len` and simpleeval's own defaults are available, and the host adds the
rest through the `funcs` argument to `Engine`.

## Limits

Each evaluation step spends one unit of gas, capped at `MAX_GAS` in `engine.py`,
which is 2000 by default and raised by the CLI. Nesting is capped at `MAX_DEPTH`
of 20, counting every nested block, document expansion and sub-document call
except `{**name}` and an expansion in tail position. Both raise `Abort`, which
ends the render and leaves the reason in `Engine.errors`.

## Constraints behind irregular rules

`=` assigns as a statement and tests equality in a condition, and existing
documents depend on both meanings. `==` also tests equality and is the clearer
thing to write in a condition.

A conditional is not an expression because `branch` conflicts with
`assign_or_equal` under LALR. The same parser generator is why a document name
must precede any statement in its block, and why `BranchNormalizer` exists at
all: lark parses `a ? b; c; d` with the branch swallowing every following
statement, and that pass puts the extra statements back outside the branch.
