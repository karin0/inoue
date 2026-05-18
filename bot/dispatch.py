import inspect
from itertools import islice
from typing import Callable, Iterable, Awaitable, Protocol, cast, overload

from telegram import CallbackQuery, Message, Bot

from . import env
from .env import log
from .app import bot
from .responder import Responder
from .inline_responder import InlineResponder

type MessageArg = str
type CallbackData = str
type RequireDefer = Callable[[], Awaitable[None]] | None

type CallbackParam = Message | MessageArg | CallbackQuery | CallbackData | Bot | Responder | RequireDefer | str | int

UNIQUE_PARAM_TYPES = frozenset(
    (
        Message,
        MessageArg,
        CallbackQuery,
        CallbackData,
        Bot,
        Responder,
        RequireDefer,
    )
)
FREE_PARAM_TYPES = (str, int)

type DefaultType = tuple[str, object]
type ParamType = type[CallbackParam] | DefaultType


def _unwrap[T](x: T | None) -> T:
    if x is None:
        raise ValueError('Expected value, got None')
    return x


class Handler[**P, T](Protocol):
    __slots__ = ()

    @property
    def __name__(self) -> str: ...

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> Awaitable[T]: ...


class Decorated[**P, T](Handler[P, T], Protocol):
    __slots__ = ()

    @property
    def route(self) -> Route[P, T]: ...


class Route[**P, T]:
    __slots__ = ('_func', 'public', '_params', '_va')

    def __init__(self, func: Handler[P, T], public: bool):
        self._func: Handler[..., T] = func
        self.public = public
        self._va: type[str | int] | None = None

        sig = inspect.signature(func)
        self._params = tuple(self._build(sig))

        log.debug(
            'Handler: %s%s: public=%s, params=%d, va=%s',
            func.__name__,
            sig,
            public,
            len(self._params),
            self._va,
        )

    def _build(self, sig: inspect.Signature) -> Iterable[ParamType]:
        seen = set()
        for name, param in sig.parameters.items():
            kind = param.kind
            ty = param.annotation

            if kind == param.VAR_POSITIONAL:
                if self._va is not None:
                    raise TypeError('Multiple *args')
                if ty not in (str, int):
                    raise TypeError(f'*args must be str or int, got {ty}')
                self._va = ty
                continue

            if kind == param.VAR_KEYWORD:
                continue

            if kind == param.KEYWORD_ONLY:
                raise TypeError('Keyword-only parameters are unsupported')

            assert kind in (param.POSITIONAL_ONLY, param.POSITIONAL_OR_KEYWORD)

            if ty in UNIQUE_PARAM_TYPES:
                if ty in seen:
                    raise TypeError(f'Duplicate parameter: {ty}')
                seen.add(ty)
            elif ty not in FREE_PARAM_TYPES:
                if param.default is param.empty:
                    raise TypeError(f'Bad parameter: {param}')
                ty = (name, param.default)

            yield ty

    def __call__(
        self, rs: Responder | None, argv: Iterable[str] = (), **kwargs
    ) -> Awaitable[T]:
        log.debug('Calling route: %s: %r', self, argv)
        update = env.driver.get_update(rs, public=self.public)

        it = iter(argv)
        args = []
        defer = None
        for ty in self._params:
            if ty is Message:
                args.append(_unwrap(rs).get_message())
            elif ty is MessageArg:
                args.append('' if rs is None else rs.get_arg())
            elif ty is CallbackQuery:
                args.append(_unwrap(update.callback_query))
            elif ty is CallbackData:
                args.append(_unwrap(_unwrap(update.callback_query).data))
            elif ty is Bot:
                args.append(bot)
            elif ty is Responder:
                args.append(_unwrap(rs))
            elif ty is str:
                args.append(next(it))
            elif ty is int:
                args.append(int(next(it)))
            elif ty is RequireDefer:
                if isinstance(rs, InlineResponder):
                    defer = (rs.wait_until, len(args))
                    args.append(rs.flush)
                else:
                    args.append(None)
            elif isinstance(ty, tuple):
                ty_: DefaultType = ty
                args.append(kwargs.get(*ty_))
            else:
                raise TypeError(f'Bad parameter: {ty}')

        if self._va is not None:
            args.extend(map(self._va, it))

        log.debug('Injected to %s: %r', self._func.__name__, args)
        coro = self._func(*args)
        if defer is not None:
            if (fut := defer[0](coro)) is not None:
                return fut
            args[defer[1]] = None
        return coro

    def __repr__(self) -> str:
        return f'<{"Public " if self.public else ""}Route: {self._func.__name__}>'

    @property
    def __name__(self) -> str:
        return repr(self)


def _wrap[**P, T](
    func: Handler[P, T], public: bool
) -> tuple[Route[P, T], Decorated[P, T]]:
    route = Route(func, public=public)
    setattr(func, 'route', route)
    return route, cast(Decorated[P, T], func)


type Decorator[**P, T] = Callable[[Handler[P, T]], Decorated[P, T]]

commands: dict[str, Route] = {}


@overload
def command[**P, T](
    func: Handler[P, T], /, *, public: bool = False
) -> Decorated[P, T]: ...


@overload
def command[**P, T](
    func: str | None = None, /, *, public: bool = False
) -> Decorator[P, T]: ...


def command[**P, T](
    func: Handler[P, T] | str | None = None, /, *, public: bool = False
) -> Decorated[P, T] | Decorator[P, T]:
    name_ = None

    def decorator(func: Handler[P, T]) -> Decorated[P, T]:
        name = name_

        if name is None:
            parts = func.__name__.split('_')
            if not (len(parts) == 2 and parts[0] == 'handle'):
                raise ValueError(
                    'command: name must be provided if function name does not match handle_*'
                )
            name = parts[1]

        if not name:
            raise ValueError('command: name cannot be empty')

        if name in commands:
            raise ValueError(f'command: {name} already exists')

        route, func = _wrap(func, public)
        commands[name] = route
        return func

    if func is None or isinstance(func, str):
        name_ = func
        return decorator

    return decorator(func)


_cb_handlers: dict[str, Route] = {}
_cb_filters: list[tuple[Callable[[CallbackData], bool], Route]] = []


def callback_query[**P, T](
    key: str | None = None,
    *,
    filter: Callable[[CallbackData], bool] | None = None,
    public: bool = False,
) -> Decorator[P, T]:
    def decorator(func: Handler[P, T]) -> Decorated[P, T]:
        if filter is not None:
            if key is not None:
                raise ValueError(
                    'callback_query: key and filter cannot be used together'
                )
            if not callable(filter):
                raise TypeError('callback_query: filter must be callable')
            route, func = _wrap(func, public)
            _cb_filters.append((filter, route))
        elif key is not None:
            if key in _cb_handlers:
                raise ValueError(f'callback_query: {key} already exists')
            route, func = _wrap(func, public)
            _cb_handlers[key] = route
        else:
            raise ValueError('callback_query: either key or filter must be provided')

        return func

    return decorator


def _dispatch_argv(
    rs: Responder | None, data: str, map: dict[str, Route]
) -> Awaitable | None:
    args = data.split('_')
    if (route := map.get(args[0])) is not None:
        return route(rs, islice(args, 1, None))


def dispatch_callback(rs: Responder | None, data: str) -> Awaitable:
    for filter_func, route in _cb_filters:
        if filter_func(data):
            return route(rs)

    if fut := _dispatch_argv(rs, data, _cb_handlers):
        return fut

    raise ValueError(f'Bad callback query: {data}')


_start_handlers: dict[str, Route] = {}


def start[**P, T](key: str, *, public: bool = False) -> Decorator[P, T]:
    def decorator(func: Handler[P, T]) -> Decorated[P, T]:
        if not key:
            raise ValueError('start: key cannot be empty')

        if key in _start_handlers:
            raise ValueError(f'start: {key} already exists')

        route, func = _wrap(func, public)
        _start_handlers[key] = route
        return func

    return decorator


def dispatch_start(rs: Responder, arg: MessageArg) -> Awaitable | None:
    return _dispatch_argv(rs, arg, _start_handlers) if arg else None
