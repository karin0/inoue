import inspect
from itertools import islice
from typing import Any, Callable, Iterable, Coroutine, Type, overload

from telegram import CallbackQuery, Message, Update, Bot
from telegram.ext import ContextTypes

from util import USER_ID, log, get_arg, get_msg

type MessageArg = str
type CallbackData = str

type CallbackParam = Update | ContextTypes.DEFAULT_TYPE | Message | MessageArg | CallbackQuery | CallbackData | Bot | str | int

CALLBACK_PARAM_TYPES = (
    Update,
    ContextTypes.DEFAULT_TYPE,
    Message,
    MessageArg,
    CallbackQuery,
    CallbackData,
    Bot,
    str,
    int,
)

type Handler[**P, R] = Callable[P, Coroutine[Any, Any, R]]

type PTBHandler[T] = Callable[
    [Update, ContextTypes.DEFAULT_TYPE], Coroutine[Any, Any, T]
]


def _unwrap[T](x: T | None) -> T:
    if x is None:
        raise ValueError('Expected value, got None')
    return x


class Route[**P, R]:
    __slots__ = ('_func', '_public', '_params', '_va')

    def __init__(self, func: Handler[P, R], public: bool):
        params: list[Type[CallbackParam]] = []
        va_ty: Type[str | int] | None = None
        sig = inspect.signature(func)
        for name, param in sig.parameters.items():
            kind = param.kind
            ty = param.annotation

            if kind == param.VAR_POSITIONAL:
                if va_ty is not None:
                    raise TypeError('Multiple *args')
                if ty not in (str, int):
                    raise TypeError(f'*args must be str or int, got {ty}')
                va_ty = ty
                continue

            if kind not in (param.POSITIONAL_ONLY, param.POSITIONAL_OR_KEYWORD):
                raise TypeError(f'Keyword argument: {name}')

            if ty not in CALLBACK_PARAM_TYPES:
                raise TypeError(f'Bad parameter: {ty}')

            params.append(ty)

        log.debug(
            'Handler: %s%s: public=%s, params=%d, va=%s',
            func.__name__,
            sig,
            public,
            len(params),
            va_ty,
        )

        self._func: Callable = func
        self._public = public
        self._params = tuple(params)
        self._va = va_ty

    def call(
        self, update: Update, ctx: ContextTypes.DEFAULT_TYPE, argv: Iterable[str] = ()
    ) -> Coroutine[Any, Any, R]:
        log.debug('Calling route: %s: %r', self._func.__name__, argv)
        if not (
            self._public
            or ((u := update.effective_user) is not None and u.id == USER_ID)
        ):
            raise PermissionError('Unauthorized')

        it = iter(argv)
        args = []
        for ty in self._params:
            if ty is Update:
                args.append(update)
            elif ty is ContextTypes.DEFAULT_TYPE:
                args.append(ctx)
            elif ty is Message:
                args.append(get_msg(update))
            elif ty is MessageArg:
                args.append(get_arg(get_msg(update)))
            elif ty is CallbackQuery:
                args.append(_unwrap(update.callback_query))
            elif ty is CallbackData:
                args.append(_unwrap(_unwrap(update.callback_query).data))
            elif ty is Bot:
                args.append(ctx.bot)
            elif ty is str:
                args.append(next(it))
            elif ty is int:
                args.append(int(next(it)))
            else:
                raise TypeError(f'Bad parameter: {ty}')

        if self._va is not None:
            args.extend(map(self._va, it))
        elif (v := next(it, None)) is not None:
            raise ValueError(f'Too many arguments: {args}, {v}')

        log.debug('Injected to %s: %r', self._func.__name__, args)
        return self._func(*args)


_cmd_handlers: dict[str, Route] = {}


@overload
def command[H: Handler](func: H, /, *, public: bool = False) -> H: ...


@overload
def command[H: Handler](
    func: str | None = None, /, *, public: bool = False
) -> Callable[[H], H]: ...


def command[H: Handler](
    func: H | str | None = None, /, *, public: bool = False
) -> H | Callable[[H], H]:
    name_ = None

    def decorator(func: H) -> H:
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

        if name in _cmd_handlers:
            raise ValueError(f'command: {name} already exists')

        _cmd_handlers[name] = Route(func, public)
        return func

    if func is None or isinstance(func, str):
        name_ = func
        return decorator

    return decorator(func)


def iter_commands() -> Iterable[tuple[str, tuple[PTBHandler, bool]]]:
    return (
        (name, (route.call, route._public)) for name, route in _cmd_handlers.items()
    )


def get_command_handler(name: str) -> PTBHandler | None:
    if route := _cmd_handlers.get(name):
        return route.call


_cb_handlers: dict[str, Route] = {}
_cb_filters: list[tuple[Callable[[CallbackData], bool], Route]] = []


def callback_query(
    key: str | None = None,
    *,
    filter: Callable[[CallbackData], bool] | None = None,
    public: bool = False,
):
    def decorator[H: Handler](func: H) -> H:
        route = Route(func, public)

        if filter is not None:
            if key is not None:
                raise ValueError(
                    'callback_query: key and filter cannot be used together'
                )
            if not callable(filter):
                raise TypeError('callback_query: filter must be callable')
            _cb_filters.append((filter, route))
        elif key is not None:
            if key in _cb_handlers:
                raise ValueError(f'callback_query: {key} already exists')
            _cb_handlers[key] = route
        else:
            raise ValueError('callback_query: either key or filter must be provided')

        return func

    return decorator


async def handle_callback_query(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if (query := update.callback_query) is None:
        raise ValueError('No callback_query')

    if not (data := query.data):
        raise ValueError('No data in cq')

    if data == 'noop':
        return await query.answer()

    for filter_func, route in _cb_filters:
        if filter_func(data):
            return await route.call(update, ctx, ())

    args = data.split('_')
    if (route := _cb_handlers.get(args[0])) is not None:
        return await route.call(update, ctx, islice(args, 1, None))

    raise ValueError(f'Bad callback query: {data}')


_start_handlers: dict[str, Route] = {}


def start(key: str, *, public: bool = False) -> Callable[[Handler], Handler]:
    def decorator[H: Handler](func: H) -> H:
        route = Route(func, public)

        if not key:
            raise ValueError('start: key cannot be empty')

        if key in _start_handlers:
            raise ValueError(f'start: {key} already exists')

        _start_handlers[key] = route
        return func

    return decorator


async def dispatch_start(
    update: Update, ctx: ContextTypes.DEFAULT_TYPE, arg: MessageArg
) -> bool:
    if arg:
        args = arg.split('_')
        if (route := _start_handlers.get(args[0])) is not None:
            await route.call(update, ctx, islice(args, 1, None))
            return True
    return False
