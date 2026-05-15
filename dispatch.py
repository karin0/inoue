import inspect
from itertools import islice
from typing import Any, Callable, Iterable, Coroutine, Awaitable, Type, overload

from telegram import CallbackQuery, Message, Update, Bot
from telegram.ext import ContextTypes

from util.log import log
from util.app import bot
from util.env import USER_ID
from util.ctx import get_arg, get_msg, get_context, get_responder
from util.responder import Responder

type MessageArg = str
type CallbackData = str

type CallbackParam = Update | ContextTypes.DEFAULT_TYPE | Message | MessageArg | CallbackQuery | CallbackData | Bot | Responder | str | int

CALLBACK_PARAM_TYPES = (
    Update,
    ContextTypes.DEFAULT_TYPE,
    Message,
    MessageArg,
    CallbackQuery,
    CallbackData,
    Bot,
    Responder,
    str,
    int,
)

type Handler[**P, R] = Callable[P, Awaitable[R]]

type UpdateHandler[R] = Callable[[Update], Awaitable[R]]


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

    def __repr__(self) -> str:
        return f'<{"Public " if self._public else ""}Route: {self._func.__name__}>'

    __str__ = __repr__

    @property
    def __name__(self) -> str:
        return repr(self)

    def __call__(
        self, update: Update, argv: Iterable[str] = ()
    ) -> Coroutine[Any, Any, R]:
        log.debug('Calling route: %s: %r', self, argv)
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
                args.append(get_context().ptb)
            elif ty is Message:
                args.append(get_msg(update))
            elif ty is MessageArg:
                args.append(get_arg(get_msg(update)))
            elif ty is CallbackQuery:
                args.append(_unwrap(update.callback_query))
            elif ty is CallbackData:
                args.append(_unwrap(_unwrap(update.callback_query).data))
            elif ty is Bot:
                args.append(bot)
            elif ty is Responder:
                args.append(get_responder(get_msg(update)))
            elif ty is str:
                args.append(next(it))
            elif ty is int:
                args.append(int(next(it)))
            else:
                raise TypeError(f'Bad parameter: {ty}')

        if self._va is not None:
            args.extend(map(self._va, it))

        log.debug('Injected to %s: %r', self._func.__name__, args)
        return self._func(*args)


_cmd_handlers: dict[str, Route] = {}

get_command_handler = _cmd_handlers.get


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


def iter_commands() -> Iterable[tuple[str, tuple[UpdateHandler, bool]]]:
    return ((name, (route, route._public)) for name, route in _cmd_handlers.items())


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


def _dispatch_argv(
    update: Update, data: str, map: dict[str, Route]
) -> Coroutine | None:
    args = data.split('_')
    if (route := map.get(args[0])) is not None:
        return route(update, islice(args, 1, None))


async def handle_callback_query(query: CallbackQuery, update: Update):
    if not (data := query.data):
        raise ValueError('No data in cq')

    if data == 'noop':
        return query.answer()

    try:
        for filter_func, route in _cb_filters:
            if filter_func(data):
                return await route(update)

        if fut := _dispatch_argv(update, data, _cb_handlers):
            return await fut

        raise ValueError(f'Bad callback query: {data}')
    except Exception as e:
        await query.answer('Error', show_alert=True)
        raise e


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


def dispatch_start(update: Update, arg: MessageArg) -> Coroutine | None:
    return _dispatch_argv(update, arg, _start_handlers) if arg else None
