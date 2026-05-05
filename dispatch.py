import inspect
from itertools import islice
from typing import Any, Callable, Iterable, Coroutine, Type

from telegram import CallbackQuery, Message, Update, Bot
from telegram.ext import ContextTypes

from util import USER_ID, log, get_msg, get_msg_arg

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

type Handler = Callable[..., Coroutine[Any, Any, Any]]


def _unwrap[T](x: T | None) -> T:
    if x is None:
        raise ValueError('Expected value, got None')
    return x


class Route:
    __slots__ = ('_func', '_public', '_params', '_va')

    def __init__(self, func: Handler, public: bool):
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

        self._func = func
        self._public = public
        self._params = tuple(params)
        self._va = va_ty

    def call(
        self, update: Update, ctx: ContextTypes.DEFAULT_TYPE, argv: Iterable[str]
    ) -> Coroutine[Any, Any, Any]:
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
                args.append(get_msg_arg(update)[1])
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
            args.extend(self._va(x) for x in it)
        elif (v := next(it, None)) is not None:
            raise ValueError(f'Too many arguments: {args}, {v}')

        log.debug('Injected to %s: %r', self._func.__name__, args)
        return self._func(*args)


_cb_handlers: dict[str, Route] = {}
_cb_filters: list[tuple[Callable[[CallbackData], bool], Route]] = []


def callback_query(
    key: str | None = None,
    *,
    filter: Callable[[CallbackData], bool] | None = None,
    public: bool = False,
):
    def decorator(func: Handler) -> Handler:
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
    if not (query := update.callback_query):
        raise ValueError('No callback_query')

    if not (data := query.data):
        raise ValueError('No data in cq')

    if data == 'noop':
        return await query.answer()

    for filter_func, route in _cb_filters:
        if filter_func(data):
            return await route.call(update, ctx, ())

    args = data.split('_')
    if route := _cb_handlers.get(args[0]):
        return await route.call(update, ctx, islice(args, 1, None))

    raise ValueError(f'Bad callback query: {data}')
