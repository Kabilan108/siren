import asyncio
from collections.abc import Callable
from concurrent.futures import Future
from threading import Thread
from typing import ParamSpec, TypeVar

P = ParamSpec("P")
T = TypeVar("T")


async def run_in_worker_thread(
    function: Callable[P, T],
    *args: P.args,
    **kwargs: P.kwargs,
) -> T:
    future: Future[T] = Future()

    def run() -> None:
        if not future.set_running_or_notify_cancel():
            return
        try:
            future.set_result(function(*args, **kwargs))
        except BaseException as exc:
            future.set_exception(exc)

    Thread(target=run).start()
    try:
        return await asyncio.wrap_future(future)
    except asyncio.CancelledError:
        # GPU inference cannot be interrupted; wait for the thread so callers
        # holding the inference semaphore never overlap two inferences.
        if not future.cancel():
            try:
                await asyncio.shield(asyncio.wrap_future(future))
            except Exception:
                pass
        raise
