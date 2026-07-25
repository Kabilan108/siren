import asyncio
from collections.abc import Callable
from typing import ParamSpec, TypeVar

P = ParamSpec("P")
T = TypeVar("T")


async def run_in_worker_thread(
    function: Callable[P, T],
    *args: P.args,
    **kwargs: P.kwargs,
) -> T:
    return await asyncio.to_thread(function, *args, **kwargs)
