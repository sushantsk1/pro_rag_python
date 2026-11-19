"""NDJSON progress: the same experiment, with results delivered at stage boundaries."""

import asyncio
import json
import threading

from starlette.concurrency import run_in_threadpool

from .errors import RagError
from .playground import PlaygroundCancelled


async def stream_experiment(playground, request):
    messages = asyncio.Queue()
    loop = asyncio.get_running_loop()
    cancelled = threading.Event()

    def emit(message):
        if cancelled.is_set():
            return
        # Serialize here to snapshot mutable stage dictionaries before the next stage.
        encoded = json.dumps(message, allow_nan=False) if message is not None else None
        try:
            loop.call_soon_threadsafe(messages.put_nowait, encoded)
        except RuntimeError:
            cancelled.set()

    def work():
        try:
            result = playground.run(request, emit=emit, cancelled=cancelled)
            emit({"type": "run_completed", "run": result})
        except PlaygroundCancelled:
            pass
        except RagError as error:
            emit(
                {
                    "type": "run_error",
                    "error": {"code": error.code, "message": error.message, "status": error.status},
                }
            )
        except Exception:
            emit(
                {
                    "type": "run_error",
                    "error": {
                        "code": "INTERNAL_ERROR",
                        "message": "The experiment failed. Earlier stage outputs remain available.",
                        "status": 500,
                    },
                }
            )
        finally:
            emit(None)

    task = asyncio.create_task(run_in_threadpool(work))
    try:
        while True:
            message = await messages.get()
            if message is None:
                break
            yield message + "\n"
    finally:
        cancelled.set()
        # Do not cancel a running model call; its next stage checks the disconnect flag.
        # The task is retained until its worker finishes, including after disconnect.
        task.add_done_callback(lambda finished: finished.exception() if not finished.cancelled() else None)
