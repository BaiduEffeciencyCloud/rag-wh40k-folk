import json
import time
from typing import AsyncGenerator, AsyncIterable, Iterable, Union

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from ..models.query import QueryRequest
from ..services.rag_service import RAGService

router = APIRouter()


def _sse_format(data: str) -> str:
    return f"data: {data}\n\n"

def _encode_sse_event_from_chunk(payload: str):
    """将一个payload编码为一个SSE事件（多条data行+空行）。
    返回bytes迭代器，便于测试与复用。
    """
    lines = payload.splitlines(keepends=True)
    if not lines:
        return
    for ln in lines:
        ln_no_nl = ln[:-1] if ln.endswith("\n") else ln
        yield f"data: {ln_no_nl}\n".encode("utf-8")
    # 事件结束空行
    yield b"\n"


@router.post("/", response_class=StreamingResponse)
async def stream_query(request: QueryRequest, fastapi_request: Request) -> StreamingResponse:
    service = RAGService()
    result = await service.process_query(request.query, request.advance)

    results = result.get("results", result)
    stream_iter: Union[AsyncIterable, Iterable, None] = (
        results.get("final_answer_stream") if isinstance(results, dict) else None
    )

    async def event_gen() -> AsyncGenerator[bytes, None]:
        # 心跳：每15秒
        last = time.time()
        try:
            if stream_iter is not None:
                # 统一封装为异步生成器，兼容异步/同步
                import asyncio

                async def _iterate_any(obj: Union[AsyncIterable, Iterable]):
                    if hasattr(obj, "__aiter__"):
                        async for it in obj:  # type: ignore
                            yield it
                    else:
                        for it in obj:  # type: ignore
                            yield it
                            # 让出事件循环，避免一次性写入
                            await asyncio.sleep(0)

                async for chunk in _iterate_any(stream_iter):
                    # 客户端断开检测
                    if await fastapi_request.is_disconnected():
                        break
                    now = time.time()
                    if now - last > 15:
                        yield _sse_format("[keepalive]").encode("utf-8")
                        last = now
                    if chunk is None:
                        continue
                    # 统一字符串化（dict 走 JSON）
                    if isinstance(chunk, (dict, list)):
                        payload = json.dumps(chunk, ensure_ascii=False)
                    else:
                        payload = str(chunk)

                    # 将一个chunk按行拆分为同一事件的多条data行，避免单独"\n"被误解析为事件结束
                    for part in _encode_sse_event_from_chunk(payload):
                        yield part
                yield _sse_format("[DONE]").encode("utf-8")
            else:
                # 同步文本结果兜底
                text = results.get("final_answer") if isinstance(results, dict) else None
                if text:
                    yield _sse_format(str(text)).encode("utf-8")
                else:
                    yield _sse_format(json.dumps(results, ensure_ascii=False)).encode("utf-8")
                yield _sse_format("[DONE]").encode("utf-8")
        except Exception as e:
            yield _sse_format(json.dumps({"error": str(e)}, ensure_ascii=False)).encode("utf-8")
            yield _sse_format("[DONE]").encode("utf-8")

    headers = {
        "Cache-Control": "no-cache, no-transform",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",
    }
    return StreamingResponse(
        event_gen(),
        media_type="text/event-stream",
        headers=headers,
    )


