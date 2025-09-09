import pytest

from rag_api.app.api.streamquery import _encode_sse_event_from_chunk


def _decode_chunks(chunks):
    return b"".join(list(chunks)).decode("utf-8")


def test_encode_multiline_with_tabs():
    payload = "共8个\n\t突击蝎[来源1]\n\t凶暴复仇者[来源2]\n"
    out = _decode_chunks(_encode_sse_event_from_chunk(payload))

    # 期望：每一行成为一条 data: 行，事件以空行结束
    assert out == (
        "data: 共8个\n"
        "data: \t突击蝎[来源1]\n"
        "data: \t凶暴复仇者[来源2]\n"
        "\n"
    )


def test_encode_only_newline_preserved():
    payload = "\n"  # 单独换行
    out = _decode_chunks(_encode_sse_event_from_chunk(payload))
    # 应保留为一条空 data 行（内容为空字符串）+ 事件结束空行
    assert out == (
        "data: \n"
        "\n"
    )


