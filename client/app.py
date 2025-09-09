import time
import json
import re
import html
import requests
import streamlit as st
def _normalize_tabbed_list(text: str) -> str:
    """当返回是一整行但包含多个条目时，按条目自动断行并补 \t 前缀。
    - 条目边界基于 “... [来源N]” 或 “... [来源N,M]” 等结束标记。
    - 识别并单独提取首部的“共X个”。
    """
    if "\n" in text:
        return text

    # 提取开头的“共X个”统计信息
    header = ""
    m = re.match(r"^(共\d+个)", text)
    if m:
        header = m.group(1)
        text = text[m.end():].strip()

    # 以 [来源数字,数字] 作为条目边界切分（支持逗号/中文逗号）
    src_pat = r"\[来源[0-9,，]+\]"
    parts = re.split(f"({src_pat})", text)
    if not parts or len(parts) < 3:
        # 没有明显来源标记，直接返回（交由上层自动换行处理）
        return (header + "\n" if header else "") + text

    items = []
    buf = ""
    for seg in parts:
        buf += seg
        if re.fullmatch(src_pat, seg):
            items.append(buf.strip())
            buf = ""
    if buf.strip():
        items.append(buf.strip())

    if not items:
        return (header + "\n" if header else "") + text

    # 为每条目前加制表符并用换行连接
    body = "\n".join(["\t" + it for it in items])
    return (header + "\n" if header else "") + body

API_URL = "http://localhost:8000/streamquery"
PRE_STYLE = "max-width: 860px; width: 100%; white-space: pre; overflow-x: auto;"

st.set_page_config(page_title="RAG Stream Demo", page_icon="💬", layout="centered")
st.title("RAG Stream 输出演示")

# 全局自适应样式：移动端/小屏也能良好换行显示
st.markdown(
    """
<style>
  .s-pre {
    max-width: 100%;
    width: 100%;
    white-space: pre-wrap;
    word-break: break-word;
    overflow-wrap: anywhere;
    font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, 'Liberation Mono', 'Courier New', monospace;
    tab-size: 4;
    font-size: clamp(12px, 2.8vw, 16px);
    line-height: 1.45;
  }
  /* 收紧聊天消息的左右留白，提升小屏利用率 */
  section.main > div.block-container { padding-left: 0.8rem; padding-right: 0.8rem; }
  .stChatMessage { padding-left: 0.25rem; padding-right: 0.25rem; }
</style>
""",
    unsafe_allow_html=True,
)

# 关闭侧边栏配置：启用极速渲染与最简界面
stream_speed = 0.0
show_debug = False

if "history" not in st.session_state:
    st.session_state.history = []  # [(role, text)]
if "stream_debug" not in st.session_state:
    st.session_state.stream_debug = []  # 保存最近一次流会话的原始行
if "sse_count" not in st.session_state:
    st.session_state.sse_count = 0  # 统计收到的数据帧数量（不含keepalive/DONE）

# 统一渲染函数：使用 text 原样渲染，彻底绕过 Markdown/HTML 的折行与转义问题
def _render_pre(container, text: str) -> None:
    # 自适应（手机/小屏）+ 原样换行/缩进保留
    safe = html.escape(text)
    container.markdown(f"<div class='s-pre'>{safe}</div>", unsafe_allow_html=True)

def render_history():
    for role, text in st.session_state.history:
        if role == "user":
            st.chat_message("user").write(text)
        else:
            box = st.chat_message("assistant")
            _render_pre(box, text)

render_history()

if prompt := st.chat_input("输入你的问题..."):
    st.session_state.history.append(("user", prompt))
    st.chat_message("user").write(prompt)

    # 占位容器用于逐步追加
    msg_box = st.chat_message("assistant")
    placeholder = msg_box.empty()

    # 清空上一次调试缓存与计数
    st.session_state.stream_debug = []
    st.session_state.sse_count = 0

    try:
        # 直接消费 SSE 流
        with requests.post(
            API_URL,
            json={"query": prompt, "advance": False},
            headers={"Accept": "text/event-stream"},
            timeout=300,
            stream=True,
        ) as resp:
            resp.raise_for_status()
            # 无侧边栏调试输出
            buf = []
            full_text = []
            # SSE 事件缓冲：按空行分隔事件，拼接多行 payload
            evt_lines = []
            for raw in resp.iter_lines(decode_unicode=True):
                if raw is None:
                    continue
                line = raw.rstrip("\r")
                # 记录原始行以便排查
                st.session_state.stream_debug.append(line)

                if line == "":
                    # 一个事件结束，处理累计的 data 行
                    if not evt_lines:
                        continue
                    # 将 data: 前缀去除并用 \n 复原；如果全是空data行，保留为一个换行
                    def _strip_data_prefix(s: str) -> str:
                        # 仅移除前缀 "data: " 的一个空格，不移除制表符等有效前导字符
                        if s.startswith("data: "):
                            return s[len("data: "):]
                        if s.startswith("data:"):
                            # 兼容无空格写法，仅去除前缀，不做 lstrip
                            return s[len("data:"):]
                        return s

                    contents = [_strip_data_prefix(l) for l in evt_lines]
                    if contents and all(c == "" for c in contents):
                        payload = "\n"
                    else:
                        payload = "\n".join(contents)
                    evt_lines = []

                    if payload == "[keepalive]":
                        continue
                    if payload == "[DONE]":
                        break

                    # 兼容错误事件(JSON)
                    if payload.startswith("{"):
                        try:
                            obj = json.loads(payload)
                            if "error" in obj:
                                placeholder.error(f"服务端错误: {obj['error']}")
                                break
                        except Exception:
                            pass

                    # 常规文本增量（可能包含换行）
                    buf.append(payload)
                    full_text.append(payload)
                    display_text = "".join(buf)
                    _render_pre(placeholder, display_text)
                    # 极速渲染：不sleep、不计数
                else:
                    # 非空行累加入事件缓冲（包含或不包含 data: 前缀）
                    evt_lines.append(line)
            final_text = "".join(full_text)
            if final_text:
                st.session_state.history.append(("assistant", final_text))
            else:
                st.session_state.history.append(("assistant", ""))
    except Exception as e:
        placeholder.error(f"请求失败: {e}")

    # 展示调试信息
    if show_debug and st.session_state.stream_debug:
        with st.sidebar.expander("SSE 原始帧（最近一次）", expanded=False):
            st.code("\n".join(st.session_state.stream_debug))


