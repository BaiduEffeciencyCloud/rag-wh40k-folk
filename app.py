import streamlit as st
import logging
import os
import argparse
from config import (
    OPENAI_API_KEY,
    PINECONE_API_KEY,
    PINECONE_INDEX,
    DEFAULT_TEMPERATURE,
    LOG_FORMAT,
    LOG_FILE,
    LOG_LEVEL,
    APP_TITLE,
    APP_ICON,
    APP_HEADER
)

# 配置日志
logging.basicConfig(
    filename=LOG_FILE,
    level=getattr(logging, LOG_LEVEL),
    format=LOG_FORMAT
)
logger = logging.getLogger(__name__)

# 初始化OpenAI客户端
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# 解析命令行参数
parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str, default='cot',
                    help='查询处理模式：cot（思维链检索）、exp（传统扩展器+向量检索）、vector（原始向量检索）')
parser.add_argument('--top_k', type=int, default=5,
                    help='返回结果数量')
args = parser.parse_args()

MODE = args.mode or os.getenv('MODE', 'cot')
TOP_K = args.top_k or int(os.getenv('TOP_K', 5))

# 设置页面配置
st.set_page_config(
    page_title=APP_TITLE,
    page_icon=APP_ICON,
    layout="wide"
)

# 自定义CSS样式
st.markdown("""
    <style>
    .stTextInput>div>div>input {
        font-size: 18px;
    }
    .stButton>button {
        width: 100%;
        font-size: 18px;
    }
    </style>
""", unsafe_allow_html=True)

def main():
    """主函数"""
    st.title(APP_ICON+APP_TITLE)
    st.header(APP_HEADER)
    
    # 显示当前模式
    if MODE == "cot":
        mode_display = "思维链检索（cot_search）"
    elif MODE in ["exp", "expand"]:
        mode_display = "传统扩展器+向量检索"
    else:
        mode_display = "原始向量检索"
    st.sidebar.info(f"当前运行模式：{mode_display}")

    # 初始化不同模式下的处理器
    if MODE == "cot":
        from cot_search import CoTSearch
        cot_search = CoTSearch()
    elif MODE in ["exp", "expand"]:
        from query_expander import QueryExpander
        from vector_search import VectorSearch
        processor = QueryExpander(temperature=DEFAULT_TEMPERATURE)
        vector_search = VectorSearch(
            pinecone_api_key=PINECONE_API_KEY,
            index_name=PINECONE_INDEX,
            openai_api_key=OPENAI_API_KEY
        )
    else:
        from vector_search import VectorSearch
        vector_search = VectorSearch(
            pinecone_api_key=PINECONE_API_KEY,
            index_name=PINECONE_INDEX,
            openai_api_key=OPENAI_API_KEY
        )

    # 创建输入框
    user_query = st.text_input("请输入您的规则查询：", placeholder="例如：载具在近战范围内可以使用警戒射击技能吗？")

    # 创建提交按钮
    if st.button("提交问题"):
        if user_query:
            with st.spinner("机魂正在思索..."):
                try:
                    if MODE == "cot":
                        result = cot_search.search(user_query, top_k=TOP_K)
                        st.write("\n💡 最终答案：")
                        st.write(result.get('final_answer', '无答案'))
                        st.write("\n🔍 检索详情：")
                        for i, r in enumerate(result.get('integrated_results', []), 1):
                            st.write(f"{i}. {r.text}")
                    elif MODE in ["exp", "expand"]:
                        expanded_query = processor.expand_query(user_query)
                        st.write(f"\n🤔 扩展后的查询：{expanded_query}")
                        st.write("\n搜索结果：")
                        results = vector_search.search_and_generate_answer(expanded_query, top_k=TOP_K)
                        for i, result in enumerate(results, 1):
                            st.write(f"{i}. {result['text']}")
                    else:
                        st.write("\n搜索结果：")
                        results = vector_search.search_and_generate_answer(user_query, top_k=TOP_K)
                        for i, result in enumerate(results, 1):
                            st.write(f"{i}. {result['text']}")
                except Exception as e:
                    st.error(f"处理查询时出错：{str(e)}")
        else:
            st.warning("请输入问题！")

    # 添加模式说明
    st.markdown("### 模式说明")
    st.markdown("""
        **查询扩展模式**：
        - **cot模式**：使用思维链检索（cot_search），多变体检索+LLM答案生成
        - **exp/expand模式**：传统扩展器+向量检索
        - **vector模式**：原始向量检索
    """)

if __name__ == "__main__":
    main() 