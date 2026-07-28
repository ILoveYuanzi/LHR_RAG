import streamlit as st
import os
import requests
import jieba


from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    load_index_from_storage,
    Settings,
)
from llama_index.core.node_parser import MarkdownNodeParser   # 改用 Markdown 切分器
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.retrievers.bm25 import BM25Retriever
from reranker import BGEReranker
# ========== 配置 ==========
DATA_DIR = "data"
PERSIST_DIR = "./storage"
EMBED_MODEL = "nomic-embed-text"
LLM_MODEL = "qwen2:0.5b"
OLLAMA_BASE_URL = "http://localhost:11434"
HYBRID_TOP_K = 5

Settings.embed_model = OllamaEmbedding(
    model_name=EMBED_MODEL,
    base_url=OLLAMA_BASE_URL,
)
Settings.llm = Ollama(
    model=LLM_MODEL,
    base_url=OLLAMA_BASE_URL,
    temperature=0.1,
    request_timeout=120.0,
)

# ========== 中文分词器 ==========
def chinese_tokenizer(text: str):
    return list(jieba.cut(text))

# ========== 索引管理 ==========
@st.cache_resource
def get_index():
    if not os.path.exists(DATA_DIR):
        st.error(f"数据目录 `{DATA_DIR}` 不存在，请创建并放入 .md 文件。")
        st.stop()

    if os.path.exists(PERSIST_DIR):
        try:
            storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
            index = load_index_from_storage(storage_context)
            return index
        except Exception:
            st.warning("索引加载失败，将重新构建...")

    st.info("首次运行，正在读取 Markdown 资料并构建向量索引，请稍候...")
    documents = SimpleDirectoryReader(DATA_DIR).load_data()
    if len(documents) == 0:
        st.error(f"`{DATA_DIR}` 中没有找到任何文件，请放入 .md 文件。")
        st.stop()

    # 使用 Markdown 切分器：按标题层级切分，每块 400 token，重叠 50
    node_parser = MarkdownNodeParser(
        chunk_size=400,
        chunk_overlap=50,
    )
    index = VectorStoreIndex.from_documents(
        documents,
        transformations=[node_parser],
    )
    index.storage_context.persist(persist_dir=PERSIST_DIR)
    st.success("索引构建完成！")
    return index

# ========== 混合检索器 ==========
def get_hybrid_query_engine(index):
    vector_retriever = index.as_retriever(similarity_top_k=4)
    all_nodes = list(index.docstore.docs.values())

    bm25_retriever = BM25Retriever.from_defaults(
        nodes=all_nodes,
        similarity_top_k=4,
        tokenizer=chinese_tokenizer,
    )

    fusion_retriever = QueryFusionRetriever(
        [vector_retriever, bm25_retriever],
        similarity_top_k=10,            # 召回 10 个供 Reranker 选择
        num_queries=1,
        mode="reciprocal_rerank",
        use_async=False,
    )

    reranker = BGEReranker(top_n=5)     # 使用自定义重排序器

    query_engine = RetrieverQueryEngine.from_args(
        fusion_retriever,
        node_postprocessors=[reranker]
    )
    return query_engine

# ========== Streamlit 界面 ==========
st.set_page_config(page_title="对话我的简历", page_icon="📄")
st.title("📄 跟我的数字分身聊聊吧")
st.caption("基于混合检索（向量+BM25+中文分词）的 RAG 简历问答系统")

# 检查 Ollama 服务
try:
    resp = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=2)
    if resp.status_code != 200:
        st.error("❌ 无法连接到 Ollama 服务，请确保 Ollama 已启动。")
        st.stop()
except Exception:
    st.error("❌ 未检测到 Ollama 服务，请先启动桌面应用或运行 `ollama serve`。")
    st.stop()

try:
    index = get_index()
    query_engine = get_hybrid_query_engine(index)
except Exception as e:
    st.error(f"初始化失败：{e}")
    st.stop()

# 聊天历史
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "你好！我是简历助手，你可以问我任何关于我的经历、技能、项目的问题～"}
    ]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("请输入你的问题"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("思考中..."):
            response = query_engine.query(prompt)
            answer = str(response)
            st.markdown(answer)

            # 展示检索来源
            with st.expander("📎 信息来源"):
                source_nodes = response.source_nodes
                for i, node in enumerate(source_nodes):
                    file = node.metadata.get("file_name", "未知")
                    text_preview = node.text[:100].replace("\n", " ")
                    st.markdown(f"**{i+1}.** `{file}` — {text_preview}...")

    st.session_state.messages.append({"role": "assistant", "content": answer})