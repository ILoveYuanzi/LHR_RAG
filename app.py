import streamlit as st
import os
from pathlib import Path
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    load_index_from_storage,
    Settings,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
import requests

# ========== 配置 ==========
DATA_DIR = "data"
PERSIST_DIR = "./storage"               # 持久化目录（SimpleVectorStore 用）
EMBED_MODEL = "nomic-embed-text"
LLM_MODEL = "qwen2:0.5b"
OLLAMA_BASE_URL = "http://localhost:11434"

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

# ========== 索引管理 ==========
@st.cache_resource
def get_index():
    if not os.path.exists(DATA_DIR):
        st.error(f"数据目录 `{DATA_DIR}` 不存在，请创建并放入 txt 文件。")
        st.stop()

    # 如果已有持久化索引，直接加载
    if os.path.exists(PERSIST_DIR):
        try:
            storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
            index = load_index_from_storage(storage_context)
            return index
        except Exception:
            st.warning("索引加载失败，将重新构建...")

    # 构建新索引
    st.info("首次运行，正在读取资料并构建向量索引，请稍候...")
    documents = SimpleDirectoryReader(DATA_DIR).load_data()
    if len(documents) == 0:
        st.error(f"`{DATA_DIR}` 中没有找到任何文件，请放入 .txt 文件。")
        st.stop()

    # 文本切分（简历适合较小块）
    node_parser = SentenceSplitter(chunk_size=400, chunk_overlap=50)

    # 创建索引（自动使用 SimpleVectorStore 并持久化到 PERSIST_DIR）
    index = VectorStoreIndex.from_documents(
        documents,
        transformations=[node_parser],
    )
    # 保存索引到磁盘
    index.storage_context.persist(persist_dir=PERSIST_DIR)
    st.success("索引构建完成！")
    return index

# ========== Streamlit 界面 ==========
st.set_page_config(page_title="对话我的简历", page_icon="📄")
st.title("📄 跟我的数字分身聊聊吧")
st.caption("完全本地运行的 RAG 简历问答系统")

# 检查 Ollama 服务
try:
    resp = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=2)
    if resp.status_code != 200:
        st.error("❌ 无法连接到 Ollama 服务，请确保 Ollama 已启动。")
        st.stop()
except Exception:
    st.error("❌ 未检测到 Ollama 服务，请先运行 `ollama serve` 或启动桌面应用。")
    st.stop()

# 获取索引
try:
    index = get_index()
    query_engine = index.as_query_engine()
except Exception as e:
    st.error(f"初始化失败：{e}")
    st.stop()

# 初始化聊天历史
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "你好！我是刘浩然的简历助手，你可以问我任何关于他的问题～"}
    ]

# 显示历史消息
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 接收用户输入
if prompt := st.chat_input("请输入你的问题"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("思考中..."):
            response = query_engine.query(prompt)
            answer = str(response)
            st.markdown(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})