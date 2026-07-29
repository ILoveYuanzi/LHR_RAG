import streamlit as st
import os
import time
import requests
import jieba
from llama_index.core import (
    VectorStoreIndex, SimpleDirectoryReader, StorageContext, load_index_from_storage, Settings,
)
from llama_index.core.node_parser import MarkdownNodeParser
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core.response_synthesizers import get_response_synthesizer
from llama_index.core.schema import QueryBundle
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

# ========== 中文分词 ==========
def chinese_tokenizer(text):
    return list(jieba.cut(text))

# ========== 索引管理 ==========
@st.cache_resource
def get_index():
    if not os.path.exists(DATA_DIR):
        st.error(f"数据目录 `{DATA_DIR}` 不存在，请创建并放入 .md 文件。")
        st.stop()

    # 尝试加载已有索引
    if os.path.exists(PERSIST_DIR):
        try:
            storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
            index = load_index_from_storage(storage_context)
            st.success("✅ 已加载现有索引")
            return index
        except Exception as e:
            st.warning(f"索引加载失败 ({e})，将重新构建...")

    # 构建新索引
    with st.status("正在构建向量索引...", expanded=True) as status:
        st.write("📖 读取资料文件...")
        documents = SimpleDirectoryReader(DATA_DIR).load_data()
        if len(documents) == 0:
            st.error(f"`{DATA_DIR}` 中没有找到任何文件，请放入 .md 文件。")
            st.stop()

        st.write(f"🔹 共读取到 {len(documents)} 个文档，正在切分与向量化...")
        node_parser = MarkdownNodeParser(chunk_size=400, chunk_overlap=50)
        start_time = time.time()
        index = VectorStoreIndex.from_documents(
            documents,
            transformations=[node_parser],
            show_progress=True,
        )
        index.storage_context.persist(persist_dir=PERSIST_DIR)
        elapsed = time.time() - start_time
        status.update(label=f"✅ 索引构建完成！耗时 {elapsed:.1f} 秒", state="complete")
    return index

# ========== 混合检索器 + 重排序器 ==========
def get_retriever_and_reranker(index):
    vector_retriever = index.as_retriever(similarity_top_k=4)
    all_nodes = list(index.docstore.docs.values())
    bm25_retriever = BM25Retriever.from_defaults(
        nodes=all_nodes, similarity_top_k=4, tokenizer=chinese_tokenizer
    )
    fusion_retriever = QueryFusionRetriever(
        [vector_retriever, bm25_retriever],
        similarity_top_k=10,
        num_queries=1,
        mode="reciprocal_rerank",
        use_async=False,
    )
    reranker = BGEReranker(top_n=5)
    return fusion_retriever, reranker

# ========== 查询重写 ==========
def rewrite_query(original_query):
    prompt = (
        "你是一个查询重写助手。请将以下问题改写成 2 个不同角度、但含义相同的查询，"
        "以便从简历文档中检索信息。\n"
        "只输出查询本身，每行一个，不要编号，不要解释。\n\n"
        f"原始问题：{original_query}\n改写查询："
    )
    try:
        resp = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={
                "model": LLM_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.3, "num_predict": 100},
            },
            timeout=15,
        )
        text = resp.json().get("response", "").strip()
        rewritten = [line.strip() for line in text.split("\n") if line.strip()][:2]
        return rewritten if rewritten else [original_query]
    except Exception:
        return [original_query]

# ========== 指代消解 ==========
def resolve_coreference(history, current_query, n=3):
    if len(history) <= 1:  # 仅有欢迎语
        return current_query
    # 排除最初的欢迎消息
    recent = [msg for msg in history if msg["role"] != "assistant" or "你好！我是" not in msg["content"]]
    recent = recent[-(n*2):]
    formatted = []
    for msg in recent:
        if msg["role"] == "user":
            formatted.append(f"用户：{msg['content']}")
        elif msg["role"] == "assistant":
            content = msg["content"][:200]
            formatted.append(f"助手：{content}")
    context = "\n".join(formatted)
    prompt = f"""你是一个对话改写助手。请根据对话历史，将用户的当前问题改写为完整的、不依赖上下文也能理解的独立问题。
只需输出改写后的问题，不要解释。

对话历史：
{context}

用户当前问题：{current_query}
改写后的独立问题："""
    try:
        resp = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={"model": LLM_MODEL, "prompt": prompt, "stream": False, "options": {"temperature": 0.1, "num_predict": 80}},
            timeout=15
        )
        resolved = resp.json().get("response", "").strip()
        return resolved if resolved else current_query
    except:
        return current_query

# ========== 界面 ==========
st.set_page_config(page_title="对话我的简历", page_icon="📄")
st.title("📄 跟我的数字分身聊聊吧")
st.caption("基于混合检索 + 重排序 + 查询重写 + 指代消解的 RAG 简历问答系统")

# 检查 Ollama 服务
try:
    resp = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=3)
    if resp.status_code != 200:
        st.error("❌ 无法连接到 Ollama 服务，请确认已启动。")
        st.stop()
except Exception:
    st.error("❌ 未检测到 Ollama 服务，请先启动桌面应用或运行 `ollama serve`。")
    st.stop()

# 加载索引和检索引擎（关键初始化）
try:
    index = get_index()
    fusion_retriever, reranker = get_retriever_and_reranker(index)   # 这里必须赋值
except Exception as e:
    st.error(f"初始化失败：{e}")
    st.stop()

# 聊天历史
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "你好！我是简历助手，可以问我任何问题～"}
    ]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("请输入你的问题"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            # 指代消解
            resolved = resolve_coreference(st.session_state.messages[:-1], prompt)
            with st.expander("🔍 指代消解 & 查询重写"):
                st.write(f"**原始问题**：{prompt}")
                st.write(f"**消解后**：{resolved}")

            with st.spinner("思考中（指代消解 + 多路召回 + 重排序）..."):
                # 查询重写
                rewritten = rewrite_query(resolved)
                all_queries = [resolved] + rewritten

                # 多路检索 + 去重
                all_nodes = []
                for q in all_queries:
                    nodes = fusion_retriever.retrieve(q)
                    all_nodes.extend(nodes)
                seen = set()
                unique_nodes = []
                for n in all_nodes:
                    if n.node_id not in seen:
                        seen.add(n.node_id)
                        unique_nodes.append(n)

                # 重排序
                reranked = reranker.postprocess_nodes(unique_nodes, QueryBundle(resolved))
                final_nodes = reranked[:HYBRID_TOP_K]

                # 生成回答
                synthesizer = get_response_synthesizer(response_mode="compact")
                response = synthesizer.synthesize(resolved, final_nodes)
                answer = str(response)
                st.markdown(answer)

                # 展示来源
                with st.expander("📎 信息来源（多查询融合后重排序）"):
                    for i, node in enumerate(final_nodes):
                        file = node.metadata.get("file_name", "未知")
                        text_preview = node.text[:100].replace("\n", " ")
                        st.markdown(f"**{i+1}.** `{file}` — {text_preview}...")

        except Exception as e:
            answer = f"⚠️ 生成回答失败：{e}"
            st.error(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})