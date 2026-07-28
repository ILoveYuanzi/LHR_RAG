import json
import jieba
import requests
from llama_index.core import StorageContext, load_index_from_storage, Settings
from llama_index.core.retrievers import QueryFusionRetriever
from reranker import BGEReranker
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.retrievers.bm25 import BM25Retriever

Settings.embed_model = OllamaEmbedding(
    model_name="nomic-embed-text",
    base_url="http://localhost:11434"
)
Settings.llm = None

PERSIST_DIR = "./storage"
OLLAMA_BASE = "http://localhost:11434"
EVAL_MODEL = "qwen2:0.5b"

storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
index = load_index_from_storage(storage_context)

def chinese_tokenizer(text: str):
    return list(jieba.cut(text))

vector_retriever = index.as_retriever(similarity_top_k=4)
all_nodes = list(index.docstore.docs.values())
bm25_retriever = BM25Retriever.from_defaults(
    nodes=all_nodes,
    similarity_top_k=4,
    tokenizer=chinese_tokenizer,
)

# 混合检索器（召回 10 个）
hybrid_retriever = QueryFusionRetriever(
    [vector_retriever, bm25_retriever],
    similarity_top_k=10,
    num_queries=1,
    mode="reciprocal_rerank",
    use_async=False,
)

# Reranker
reranker = BGEReranker(top_n=5)   

def is_relevant(query: str, node_text: str) -> bool:
    prompt = (
        f"问题：{query}\n"
        f"资料片段：{node_text}\n"
        f"请判断该资料片段是否包含问题的答案，只回答 YES 或 NO。"
    )
    try:
        resp = requests.post(
            f"{OLLAMA_BASE}/api/generate",
            json={"model": EVAL_MODEL, "prompt": prompt, "stream": False, "options": {"temperature": 0}},
            timeout=30
        )
        answer = resp.json().get("response", "").strip().upper()
        return "YES" in answer
    except:
        return False

with open("eval_questions.json", "r", encoding="utf-8") as f:
    test_cases = json.load(f)

def evaluate(retriever, reranker, top_k=3):
    recall_hits = 0
    reciprocal_ranks = []

    for case in test_cases:
        query = case["query"]
        raw_nodes = retriever.retrieve(query)       # 召回 10 个
        # 应用 Reranker 得到重排序后的节点列表
        reranked_nodes = reranker.postprocess_nodes(raw_nodes, query_bundle=QueryBundle(query))
        top_nodes = reranked_nodes[:top_k]

        relevant_at = 0
        for rank, node in enumerate(top_nodes, start=1):
            if is_relevant(query, node.text):
                relevant_at = rank
                break

        if relevant_at > 0:
            recall_hits += 1
            reciprocal_ranks.append(1 / relevant_at)
        else:
            reciprocal_ranks.append(0)

    recall = recall_hits / len(test_cases)
    mrr = sum(reciprocal_ranks) / len(reciprocal_ranks)
    return recall, mrr

for k in [3, 5]:
    rec, mrr_val = evaluate(hybrid_retriever, reranker, top_k=k)
    print(f"【混合检索 + Rerank】 Recall@{k}: {rec:.3f}  MRR@{k}: {mrr_val:.3f}")