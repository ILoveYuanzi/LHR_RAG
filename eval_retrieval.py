import json
import jieba
import requests
from llama_index.core import StorageContext, load_index_from_storage, Settings
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core.schema import QueryBundle
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.retrievers.bm25 import BM25Retriever
from reranker import BGEReranker

# ========== 配置 ==========
Settings.embed_model = OllamaEmbedding(model_name="nomic-embed-text", base_url="http://localhost:11434")
Settings.llm = None  # 评估阶段不需要生成

PERSIST_DIR = "./storage"
OLLAMA_BASE = "http://localhost:11434"
EVAL_MODEL = "qwen2:0.5b"  # 用于相关性判断

# ========== 中文分词 ==========
def chinese_tokenizer(text):
    return list(jieba.cut(text))

# ========== 加载索引 ==========
try:
    storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
    index = load_index_from_storage(storage_context)
    print("✅ 索引加载成功")
except Exception as e:
    print(f"❌ 索引加载失败：{e}")
    exit(1)

# ========== 混合检索器 + 重排序器 ==========
vector_retriever = index.as_retriever(similarity_top_k=4)
all_nodes = list(index.docstore.docs.values())
bm25_retriever = BM25Retriever.from_defaults(nodes=all_nodes, similarity_top_k=4, tokenizer=chinese_tokenizer)
fusion_retriever = QueryFusionRetriever(
    [vector_retriever, bm25_retriever],
    similarity_top_k=10,
    num_queries=1,
    mode="reciprocal_rerank",
    use_async=False,
)
reranker = BGEReranker(top_n=5)

# ========== 查询重写（用于评估） ==========
def rewrite_query(q):
    prompt = f"请将以下问题改写成2个不同角度的查询，每行一个。\n问题：{q}\n改写："
    try:
        resp = requests.post(
            f"{OLLAMA_BASE}/api/generate",
            json={"model": EVAL_MODEL, "prompt": prompt, "stream": False, "options": {"temperature": 0.3, "num_predict": 100}},
            timeout=15
        )
        text = resp.json().get("response", "").strip()
        lines = [line.strip() for line in text.split('\n') if line.strip()][:2]
        return lines if lines else [q]
    except:
        return [q]

# ========== LLM 相关性判断 ==========
def is_relevant(query, node_text):
    prompt = f"问题：{query}\n资料片段：{node_text}\n请判断该资料是否包含问题的答案，只回答 YES 或 NO。"
    try:
        resp = requests.post(
            f"{OLLAMA_BASE}/api/generate",
            json={"model": EVAL_MODEL, "prompt": prompt, "stream": False, "options": {"temperature": 0}},
            timeout=20
        )
        ans = resp.json().get("response", "").strip().upper()
        return "YES" in ans
    except:
        return False

# ========== 加载测试集 ==========
try:
    with open("eval_questions.json", "r", encoding="utf-8") as f:
        test_cases = json.load(f)
    print(f"📋 测试集共 {len(test_cases)} 条问题")
    if len(test_cases) == 0:
        print("⚠️ 测试集为空，请添加问题！")
        exit(0)
except FileNotFoundError:
    print("❌ eval_questions.json 文件不存在，请创建！")
    exit(1)

# ========== 评估函数 ==========
def evaluate(top_k=3):
    recall_hits = 0
    reciprocal_ranks = []

    for idx, case in enumerate(test_cases, 1):
        query = case["query"]
        print(f"正在评估 [{idx}/{len(test_cases)}]: {query}")

        # 查询重写
        all_queries = [query] + rewrite_query(query)

        # 多路检索
        all_nodes = []
        for q in all_queries:
            nodes = fusion_retriever.retrieve(q)
            all_nodes.extend(nodes)

        # 去重
        seen = set()
        unique_nodes = []
        for n in all_nodes:
            if n.node_id not in seen:
                seen.add(n.node_id)
                unique_nodes.append(n)

        # 重排序
        reranked = reranker.postprocess_nodes(unique_nodes, QueryBundle(query))
        top_nodes = reranked[:top_k]

        # LLM 相关性判断
        relevant_rank = 0
        for rank, node in enumerate(top_nodes, start=1):
            if is_relevant(query, node.text):
                relevant_rank = rank
                break

        if relevant_rank > 0:
            recall_hits += 1
            reciprocal_ranks.append(1/relevant_rank)
        else:
            reciprocal_ranks.append(0)

        print(f"  -> 命中排名: {relevant_rank if relevant_rank > 0 else '未命中'}")

    recall = recall_hits / len(test_cases)
    mrr = sum(reciprocal_ranks) / len(reciprocal_ranks)
    return recall, mrr

# ========== 运行评估 ==========
print("\n开始计算指标...\n")
for k in [3, 5]:
    rec, mrr_val = evaluate(top_k=k)
    print(f"\n【查询重写 + 混合检索 + 重排序 + LLM 裁判】 Recall@{k}: {rec:.3f}  MRR@{k}: {mrr_val:.3f}\n")