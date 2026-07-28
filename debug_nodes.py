from llama_index.core import StorageContext, load_index_from_storage, Settings
from llama_index.embeddings.ollama import OllamaEmbedding

Settings.embed_model = OllamaEmbedding(model_name="nomic-embed-text", base_url="http://localhost:11434")
Settings.llm = None

storage_context = StorageContext.from_defaults(persist_dir="./storage")
index = load_index_from_storage(storage_context)
all_nodes = list(index.docstore.docs.values())
print(f"节点总数: {len(all_nodes)}")
for i, node in enumerate(all_nodes[:3]):
    print(f"--- 节点 {i} ---")
    print("文本:", node.text[:200])
    print("元数据:", node.metadata)