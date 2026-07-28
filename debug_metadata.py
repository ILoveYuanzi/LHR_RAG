# debug_metadata.py
from llama_index.core import StorageContext, load_index_from_storage, Settings
from llama_index.embeddings.ollama import OllamaEmbedding

Settings.embed_model = OllamaEmbedding(model_name="nomic-embed-text", base_url="http://localhost:11434")
PERSIST_DIR = "./storage"

storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
index = load_index_from_storage(storage_context)
retriever = index.as_retriever(similarity_top_k=3)

nodes = retriever.retrieve("测试问题，比如你的技能")
for node in nodes:
    print("=== Node ===")
    print("Text:", node.text[:100])
    print("Metadata:", node.metadata)
    print("Score:", node.score)