import os
os.environ["HF_HUB_OFFLINE"] = "1"   # 强制离线，必须放在所有导入之前

from typing import List, Optional
from llama_index.core.schema import NodeWithScore, QueryBundle
from sentence_transformers import CrossEncoder

class BGEReranker:
    def __init__(self, top_n: int = 5, model_name: str = "BAAI/bge-reranker-v2-m3"):
        self.top_n = top_n
        self.model = CrossEncoder(model_name)

    def postprocess_nodes(
        self, nodes: List[NodeWithScore], query_bundle: Optional[QueryBundle] = None
    ) -> List[NodeWithScore]:
        if query_bundle is None or not nodes:
            return nodes

        texts = [node.text for node in nodes]
        query = query_bundle.query_str

        scores = self.model.predict([(query, text) for text in texts])

        for node, score in zip(nodes, scores):
            node.score = float(score)
        sorted_nodes = sorted(nodes, key=lambda x: x.score, reverse=True)
        return sorted_nodes[:self.top_n]