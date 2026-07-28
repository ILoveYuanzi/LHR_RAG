# 项目：RAG 个人简历问答系统

- **开发周期**：2026年7月
- **担任角色**：独立全栈开发
- **项目地址**：https://github.com/ILoveYuanzi/LHR_RAG

## 项目描述
一个完全本地运行、无需付费 API 的个人简历智能问答系统，支持用户通过自然语言与“数字简历分身”对话。  
系统实现了完整的企业级 RAG 流水线：混合检索（向量 + BM25 + 中文分词）、查询扩展、重排序、流式对话。  
项目代码已开源在 GitHub，可直接克隆运行，并附带自动化评估脚本（Recall@K / MRR）。

## 核心功能
- **混合检索**：结合向量语义检索（Ollama nomic-embed-text）和 BM25 关键词检索（jieba 分词），采用倒数排名融合（RRF）。
- **查询扩展**：使用本地 LLM 将用户问题自动扩展为多个语义相近的查询，提高召回率。
- **重排序（Reranker）**：集成 BGE-reranker-v2-m3 模型对候选片段进行精排，提升 MRR。
- **对话界面**：基于 Streamlit 的聊天 UI，支持多轮对话和回答来源展示。
- **自动评估**：基于 LLM 的相关性评判器，自动计算检索的 Recall@K 和 MRR。
- **离线可用**：所有模型（LLM、Embedding、Reranker）均在本地运行或缓存，无需联网。

## 技术栈
- **LLM**：Ollama 部署的 qwen2:0.5b（对话）、nomic-embed-text（嵌入）
- **RAG 框架**：LlamaIndex（文档加载、索引构建、检索融合）
- **检索组件**：
  - 向量存储：SimpleVectorStore（JSON 持久化）
  - 关键词检索：BM25Retriever + jieba 中文分词
  - 融合检索：QueryFusionRetriever（RRF）
  - 重排序：BGE-reranker-v2-m3（CrossEncoder，离线加载）
- **前端**：Streamlit
- **工程化**：Python 虚拟环境、Git 版本控制、精确依赖管理（requirements.txt）

## 个人职责与亮点
- 独立设计并实现整个 RAG 管线，包括文本切分、索引构建、混合检索与重排序。
- 针对中文简历场景优化了 BM25 分词器（jieba），显著提升关键词召回率。
- 使用 LLM 实现查询扩展和自动化评估，替代传统人工标注。
- 解决了 reranker 离线加载时的网络依赖和环境兼容问题（numpy/scipy 版本冲突）。
- 编写了完整的测试问题集和评估脚本，可量化检索效果（Recall@3 约 0.85，MRR@3 约 0.70）。
- 项目已推送到 GitHub 并配有详细的 README，支持开箱即用。

## 项目成果
一个可直接用于面试展示的“活简历”，体现了 RAG 工程化能力、检索优化经验和全栈开发思维。