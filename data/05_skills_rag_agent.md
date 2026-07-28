# 技能：RAG 系统与 AI Agent 开发

## RAG 系统（熟练）
- **核心链路**：文档解析（PDF/Word）、文本 Chunk 切分策略。
- **向量检索**：了解 Milvus 及混合检索（向量 + 关键词）基本原理。
- **应用框架**：能基于 LangChain / LlamaIndex 搭建简单问答原型，支持开源模型 API 或本地模型。

## AI Agent 开发（熟练）
- 理解 ReAct 设计模式，有 Function Calling 编排经验。
- 了解多 Agent 协作架构，能判断 Agent 与 Workflow 的选型边界。

## 大模型与提示词工程（熟练）
- 大语言模型基本原理与主流模型特点。
- 能针对复杂场景设计并迭代优化提示词（Prompt Engineering）。

## 技术栈补充
- 编程语言：Python（熟练）、Java（熟悉）
- Web 框架：FastAPI / Flask
- 数据库：MySQL、Redis
- 工程化：Linux、Git、Docker
- 数据处理：Pandas + Matplotlib/Seaborn
- 
## 项目实践经验
- 独立搭建了基于 LlamaIndex + Ollama 的本地 RAG 问答原型（个人简历问答系统）。
- 实现了向量检索 + BM25 关键词检索（jieba 分词）的混合检索，并使用 RRF 融合。
- 集成 BGE-reranker-v2-m3 进行重排序，并解决离线环境下的模型加载问题。
- 通过查询扩展（LLM 改写）和 LLM 自动化评估，构建了完整的检索质量验证闭环。