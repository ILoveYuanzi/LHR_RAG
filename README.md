# 📄 RAG Resume Chat — 让面试官与我的数字分身对话

**一个基于全开源本地模型的 RAG 智能简历问答系统。**  
可以把它想象成一个能和你聊天的“数字简历”：你可以直接问它“他有哪些项目经验？”“RAG 系统技能怎么样？”，它会立刻从我的真实简历资料中检索并生成准确回答。

> 🏗️ **项目定位**：展示 **RAG 工程能力 + AI 产品思维** 的作业级项目，全链路自研，非 Demo 拼接。

---

## ✨ 项目亮点

- **全本地、零成本、开箱即用**  
  无需 API Key，不依赖任何付费服务。仅需安装 Ollama 和 Python 环境，一键启动即可在本地运行完整 RAG 管线。
- **工业级检索架构（混合检索 + 重排序）**  
  融合向量检索（语义）与 BM25 检索（关键词），辅以中文分词，再通过 `bge-reranker-v2-m3` 重排序模型精筛片段，大幅提升召回精准度。
- **智能查询理解（Query Rewriting + 指代消解）**  
  自动改写模糊问题为 2 个细化查询进行多路召回；支持多轮对话中的代词消解（如把“他”还原为姓名），让系统真正具备上下文理解能力。
- **可量化的检索评估**  
  提供独立的评估脚本，用 LLM 作为裁判自动计算 `Recall@K` 和 `MRR`，可验证每一步优化带来的真实提升。
- **工程细节严谨**  
  使用 `MarkdownNodeParser` 按简历结构切分文档，保持片段完整性；所有依赖版本锁定，支持离线运行。

---

## 🧠 系统架构
用户提问 → 指代消解（多轮对话） → 查询改写（2路扩展）
#### ↓
混合检索（向量 + BM25） ← 简历向量库（nomic-embed-text）
#### ↓
去重合并 → Reranker（bge-reranker-v2-m3）重排序 Top-5
#### ↓
上下文 + 原问题 → LLM（qwen2.5:7b）生成回答 text
#### ↓

---

## 🛠 技术栈

| 层级       | 技术/模型 | 
|------------|-----------|
| LLM        | Ollama + `qwen2.5:7b`（支持切换） |
| Embedding  | Ollama + `nomic-embed-text` |
| 向量存储   | LlamaIndex `SimpleVectorStore`（JSON 持久化） |
| 检索框架   | LlamaIndex（`QueryFusionRetriever`） |
| 混合检索   | 向量检索 + BM25 关键词检索 + 中文分词（jieba） |
| 重排序     | `BAAI/bge-reranker-v2-m3`（CrossEncoder） |
| 前端       | Streamlit |
| 评估       | 自研 LLM-as-a-Judge（Ollama 小模型评判 `Recall@K / MRR`） |

---

## 📂 项目结构
.
├── data/ # 个人简历 Markdown 文件（按主题拆分）
│ ├── 01_personal_info.md
│ ├── 02_project_news_portal.md
│ ├── 03_project_miniprogram.md
│ ├── 04_skills_web_backend.md
│ ├── 05_skills_rag_agent.md
│ └── 06_skills_others.md
├── app.py # Streamlit 主界面（含完整 RAG 管线）
├── eval_retrieval.py # 检索评估脚本（Recall@K, MRR）
├── reranker.py # 自定义 Reranker 后处理器
├── eval_questions.json # 测试问题集（用于评估）
├── requirements.txt # Python 依赖
└── storage/ # 向量索引持久化（自动生成，已 gitignore）

text

---

## 🚀 快速开始

### 1. 环境准备
- Python 3.9+（推荐 3.12，已适配）
- [Ollama](https://ollama.com) 安装并运行

### 2. 拉取模型
```bash
ollama pull nomic-embed-text
ollama pull qwen2.5:7b      # 或 qwen2:0.5b 轻量测试
3. 安装依赖

bash
pip install -r requirements.txt
4. 准备个人数据

在 data/ 文件夹下放入自己的 .md 简历文件（可参考已提供的示例结构）。

5. 启动应用

bash
streamlit run app.py
首次运行会自动构建向量索引（需等待几分钟），之后启动秒开。
浏览器访问 http://localhost:8501 即可与“数字分身”对话。

📊 检索效果评估

项目内置评估脚本，使用 LLM 自动判断检索结果是否包含答案。

bash
python eval_retrieval.py
输出示例：

text
【查询重写 + 混合检索 + 重排序 + LLM 裁判】 Recall@3: 0.842  MRR@3: 0.700
【查询重写 + 混合检索 + 重排序 + LLM 裁判】 Recall@5: 0.947  MRR@5: 0.720
你也可以通过修改 eval_questions.json 自定义测试问题。

