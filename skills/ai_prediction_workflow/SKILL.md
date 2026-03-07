---
name: AI Prediction Workflow
description: Logic and process for AI-driven lottery analysis and prediction.
---

# AI Prediction Workflow

This skill documents the technical flow and logic for AI model integration within the Lottery project.

## 1. System Components
- **UI Interaction**: `app.py` -> `render_ai()`
- **Logic Helper**: `funcs/ai_helper.py` -> `prepare_lottery_data_text()`, `generate_ai_prediction()`, `_build_prediction_prompt()`
- **API Clients**: `google-genai` (Gemini), `openai` (NVIDIA, MiniMax, DashScope)

## 2. Logic Flow

### Step 1: Data Preparation
The user selects a lottery and the number of periods (N) to analyze. `app.py` extracts the latest N rows from the historical CSV data.

### Step 2: Data Textualization
`funcs/ai_helper.py` converts the raw dataframe rows into a human-readable text format suitable for Large Language Models (LLMs). This includes draw numbers, red balls, and blue balls.

### Step 3: Prompt Engineering
The system constructs a detailed "Prompt" (instruction book) for the AI. Key components:
- **Expert Role**: Assigns the AI as a professional lottery analyst.
- **Constraints**: Enforces lottery-specific rules (e.g., SSQ ranges 1-33).
- **Morphology Focus**: Explicitly instructs the AI to consider "Morphological distributions" such as consecutive numbers (连号), jump numbers (跳号), and repeated numbers (重号).

### Step 4: API Dispatch
The system routes the prompt to the selected AI provider:
- **Gemini**: Direct integration via `Vertex AI` or `Gemini API`.
- **DeepSeek**: Routes via `OpenAI` client using the `https://api.deepseek.com` base URL.
- **OpenAI Compatible Providers**: Routes via `OpenAI` client using provider-specific base URLs and API keys.

### Step 5: Result Rendering
The AI returns a Markdown report containing:
- Deep analysis of historical patterns.
- 5 groups of recommended numbers (for standard lotteries) or 20 numbers (for KL8).
- Rationale for the selections.

## 3. Key Configuration
API keys must be stored in the `.Renviron` file or system environment variables:
- `GEMINI_API_KEY`
- `DEEPSEEK_API_KEY`
- `NV_API_KEY` (NVIDIA)
- `MINIMAX_API_KEY`
- `ALIYUNCS_API_KEY` (DashScope)

## 4. 数据同步与规范化 (Data Synchronization & Normalization)

项目使用 `request_data_update.py` 进行增量数据更新，并有一套严格的期号（Issue）规范化流程，以解决不同来源数据格式不一致的问题。

### 期号规范化规则 (`normalize_issue`)
根据彩种所属机构，系统会自动转换并统一期号格式：
- **体育彩票 (Sports Lottery)**: 统一使用 **5位** 格式 (`YYNNN`)。
    - 涉及彩种：超级大乐透 (281), 排列三 (283), 排列五 (284), 七星彩 (287)。
    - 转换逻辑：`2026022` (7位) $\rightarrow$ `26022` (5位)；`7001` (4位) $\rightarrow$ `07001` (5位)。
- **福利彩票 (Welfare Lottery)**: 统一使用 **7位** 格式 (`YYYYNNN`)。
    - 涉及彩种：双色球 (1), 福彩3D (2), 七乐彩 (3), 快乐8 (6)。
    - 转换逻辑：`26024` (5位) $\rightarrow$ `2026024` (7位)。

### 增量更新流程 (`request_data_update.py`)
1. **本地检测**: 读取 `data/` 目录下各彩种 CSV，获取最大期号，并调用 `normalize_issue` 进行规范化。
2. **系统检测**: 调用 `get_latest_issue_from_system` 获取官方接口最新期号，并同样进行规范化。
3. **格式化比较**: 将两个规范化后的期号转换为整数进行对比。
4. **抓取与过滤**: 若系统有新期号，请求最近 100 条记录。在解析时对每一条记录执行 `normalize_issue`，确保存入本地 CSV 的期号格式始终保持项目约定的标准。
5. **合并与排序**: 将新记录与旧数据合并，按期号降序排列并去重。

---

## 5. 自动批量预测与归档 (`ai_batch_predict.py`)

系统支持全彩种每日自动预测。当数据同步完成后，`ai_batch_predict.py` 会遍历所有配置的彩种，调用 Gemini 1.5 Pro/Flash 模型进行分析，并将生成的预测报告（带时间戳）保存在 `data/ai_predictions_history.csv` 中，供前端（如 `app.py` 的历史归档页面）展示。

---

## 6. 多模型机器学习引擎 (`multi_model.py`)

项目集成了一套基于传统统计学与现代机器学习的本地预测引擎，支持多种模型的并行分析与综合推荐。

### 6.1 建模逻辑：双奖池 vs 单奖池 (Dual-Pool vs Single-Pool)
根据彩种真实的摇奖机制，系统采用两种不同的建模策略：

*   **双奖池模式 (Separate Pool = True)**:
    *   **适用彩种**: 双色球 (SSQ), 超级大乐透 (DLT), 七星彩 (XQXC)。
    *   **核心逻辑**: 红球和蓝球来自独立奖池（号码可重复）。
    *   **模型目标**: 目标向量构造为 `[红球位向量] + [蓝球位向量]`（例如双色球为 33+16=49位）。
    *   **特征分离**: 红球和蓝球的遗漏值（Omissions）与出现频率分别作为独立特征输入。
    *   **输出**: 独立生成并展示红球与蓝球的推荐列表。

*   **单奖池模式 (Separate Pool = False)**:
    *   **适用彩种**: 七乐彩 (QLC), 快乐8 (KL8), 福彩3D/排列三/排列五。
    *   **核心逻辑**: 所有号码（含特别号）来自同一奖池（号码不重复）。
    *   **模型目标**: 视为单一序列进行整体训练。
    *   **输出**: 针对七乐彩等彩种，从预测的 Top N 号码中按规则分离出“基本号”与“特别号”展示。

### 6.2 特征工程 (Feature Engineering)
引擎动态提取以下核心特征：
- **遗漏特征 (Omission Features)**: 每个号码自上次出现以来的间隔期数。双奖池模式下红蓝球遗漏独立计算。
- **统计特征 (Statistical Features)**: 动态识别并计算和值、AC值、跨度、奇数比例、大号比例、重/邻/孤号分布。
- **模式特征 (Pattern Features)**: 自动识别二连、三连、二跳、三跳等走势模式。
- **末尾频率 (Tail Frequencies)**: 号码尾数 (0-9) 的分布情况。

### 6.3 预测模型与综合策略 (Ensemble Strategy)
系统并行运行四种性质不同的模型：
1.  **Method A (统计相似度)**: 基于历史走势片段的模式匹配与属性范围分析。
2.  **Method B (随机森林 RF)**: 捕捉非线性特征关联，处理高维遗漏数据。
3.  **Method C (XGBoost)**: 梯度提升决策树，强化对强特征（如遗漏）的利用。
4.  **Method D (深度学习 LSTM)**: 使用 Embedding 层处理号码 ID，捕捉时序规律。

**综合推荐 (Ensemble)**:
采用“基于回测命中率权重的标准化概率融合”方案。仅筛选历史回测表现优异的模型进入综合池，通过 Min-Max 标准化消除不同模型概率分布的差异，生成最终建议。
