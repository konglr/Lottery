# 🎰 Lottery Analysis & Prediction (彩票数据采集、分析与预测)

[![Streamlit App](https://img.shields.io/badge/Streamlit%20App-Online-brightgreen)](https://lottery-analysis.streamlit.app)

基于 Python 和 Streamlit 构建的一站式彩票数据处理平台，集成自动化数据采集、深度统计分析及多模型 AI 预测功能。

## 🌟 核心功能

### 1. 自动化数据管理
- **实时采集**：支持中国福利彩票与体育彩票官方数据的自动增量更新。
- **数据校验**：内置完整性检查脚本，确保历史开奖数据的准确无误。

### 2. 多维统计分析
- **冷热分析**：可视化红蓝球出现频率、遗漏值及百分比。
- **规律挖掘**：包含奇偶比、大小比、连号、跳号、区间分布、和值趋势等 20+ 项技术指标。
- **关联分析**：支持 Top 伴随对、三连组合等深度关联挖掘。

### 3. 多模型 AI 预测
集成全球顶级大模型，通过历史规律辅助选号：
- **Google Gemini** (Gemini 2.0/1.5 系列)
- **NVIDIA NIM** (GLM-4.7, Llama 3.1)
- **MiniMax** (M2.5)
- **阿里 DashScope** (通义千问 Qwen 系列)

### 4. 自动化回测
- 内置模型评估系统，通过历史数据验证预测策略的有效性，并生成可视化回测报告。

## 📊 支持彩种

项目现已支持以下 8 种主流彩票：
- **福利彩票 (CWL)**：双色球 (SSQ)、快乐8 (KL8)、福彩3D (D3)、七乐彩 (QLC)
- **体育彩票 (CSL)**：超级大乐透 (DLT)、排列三 (PL3)、排列五 (PL5)、七星彩 (XQXC)

## 🚀 快速开始

### 本地运行
1. **环境准备**：推荐使用 Python 3.12。
2. **安装依赖**：
   ```bash
   pip install -r requirements.txt
   ```
3. **配置密钥**：在根目录创建 `.Renviron` 或系统环境变量，配置 API 密钥（如 `GEMINI_API_KEY`, `ALIYUNCS_API_KEY` 等）。
4. **启动应用**：
   ```bash
   streamlit run app.py
   ```

### 部署
项目支持通过 GitHub 轻松部署至 **Streamlit Cloud**。部署后请在 Streamlit 控制台的 "Secrets" 中配置相关的 API 密钥。

## 🛠️ 技术栈
- **Frontend**: Streamlit
- **Analysis**: Pandas, NumPy, Collections
- **Visualization**: Altair, Matplotlib
- **AI SDK**: google-genai, openai (compatible APIs)

---

## 🔗 体验网址
[https://lottery-analysis.streamlit.app](https://lottery-analysis.streamlit.app)

**免责声明**：本项目仅用于数据分析与技术研究，不构成任何投资建议。请理性购彩。
