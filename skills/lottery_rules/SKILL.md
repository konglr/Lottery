---
name: lottery_rules
description: 中国福利彩票与体育彩票官方规则手册，包含购彩、开奖、中奖标准及兑奖信息。
---

# 中国彩票官方规则手册

本 Skill 整理了中国福利彩票（CWL）和中国体育彩票（LOTTERY）中常见彩种的官方规则，包括玩法、购彩方式、开奖时间、中奖标准及金额等。

## 目录

### 彩票规则 (Lottery Rules)
- [双色球 (SSQ)](./ssq.md) - 6红+1蓝，双色碰撞，梦想起航。
- [快乐8 (KL8)](./kl8.md) - 01-80选1-10，多法玩法，快乐加倍。
- [福彩3D (3D)](./d3.md) - 0-9选3，数字魅力，固定奖金。
- [七乐彩 (QLC)](./qlc.md) - 30选7，七位一体，幸运连连。

### 体育彩票 (China Sports Lottery)
- [超级大乐透 (DLT)](./dlt.md) - 5+2模式，千万大奖。
- [七星彩 (XQXC)](./xqxc.md) - 6位数字+1位数字，星光闪耀。
- [排列三 (PL3)](./pl3.md) - 0-9选3，直选组选。
- [排列五 (PL5)](./pl5.md) - 0-9选5，固定奖金。

### 技术配置 (Technical Configs)
- [AI API 配置](../ai_api_configs/SKILL.md) - 包含 DashScope, MiniMax, NVIDIA NIM 等接口规范。

## 技术运行环境

本项目基于 **Python 3.12** 开发，并使用专用的虚拟环境进行隔离。

- **Virtual Environment**: `./venv_312`
- **Main Binary**: `./venv_312/bin/python3`
- **Dependencies**: 包含 `torch`, `xgboost`, `scikit-learn`, `streamlit`, `pandas` 等核心库。
- **Usage**: 推荐通过 `./venv_312/bin/python3` 直接执行脚本，或激活环境后运行。

## 项目结构 (Project Structure)

项目已完成通用化改造，支持多种彩种的预测与回测。

- `app.py`: 基于 Streamlit 的 Web 交互界面，支持各彩种预测、回测结果展示及 AI 辅助分析。
- `config.py`: **核心配置文件**。集中管理所有彩种的配置（号码范围、中奖球数、回测阈值等）。
- `multi_model.py`: **通用预测引擎**。支持多种模型（统计、RF、XGBoost、LSTM）的训练与回测。
- `data/`: 数据存储目录。
    - `*_lottery_data.csv`: 各彩种的历史开奖数据。
    - `*_backtest.csv`: 各彩种的自动化回测预测结果，实现结果隔离。
- `request_data_checking.py`: 数据完整性校验脚本。
- `request_process_all_data.py`: 数据预处理与特征生成脚本。
- `skills/`: 项目知识库（含本及各彩种详细规则）。

## 常用通用规则

1. **购彩限制**：禁止向未成年人销售彩票或兑付奖金。
2. **兑奖期限**：自开奖之日起 60 个自然日内，逾期未兑视为弃奖（特殊情况除外）。
3. **有效凭证**：中奖彩票是唯一的兑奖凭证，请妥善保管。
4. **单注上限**：浮动奖单注最高限额通常为 500 万或 1000 万（视彩种及调节基金情况而定）。
5. **投注倍数**：单张彩票多倍投注通常限制在 2-99 倍。

---
*数据来源：中国福利彩票官网、中国体育彩票官网*
