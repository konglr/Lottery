---
name: prediction_models
description: 多模型预测引擎详解，包含统计学、机器学习、深度学习及启发式算法模型 (Models A-J)。
---

# 多模型预测引擎 (Multi-Model Prediction Engine)

本项目采用集成学习思想，结合多种不同原理的预测模型（A-J），旨在捕捉彩票数据中的线性、非线性、时序及统计规律。

## 模型列表 (Models List)

### 统计与概率类 (Statistical & Probabilistic)

- **Model A: Statistical Similarity (统计相似度)**
  - **原理**: 基于历史数据的模式匹配。寻找与近期走势（和值、跨度、AC值、连号等）最相似的历史片段，统计其后一期的号码分布。
  - **适用**: 捕捉历史重复规律。

- **Model G: HMM (Hidden Markov Model, 隐马尔可夫模型)**
  - **原理**: 假设号码走势受背后的“隐状态”（如冷/热/偏态）控制。通过观测序列（和值、跨度等）推断当前隐状态，并预测下一状态的号码分布。
  - **适用**: 宏观状态转移分析。

- **Model H: EVT (Extreme Value Theory, 极值理论)**
  - **原理**: 基于均值回归思想。监测核心指标（如和值）的 3σ 异常波动，预测极值后的强力反弹或回归。
  - **适用**: 捕捉“物极必反”的转折点。

- **Model J: Poisson & Convergence (泊松分布)**
  - **原理**: 计算号码的“遗漏压力”。基于泊松分布计算某号码在当前遗漏值下“本该出现”的概率压力。
  - **适用**: 狙击长期未出的冷态号码。

### 机器学习类 (Machine Learning)

- **Model B: Random Forest (随机森林)**
  - **原理**: 集成多个决策树的分类模型。通过特征工程（遗漏、频率、统计指标）训练，输出每个号码出现的概率。
  - **特点**: 抗过拟合能力强，鲁棒性高。

- **Model C: XGBoost (eXtreme Gradient Boosting)**
  - **原理**: 梯度提升决策树。通过迭代优化残差，逐步提升模型精度。
  - **特点**: 训练效率高，对非线性特征捕捉能力强。

- **Model E: LightGBM (Light Gradient Boosting Machine)**
  - **原理**: 基于直方图的决策树算法，采用 GOSS（单侧采样）和 EFB（互斥特征捆绑）。
  - **特点**: 专为高维数据（如快乐8）优化，训练速度极快。

- **Model F: CatBoost (Categorical Boosting)**
  - **原理**: 专为处理类别特征优化的梯度提升算法，采用排序提升（Ordered Boosting）减少预测偏移。
  - **特点**: 在小样本数据上泛化能力优异。

### 深度学习类 (Deep Learning)

- **Model D: LSTM (Long Short-Term Memory)**
  - **原理**: 循环神经网络（RNN）的变体。将彩票视为时间序列，通过记忆单元捕捉长距离的时间依赖关系。
  - **特点**: 擅长挖掘序列中的隐含时序规律。

### 启发式算法类 (Heuristic Algorithms)

- **Model I: GA (Genetic Algorithm, 遗传算法)**
  - **原理**: 模拟生物进化过程。将号码组合视为“染色体”，以近期形态规律（连号、重号等）为适应度函数，进化出最优组合。
  - **特点**: 在庞大的组合空间中寻找符合当前形态热度的最优解。

## 运行与配置

所有模型均通过 `multi_model.py` 统一调度。

```bash
# 运行所有模型预测双色球
python multi_model.py --lottery 双色球 --method all

# 单独运行模型 A
python multi_model.py --lottery 双色球 --method A
```

## 依赖库

- `scikit-learn`: Model A, B, G, H
- `xgboost`: Model C
- `lightgbm`: Model E
- `catboost`: Model F
- `torch`: Model D
- `hmmlearn`: Model G
- `scipy`: Model J
- `pygad`: Model I