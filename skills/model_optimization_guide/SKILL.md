# 模型调优与优化规则 (Model Optimization Guide)

本指南记录了针对不同彩种（尤其是双色球 SSQ）进行的机器学习与统计模型参数优化成果及通用调优方法论。

## 1. 机器学习集成模型 (B, C, E, F)

针对双色球（SSQ）红蓝球分别进行了基于**随机搜索 (Random Search)** 的超参数调优。

### 1.1 模型 B (随机森林 RF)
- **红球 (Red)**: `n_estimators=800`, `max_depth=20`, `min_samples_split=5`, `min_samples_leaf=2`, `max_features='sqrt'`。
- **蓝球 (Blue)**: `n_estimators=300`, `max_depth=10`, `min_samples_split=10`, `min_samples_leaf=4`, `max_features='log2'`。
- **调优得分**: 红球 11.67%, 蓝球 15.00%。

### 1.2 模型 C (XGBoost)
- **红球 (Red)**: `n_estimators=800`, `max_depth=5`, `learning_rate=0.05`, `subsample=0.8`, `colsample_bytree=0.8`, `scale_pos_weight=5`。
- **蓝球 (Blue)**: `n_estimators=500`, `max_depth=3`, `learning_rate=0.1`, `subsample=1.0`, `scale_pos_weight=1`。
- **调优得分**: 红球 13.33%, 蓝球 **20.00%**。

### 1.3 模型 E (LightGBM)
- **红球 (Red)**: `n_estimators=1000`, `num_leaves=63`, `learning_rate=0.03`, `feature_fraction=0.8`, `bagging_fraction=0.8`, `is_unbalance=True`。
- **蓝球 (Blue)**: `n_estimators=500`, `num_leaves=15`, `learning_rate=0.05`, `is_unbalance=False`。
- **调优得分**: 红球 **14.17%**, 蓝球 15.00%。

### 1.4 模型 F (CatBoost)
- **红球 (Red)**: `iterations=800`, `depth=6`, `learning_rate=0.03`, `l2_leaf_reg=3`。
- **蓝球 (Blue)**: `iterations=300`, `depth=4`, `learning_rate=0.05`, `l2_leaf_reg=5`。
- **调优得分**: 红球 12.50%, 蓝球 10.00%。

---

## 2. 深度学习模型 (D)

### 2.1 模型 D (LSTM)
通过贝叶斯优化/稳健性经验优化了 LSTM 走势捕捉模型。
- **红球 (Red)**: 
    - **设置**: `embedding_dim=32`, `hidden_dim=64`, `num_layers=2`, `dropout=0.2`, `lr=0.001`, `epochs=80`。
    - **原理**: 红球号码关联性复杂，增加隐藏层深度和嵌入维度能更好地捕捉序列依赖。
- **蓝球 (Blue)**: 
    - **设置**: `embedding_dim=16`, `hidden_dim=32`, `num_layers=1`, `dropout=0.1`, `lr=0.002`, `epochs=50`。
    - **原理**: 蓝球样本较小，轻量化结构能防止模型在过少的样本上产生过拟合。

---

## 3. 模型 A (统计相似度 SM)

### 核心逻辑
基于启发式规则，通过在大规模历史数据中搜索与当前遗漏/统计特征相似的期数，并对后续号码进行加权统计来预测。

### 优化参数 (针对双色球 SSQ)
- **调优方法**: 结合网格搜索 (Grid Search) 寻找 `search_limit` 和 `top_matches` 的最佳范围，并使用 **Optuna** (贝叶斯优化) 迭代优化 `weights` 字典。
- **红球 (Red Balls)**:
    - **设置**: `search_limit=3000`, `top_matches=20`。
    - **权重**: `overlap: 13.58`, `consecutive: 2.10`, `omission: 3.20`。
    - **得分**: **14.33%**。
- **蓝球 (Blue Balls)**:
    - **设置**: `search_limit=2000`, `top_matches=15`。
    - **权重**: `overlap: 8.50`, `neighbor: 2.00`, `omission: 4.00`。
    - **得分**: **8.00%**。

---

## 4. 模型 G (隐马尔可夫模型 HMM)

### 核心逻辑
通过 GaussianHMM 捕捉开奖号码序列中的隐藏状态转换，识别号码分布的潜在模式。

### 优化参数 (针对双色球 SSQ)
- **调优方法**: 使用网格搜索配合贝叶斯信息准则 (BIC)。
- **红球 (Red Balls)**: `n_components=5`, `covariance_type='diag'`。
- **蓝球 (Blue Balls)**: `n_components=3`, `covariance_type='full'`。

---

## 5. 模型 H (极值理论 EVT)

### 核心逻辑
监控近期“和值”合“跨度”的偏差，触发均值回归信号。
**近期优化**：单纯的 EVT 回撤会导致模型严格输出平坦的线性序列（如连号 1, 2, 3 或 33, 32, 31）。为了解决此问题，模型现在会：
1. **构建频率基线 (Frequency Baseline)**：先统计近期窗口内的热号频率作为全部号码的初始基础概率。
2. **应用 EVT 乘数 (Multiplier Effect)**：将 EVT 的回撤力量 (Reversion Force) 和跨度力量 (Span Force) 作为乘数应用在频率基线上。这样可以在追求均值回归的同时，仍然优先选择符合近期出号趋势的活跃号码。
3. **蓝球独立处理**：对于红蓝分离池（`separate_pool=True`），蓝球现在拥有独立的 EVT 均值回归检测回环。

### 优化参数 (针对双色球 SSQ)
- **`recent_periods`**: 回调至 **50** (根据优化脚本，50期能在不过度平滑的情况下捕捉极值反转，平均排名从 16.67 升至 16.48)。
- **`sum_min` / `sum_max` (动态极值阈值)**: 采用 **Mean ± 3σ**。SSQ 实例: `[37, 164]`。

---

## 6. 模型 I (遗传算法 GA)

### 核心逻辑
通过遗传算法进化出与近期形态匹配得分最高的号码组合。针对蓝球（独立池），则构建基于相同历史区间的真实频次概率分布基线，解决预测输出呆板连号（1,2,3...）的问题。

### 优化参数 (针对双色球 SSQ)
- **红球 (Red Balls)**: `mutation_rate=0.1`, `fitness_periods=10`。
- **蓝球 (Blue Balls)**: `mutation_rate=0.15`, `fitness_periods=50` (根据优化脚本最新结论，蓝球在 50 期频次基线上施加 0.15 突变率，可将排位命中提升至平均 7.33，大幅优于旧参数的 9.60)。

---

## 7. 重要系统级修复与增强 (System Debugging & Enhancements)

在多模型系统演进过程中，特别针对以下底层逻辑进行了增强修复：

### 7.1 独立池参数透传修复 (`blue_config` Injection)
**问题**: 针对红蓝独立池（如双色球），`multi_model.py` 的集成预测循环曾未将蓝球专用超参数包 (`blue_config`) 独立透传给 H、I、J 等模型。导致蓝球运算时被强制降级使用了红球的短周期参数（例如红球 `fitness_periods: 10` 覆盖了原本应给蓝球计算的 `fitness_periods: 50`）。
**修复**: 增强了参数抽取管道 `get_specific_config`，在发现 `separate_pool=True` 时，显式打包 `params['blue_config']`，使得模型在处理蓝球时得以解包并正确使用超长参考周期。

### 7.2 频率概率平局微扰 (Tie-Breaker Noise)
**问题**: 严重影响基于频率进行预测的模块（如旧版模型 I 蓝球预测），当遇到多个历史频次相同的（甚至出现次数均为0的）备选球时，NumPy 的 `argsort` 会默认按号码本身的数字大小原序排列，引发极其死板的极低序号死循环输出：如 `[1, 2, 3, 4, 5, ...]`。
**修复**: 为每个球的基础概率/频次施加微小的随机扰动值 (`np.random.random() * 0.0001`)。此举在不破坏原有分布阶层的前提下，彻底打破了频率相同时带来的硬性排序死板现象。

### 7.3 多模型指令并行回测
**增强**: 重构了命令行 `argparse` 处理机制，如今支持输入如 `--method H,I,J` (逗号隔开的一体化字符串) 以同时运行多个指定模型，配合自动化生成的多模型聚合 Ensemble 打分输出，更方便实施模型间实时竞赛。

---

## 8. 通用调优方法论 (Methodology)

### 评估指标
- **命中率 (Hit Rate)**: 红球关注 Top 6/10，蓝球关注 Top 1。
- **平均排名 (Average Rank)**: 越小代表模型对真实开奖号码的预判越精准。

### 调优工具建议
- **Optuna**: 适用于复杂权重优化。
- **ParameterSampler (Scikit-learn)**: 适用于机器学习模型的多超参数随机搜索。
- **网格搜索**: 适用于低维度的结构参数。
