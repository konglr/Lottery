"""
KL8 V22 系列共享工具 — P2/P3/P4 改进
=======================================

改进时间: 2026-09-11 14:30 GMT+8
作者: Lucky / MiniMax-M3
依据: Gemini 评估报告 B- 改进建议

包含三个核心工具:
  - select_top_n_soft (P4: 4 区位软配额 3-7)
  - score_v22_a6_smooth (P3: V22.A6 公式平滑化 - EMA + tanh)
  - detect_regime (P2: 极端反转期检测器)

用法:
    from kl8_v22_utils import (
        select_top_n_soft,
        score_v22_a6_smooth,
        detect_regime,
        compute_v22_a_baseline
    )
"""

import numpy as np
from collections import Counter


# ============================================================
# P4: 4 区位软配额选号 (3-7)
# ============================================================
def select_top_n_soft(scores, n=20, lo=3, hi=7, max_pool_size=80):
    """
    从 80 号中按评分选 n 个, 强制每区 [lo, hi] 个.

    流程:
      1. 先保证每区选 lo 个 (按评分排序)
      2. 然后按评分补足到 n 个, 每区最多 hi 个

    Args:
        scores: np.ndarray(80,) 每个号的评分
        n:      选号总数 (默认 20)
        lo:     每区最少 (默认 3)
        hi:     每区最多 (默认 7, Gemini 建议)
        max_pool_size: 候选池大小 (默认 80)

    Returns:
        selected: list[int] 选中的号码 (1-80), 已按评分降序
        zone_quota: list[int] 每个区实际选中的个数
    """
    sorted_idx = np.argsort(scores)[::-1]
    selected = []
    zone_count = [0, 0, 0, 0]

    # Step 1: 每区先保证 lo 个 (按评分顺序)
    for z in range(4):
        for idx in sorted_idx:
            if zone_count[z] >= lo:
                break
            num = int(idx) + 1
            if num in selected:
                continue
            if (num - 1) // 20 != z:
                continue
            selected.append(num)
            zone_count[z] += 1

    # Step 2: 补足到 n 个, 按评分顺序, 每区最多 hi 个
    for idx in sorted_idx:
        if len(selected) >= n:
            break
        num = int(idx) + 1
        if num in selected:
            continue
        z = (num - 1) // 20
        if zone_count[z] >= hi:
            continue
        selected.append(num)
        zone_count[z] += 1

    # 如果 n 太大 (例如 25), 还有余量就再补
    if len(selected) < n:
        for idx in sorted_idx:
            if len(selected) >= n:
                break
            num = int(idx) + 1
            if num in selected:
                continue
            z = (num - 1) // 20
            if zone_count[z] >= hi + 2:  # 极端情况放宽
                continue
            selected.append(num)
            zone_count[z] += 1

    # 按评分降序排序
    selected_sorted = sorted(selected, key=lambda x: scores[x-1], reverse=True)
    return selected_sorted, zone_count


# ============================================================
# P2: 极端反转期检测器
# ============================================================
def detect_regime(last_i, data, n_periods,
                  sum_sigma=2.5,         # 和值波动阈值 (σ)
                  zone_imbalance=2.0,    # 4 区位偏离阈值 (σ)
                  lag_overlap_p99=10):   # lag1 重号 P99 阈值
    """
    检测当前期是否处于"极端反转期".

    三个独立信号,任一触发则返回 'extreme':
      1. 近 5 期和值的均值 / 标准差, >sum_sigma 倍标准差远离全期均值
      2. 近 5 期 4 区位分布 vs 全期均值的卡方距离 > zone_imbalance
      3. 上期重号数 > lag_overlap_p99 (通常 lag1 重号均值 5, P99=10)

    返回:
        regime: 'normal' 或 'extreme'
        signals: dict — 各项指标的具体数值
    """
    if last_i + 5 >= n_periods:
        return 'normal', {'reason': 'insufficient_data'}

    # 信号 1: 近 5 期和值偏离
    recent_sums = []
    for k in range(1, 6):
        if last_i + k < n_periods:
            recent_sums.append(int(data[last_i + k].sum()))

    # 全期和值的均值和标准差 (一次性预计算外提, 这里简化用最近 200 期近似)
    base_sums = []
    for li in range(min(n_periods - 5, last_i + 200), last_i + 5):
        if li >= 0 and li + 5 < n_periods:
            base_sums.append(int(data[li].sum()))

    if len(recent_sums) >= 3 and len(base_sums) >= 30:
        recent_mean = np.mean(recent_sums)
        base_mean = np.mean(base_sums)
        base_std = np.std(base_sums) + 1e-9  # 避免除零
        sum_signal = abs(recent_mean - base_mean) / base_std
    else:
        sum_signal = 0.0

    # 信号 2: 近 5 期 4 区位偏离
    recent_zone = np.zeros(4)
    for k in range(1, 6):
        if last_i + k < n_periods:
            for n in data[last_i + k]:
                recent_zone[(n - 1) // 20] += 1
    recent_zone = recent_zone / recent_zone.sum()  # 归一化

    base_zone = np.zeros(4)
    for li in range(min(n_periods - 5, last_i + 200), last_i + 5):
        if li >= 0 and li + 5 < n_periods:
            for n in data[li]:
                base_zone[(n - 1) // 20] += 1
    base_zone = base_zone / base_zone.sum()

    # 卡方距离
    zone_signal = float(np.sum((recent_zone - base_zone) ** 2 / (base_zone + 1e-9)))

    # 信号 3: lag1 重号数
    if last_i + 1 < n_periods:
        lag1_overlap = len(set(data[last_i].tolist()) & set(data[last_i + 1].tolist()))
    else:
        lag1_overlap = 0

    signals = {
        'sum_signal': sum_signal,
        'zone_signal': zone_signal,
        'lag1_overlap': lag1_overlap,
        'recent_sums': recent_sums,
        'recent_zone': recent_zone.tolist(),
        'base_mean': float(base_mean) if base_sums else 0,
        'base_std': float(base_std) if base_sums else 0,
    }

    is_extreme = (
        sum_signal > sum_sigma or
        zone_signal > zone_imbalance or
        lag1_overlap > lag_overlap_p99
    )

    return ('extreme' if is_extreme else 'normal'), signals


# ============================================================
# P3: V22.A6 平滑化 (EMA + tanh)
# ============================================================
def score_v22_a6_smooth(last_i, data, n_periods,
                        lookback=10,        # 用近 10 期 instead of 3
                        ema_halflife=5,     # EMA 半衰期
                        scale=2.0,
                        baseline_c5=None,   # 预计算的 baseline c5 dict
                        c5_fn=None, c10_fn=None, c30_fn=None, nbr_fn=None):
    """
    V22.A6 平滑版 — 改进:
      1. 用 EMA 计算 observed (而非单期 lookback=3), 减少噪声
      2. 用 tanh 限制 adjustment 幅度在 [-1, +1]
      3. 当 observed 样本不足时 (新数据初期), 自动降权

    Args:
        last_i: 当前期索引 (data 降序)
        lookback: 参考最近 N 期 (默认 10, 比 3 更稳健)
        ema_halflife: EMA 半衰期 (默认 5 期)
        scale: 调整强度 (默认 2.0)
        baseline_c5: dict {c5_value: probability} (必须预计算传入)
        c5_fn/c10_fn/c30_fn/nbr_fn: 计算 c5/c10/c30/邻号的函数

    Returns:
        scores: np.ndarray(80,)
        meta: dict — debug 信息 (observed_ema, adj_per_c5, etc.)
    """
    if baseline_c5 is None:
        raise ValueError("baseline_c5 must be pre-computed and passed in")

    # Step 1: 计算每期 observed c5 分布
    period_observed = []  # list of dict {c5: count}
    for i in range(1, lookback + 1):
        actual_i = last_i + i  # 数据降序, +i = 过去期 (P1 已修复)
        if actual_i >= n_periods:
            continue
        c5 = c5_fn(actual_i)
        period_observed.append(c5)

    # Step 2: 用 EMA 加权 (近期权重高)
    if not period_observed:
        # 数据不足, 返回零调整
        s = np.zeros(80)
        return s, {'reason': 'no_observed'}

    ema_weights = np.array([0.5 ** (k / ema_halflife) for k in range(len(period_observed))])
    ema_weights = ema_weights / ema_weights.sum()

    # 对每个 c5 桶 (0-5), 加权统计
    observed_ema = {k: 0.0 for k in range(6)}
    for k_idx, c5 in enumerate(period_observed):
        weight = ema_weights[k_idx]
        for n_idx, cnt in enumerate(c5):
            observed_ema[int(cnt)] = observed_ema.get(int(cnt), 0.0) + weight / 80.0

    # 归一化 (确保 sum=1)
    total = sum(observed_ema.values())
    if total > 0:
        observed_ema = {k: v / total for k, v in observed_ema.items()}

    # Step 3: 基础评分 (V22.A 基线)
    c5 = c5_fn(last_i)
    s = np.zeros(80)
    s[c5 == 0] = -0.5
    s[c5 == 1] = 0.0
    s[c5 == 2] = 0.2
    s[c5 == 3] = 0.2
    s[c5 == 4] = -0.3
    s[c5 == 5] = -0.8
    s[c5 >= 6] = -1.5

    # Step 4: 平滑调整 — 用 tanh 限制幅度
    adj_per_c5 = {}
    for k in range(6):
        diff = observed_ema.get(k, 0) - baseline_c5.get(k, 0)
        # tanh 把任意 diff 限制在 [-1, +1], 乘 scale
        adj = float(np.tanh(diff * 5.0)) * scale  # tanh(5*diff) 接近 ±1, 然后 scale 控制强度
        adj_per_c5[k] = adj
        s[c5 == k] += adj

    # Step 5: c10/c30/邻号 (不变)
    t = np.zeros(80)
    c10 = c10_fn(last_i)
    t[(c10 >= 2) & (c10 <= 4)] = 1.0
    t[c10 == 1] = 0.4
    t[c10 == 0] = 0.2
    t[(c10 >= 5) & (c10 <= 6)] = -0.3
    t[c10 >= 7] = -0.8

    u = np.zeros(80)
    c30 = c30_fn(last_i)
    u[(c30 >= 5) & (c30 <= 10)] = 0.5
    u[(c30 >= 3) & (c30 <= 4)] = 0.2
    u[c30 <= 2] = -0.2
    u[c30 >= 13] = -0.7

    nbr = nbr_fn(last_i)

    final_scores = s + t + u + nbr
    meta = {
        'observed_ema': observed_ema,
        'adj_per_c5': adj_per_c5,
        'n_periods_used': len(period_observed),
        'lookback': lookback,
        'ema_halflife': ema_halflife,
        'scale': scale,
    }
    return final_scores, meta


# ============================================================
# 通用: V22.A 基线评分 (无自适应)
# ============================================================
def compute_v22_a_baseline(last_i, c5_fn, c10_fn, c30_fn, nbr_fn):
    """纯 V22.A 基线评分 — 无自适应调整"""
    c5 = c5_fn(last_i)
    s = np.zeros(80)
    s[c5 == 0] = -0.5
    s[c5 == 1] = 0.0
    s[c5 == 2] = 0.2
    s[c5 == 3] = 0.2
    s[c5 == 4] = -0.3
    s[c5 == 5] = -0.8
    s[c5 >= 6] = -1.5

    t = np.zeros(80)
    c10 = c10_fn(last_i)
    t[(c10 >= 2) & (c10 <= 4)] = 1.0
    t[c10 == 1] = 0.4
    t[c10 == 0] = 0.2
    t[(c10 >= 5) & (c10 <= 6)] = -0.3
    t[c10 >= 7] = -0.8

    u = np.zeros(80)
    c30 = c30_fn(last_i)
    u[(c30 >= 5) & (c30 <= 10)] = 0.5
    u[(c30 >= 3) & (c30 <= 4)] = 0.2
    u[c30 <= 2] = -0.2
    u[c30 >= 13] = -0.7

    nbr = nbr_fn(last_i)
    return s + t + u + nbr