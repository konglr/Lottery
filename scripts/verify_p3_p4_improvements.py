"""
P3 + P4 验证: V22.A6 平滑化 + 4 区位软配额
============================================

改进时间: 2026-09-11 15:00 GMT+8
作者: Lucky / MiniMax-M3
依据: Gemini 评估报告 — B- 改进建议 P3 + P4

对比四个版本 (200 期回测):
  V0: V22.A 基线 (无自适应, 严格 5/5/5/5)
  V1: V22.A6 修复版 (P1: bug 修了, lookback=3, scale=2.0, 4 区位 3-8)
  V2: V22.A6 平滑版 (P3: lookback=10, EMA 半衰=5, tanh 限幅, 4 区位 3-8)
  V3: V22.A6 平滑 + 软配额 (P3+P4: 同 V2, 但 4 区位改成 3-7 软配额)

输出:
  - 200 期 Top 4 / 6 / 10 / 20 命中率对比表
  - 4 区位实际配额分布
  - 推荐组合 (Top 4 表现最好的)
"""

import sys
import numpy as np
from pathlib import Path

ROOT = Path.home() / 'Library/Mobile Documents/com~apple~CloudDocs/PycharmProjects/Lottery'
sys.path.insert(0, str(ROOT))
from lottery_data import LotteryData
from kl8_v22_utils import (
    select_top_n_soft,
    score_v22_a6_smooth,
    compute_v22_a_baseline,
)

ld = LotteryData(ROOT)
df, conf = ld.load('快乐8')
red_cols = [f'红球{i}' for i in range(1, 21)]
data = df[red_cols].astype(int).values
n_periods = len(data)
print(f"数据: {n_periods} 期 (2020001 - {df['期号'].iloc[0]})\n")


# ============= 基础函数 =============
def c5_fn(last_i):
    c = np.zeros(80, dtype=int)
    for k in range(1, 6):
        if last_i + k < n_periods:
            for n in data[last_i + k]: c[n-1] += 1
    return c


def c10_fn(last_i):
    c = np.zeros(80, dtype=int)
    for k in range(1, 11):
        if last_i + k < n_periods:
            for n in data[last_i + k]: c[n-1] += 1
    return c


def c30_fn(last_i):
    c = np.zeros(80, dtype=int)
    for k in range(1, 31):
        if last_i + k < n_periods:
            for n in data[last_i + k]: c[n-1] += 1
    return c


def nbr_fn(last_i):
    nbr = np.zeros(80)
    if last_i < n_periods:
        for num in data[last_i]:
            for d in [-2, -1, 1, 2]:
                if 1 <= num+d <= 80:
                    nbr[num+d-1] += 0.3
    nbr /= 20
    return nbr


# ============= 预计算 baseline_c5 =============
print("预计算 baseline c5 分布...")
baseline_counter = {}
baseline_total = 0
for li in range(n_periods):
    if li + 5 < n_periods:
        c5 = c5_fn(li)
        for c in c5:
            baseline_counter[int(c)] = baseline_counter.get(int(c), 0) + 1
            baseline_total += 1
BASELINE_C5 = {k: baseline_counter.get(k, 0) / baseline_total for k in range(6)}
print(f"Baseline: {['%.1f%%' % (v*100) for v in BASELINE_C5.values()]}\n")


# ============= 选号函数 (4 种模式) =============
def select_strict_5555(scores, n):
    """严格 5/5/5/5 (V22.A 原版)"""
    sorted_idx = np.argsort(scores)[::-1]
    selected = []
    zc = [0, 0, 0, 0]
    for z in range(4):
        c = 0
        for idx in sorted_idx:
            if c >= 5: break
            num = int(idx) + 1
            if num in selected: continue
            if (num-1) // 20 != z: continue
            selected.append(num); zc[z] += 1; c += 1
    return sorted(selected, key=lambda x: scores[x-1], reverse=True)[:n]


def select_relaxed_38(scores, n):
    """3-8 软配额 (V22.A6 当前使用)"""
    return select_top_n_soft(scores, n=n, lo=3, hi=8)[0]


def select_relaxed_37(scores, n):
    """3-7 软配额 (P4 改进: Gemini 建议更紧)"""
    return select_top_n_soft(scores, n=n, lo=3, hi=7)[0]


# ============= 200 期回测 =============
print("="*70)
print("200 期回测: V22.A 基线 vs V22.A6 修复版 vs V22.A6 平滑版 vs +软配额")
print("="*70)

N_TEST = 200
START = n_periods - 201

versions = {
    'V0: V22.A 基线 (5/5/5/5)':           ('baseline', 'strict'),
    'V1: V22.A6 修复版 (3-8 配额)':       ('a6_fix',    'relaxed_38'),
    'V2: V22.A6 平滑 (3-8 配额)':         ('a6_smooth', 'relaxed_38'),
    'V3: V22.A6 平滑 + 3-7 软配额 ⭐':     ('a6_smooth', 'relaxed_37'),
}

hits = {v_name: {4: 0, 6: 0, 10: 0, 20: 0} for v_name in versions}
zone_dist = {v_name: {4: [], 10: [], 20: []} for v_name in versions}
extreme_count = 0

for i in range(START, n_periods - 1):
    last_i = i + 1
    actual = set(data[i].tolist())

    # 算 4 个版本的评分
    score_v0 = compute_v22_a_baseline(last_i, c5_fn, c10_fn, c30_fn, nbr_fn)

    # V22.A6 修复版 (P1 + lookback=3 + scale=2.0)
    score_v1 = score_v22_a6_smooth(
        last_i, data, n_periods,
        lookback=3, ema_halflife=1, scale=2.0,  # ema_halflife=1 等价于单期
        baseline_c5=BASELINE_C5,
        c5_fn=c5_fn, c10_fn=c10_fn, c30_fn=c30_fn, nbr_fn=nbr_fn
    )[0]

    # V22.A6 平滑版 (P3: lookback=10 + EMA 半衰=5)
    score_v2 = score_v22_a6_smooth(
        last_i, data, n_periods,
        lookback=10, ema_halflife=5, scale=2.0,
        baseline_c5=BASELINE_C5,
        c5_fn=c5_fn, c10_fn=c10_fn, c30_fn=c30_fn, nbr_fn=nbr_fn
    )[0]

    score_v3 = score_v2  # V3 = V2 评分, 只是选号不同

    score_map = {
        'V0: V22.A 基线 (5/5/5/5)':           score_v0,
        'V1: V22.A6 修复版 (3-8 配额)':       score_v1,
        'V2: V22.A6 平滑 (3-8 配额)':         score_v2,
        'V3: V22.A6 平滑 + 3-7 软配额 ⭐':     score_v3,
    }

    for v_name, (score_kind, select_kind) in versions.items():
        scores = score_map[v_name]
        if select_kind == 'strict':
            sel = select_strict_5555(scores, 20)
        elif select_kind == 'relaxed_38':
            sel = select_relaxed_38(scores, 20)
        elif select_kind == 'relaxed_37':
            sel = select_relaxed_37(scores, 20)
        # 截取 Top N
        for top_n in [4, 6, 10, 20]:
            top_set = set(sel[:top_n])
            hits[v_name][top_n] += len(top_set & actual)
            if top_n in [4, 10, 20]:
                # 4 区位分布
                zc = [0, 0, 0, 0]
                for n in top_set:
                    zc[(n - 1) // 20] += 1
                zone_dist[v_name][top_n].append(zc)


# 输出结果
print(f"\n{'TopN':<5}{'V0 基线':<14}{'V1 A6修':<14}{'V2 平滑':<14}{'V3 平滑+P4':<14}{'V3 vs V0'}")
print("-"*72)
for top_n in [4, 6, 10, 20]:
    row = f"{top_n:<5}"
    for v_name in versions:
        pct = hits[v_name][top_n] / (N_TEST * top_n) * 100
        row += f"{pct:>6.2f}%      "
    v0_pct = hits['V0: V22.A 基线 (5/5/5/5)'][top_n] / (N_TEST * top_n) * 100
    v3_pct = hits['V3: V22.A6 平滑 + 3-7 软配额 ⭐'][top_n] / (N_TEST * top_n) * 100
    row += f"{v3_pct - v0_pct:+.2f}pp"
    print(row)

print(f"\n基线理论: 25.00% (随机)")
print(f"统计期数: {N_TEST} 期 (START = {START})")


# ============= 4 区位分布对比 =============
print("\n" + "="*70)
print("4 区位实际配额分布 (Top 20 平均)")
print("="*70)
print(f"{'版本':<30}{'区1':<8}{'区2':<8}{'区3':<8}{'区4':<8}{'总'}")
print("-"*70)
for v_name in versions:
    z20 = np.mean(zone_dist[v_name][20], axis=0)
    print(f"{v_name:<30}{z20[0]:>5.2f}   {z20[1]:>5.2f}   {z20[2]:>5.2f}   {z20[3]:>5.2f}   {sum(z20):>5.1f}")


# ============= 推荐 =============
print("\n" + "="*70)
print("推荐 (按 Top 4 命中率从高到低)")
print("="*70)
ranked = sorted(versions.keys(), key=lambda v: -hits[v][4] / (N_TEST * 4))
for rank, v in enumerate(ranked, 1):
    pct4 = hits[v][4] / (N_TEST * 4) * 100
    pct20 = hits[v][20] / (N_TEST * 20) * 100
    print(f"#{rank}  {v}  Top4 {pct4:.2f}%  Top20 {pct20:.2f}%")