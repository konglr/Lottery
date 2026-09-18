"""
P2 验证: 极端反转期检测器 — 能否识别 "全模型挂掉" 的期
==============================================

改进时间: 2026-09-11 15:30 GMT+8
作者: Lucky / MiniMax-M3
依据: Gemini 评估报告 — "状态识别 + 动态风控"

核心问题:
  反转期(2026229 → 2026230)让所有模型 Top 4 命中率暴跌。
  如果能提前识别, 就能在反转期切到保守策略或不下注。

验证方法:
  1. 用 detect_regime 标出最近 200 期里的 "extreme" 期
  2. 统计 extreme 期 vs normal 期:
     - V22.A 基线 Top 4 命中率差异
     - V22.A6 修复版 Top 4 命中率差异
  3. 如果 extreme 期命中率显著低于 normal 期 → 检测器有效
"""

import sys
import numpy as np
from pathlib import Path
from collections import defaultdict

ROOT = Path.home() / 'Library/Mobile Documents/com~apple~CloudDocs/PycharmProjects/Lottery'
sys.path.insert(0, str(ROOT))
from lottery_data import LotteryData
from kl8_v22_utils import (
    detect_regime,
    compute_v22_a_baseline,
    score_v22_a6_smooth,
    select_top_n_soft,
)

ld = LotteryData(ROOT)
df, conf = ld.load('快乐8')
red_cols = [f'红球{i}' for i in range(1, 21)]
data = df[red_cols].astype(int).values
n_periods = len(data)
print(f"数据: {n_periods} 期\n")


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


# Baseline
baseline_counter = {}
baseline_total = 0
for li in range(n_periods):
    if li + 5 < n_periods:
        c5 = c5_fn(li)
        for c in c5:
            baseline_counter[int(c)] = baseline_counter.get(int(c), 0) + 1
            baseline_total += 1
BASELINE_C5 = {k: baseline_counter.get(k, 0) / baseline_total for k in range(6)}


# ============= 200 期回测 + regime 标记 =============
print("="*70)
print("P2 验证: detect_regime 在 200 期里能否识别 '全模型挂掉' 的期")
print("="*70)

N_TEST = 200
START = n_periods - 201

# 收集每期的 regime 标记 + 两个模型的命中数
regime_records = []

for i in range(START, n_periods - 1):
    last_i = i + 1
    actual = set(data[i].tolist())

    # Regime 检测 (用 last_i = i+1 的视角)
    regime, signals = detect_regime(last_i, data, n_periods)

    # V22.A 基线 Top 4 命中
    score_v0 = compute_v22_a_baseline(last_i, c5_fn, c10_fn, c30_fn, nbr_fn)
    top4_v0 = set(np.argsort(score_v0)[::-1][:4] + 1)
    hits_v0 = len(top4_v0 & actual)

    # V22.A6 修复版 Top 4 命中
    score_a6 = score_v22_a6_smooth(
        last_i, data, n_periods,
        lookback=3, ema_halflife=1, scale=2.0,
        baseline_c5=BASELINE_C5,
        c5_fn=c5_fn, c10_fn=c10_fn, c30_fn=c30_fn, nbr_fn=nbr_fn
    )[0]
    top4_a6 = set(np.argsort(score_a6)[::-1][:4] + 1)
    hits_a6 = len(top4_a6 & actual)

    regime_records.append({
        'i': i,
        'regime': regime,
        'signals': signals,
        'hits_v0': hits_v0,
        'hits_a6': hits_a6,
        'actual_count': 20,
    })


# ============= 统计 =============
extreme = [r for r in regime_records if r['regime'] == 'extreme']
normal = [r for r in regime_records if r['regime'] == 'normal']

print(f"\n200 期里:")
print(f"  Extreme 期数: {len(extreme)} ({len(extreme)/N_TEST*100:.1f}%)")
print(f"  Normal 期数: {len(normal)} ({len(normal)/N_TEST*100:.1f}%)")

# Top 4 命中率对比
print(f"\n{'模型':<25}{'Normal 期 (Top 4 命中%)':<25}{'Extreme 期 (Top 4 命中%)':<25}{'差异'}")
print("-"*85)

for model_name, hits_key in [('V22.A 基线', 'hits_v0'), ('V22.A6 修复版', 'hits_a6')]:
    normal_hits = sum(r[hits_key] for r in normal)
    extreme_hits = sum(r[hits_key] for r in extreme)
    n_total = len(normal) * 4
    e_total = len(extreme) * 4
    normal_pct = normal_hits / n_total * 100 if n_total > 0 else 0
    extreme_pct = extreme_hits / e_total * 100 if e_total > 0 else 0
    diff = extreme_pct - normal_pct
    flag = '⚠️ 反转期更差' if diff < -1 else ('✅ 检测有效' if diff < -0.5 else '无差异')
    print(f"{model_name:<25}{normal_pct:>6.2f}% ({len(normal)} 期)         {extreme_pct:>6.2f}% ({len(extreme)} 期)         {diff:+.2f}pp  {flag}")


# ============= 最近 20 期 regime 标记 =============
print(f"\n{'='*70}")
print("最近 20 期 regime 标记详情")
print("="*70)
print(f"{'期号':<10}{'regime':<10}{'sum_sig':<10}{'zone_sig':<10}{'lag1重':<8}{'V0命中':<8}{'A6命中'}")
print("-"*70)
for r in regime_records[-20:]:
    period = df['期号'].iloc[r['i']]
    sig = r['signals']
    print(f"{period:<10}{r['regime']:<10}{sig['sum_signal']:<10.2f}{sig['zone_signal']:<10.2f}{sig['lag1_overlap']:<8}{r['hits_v0']:<8}{r['hits_a6']}")


# ============= 已知反转期验证 =============
print(f"\n{'='*70}")
print("已知反转期验证 (2026229 → 2026230)")
print("="*70)

# 找 2026228 / 2026229 / 2026230 期对应的 i
target_periods = ['2026228', '2026229', '2026230']
for tp in target_periods:
    matches = df.index[df['期号'] == tp].tolist()
    if matches:
        idx = matches[0]
        if idx >= START and idx < n_periods - 1:
            r = regime_records[idx - START]
            sig = r['signals']
            print(f"{tp}: regime={r['regime']}, sum_sig={sig['sum_signal']:.2f}, zone_sig={sig['zone_signal']:.2f}, lag1={sig['lag1_overlap']}, V0命中={r['hits_v0']}/4, A6命中={r['hits_a6']}/4")


# ============= 实用建议 =============
print(f"\n{'='*70}")
print("实用建议")
print("="*70)

if extreme and normal:
    # 在反转期是否切到保守策略?
    print("\n如果反转期不下注 (假设保守策略 0% 命中):")
    print(f"  200 期总命中率 = (Normal 命中 / (200 * 4))")
    normal_v0 = sum(r['hits_v0'] for r in normal)
    new_pct = normal_v0 / (N_TEST * 4) * 100
    print(f"  V22.A 基线: {new_pct:.2f}% (vs 当前 27.25%, {new_pct - 27.25:+.2f}pp)")

    # 反过来: 反转期是否切到 V22.A6 (虽然它整体弱)?
    print("\n如果反转期切到 V22.A 基线 (原本就是基线), normal 期保持 V22.A6:")
    new_hits = sum(r['hits_a6'] for r in normal) + sum(r['hits_v0'] for r in extreme)
    new_pct = new_hits / (N_TEST * 4) * 100
    print(f"  混合策略: {new_pct:.2f}% (vs 纯 V22.A 27.25%, {new_pct - 27.25:+.2f}pp)")