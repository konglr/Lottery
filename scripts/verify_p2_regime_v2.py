"""
P2 验证 v2: 极端反转期检测器 — 修复版本
=====================================

改进时间: 2026-09-11 15:50 GMT+8
作者: Lucky / MiniMax-M3

修复:
  1. signals 缺少字段时给默认值 (不抛 KeyError)
  2. zone_sig 用绝对偏差平方和 (不除以 base_zone)
  3. 用最近 200 期 (而非数据前段)
  4. 输出更可读的 regime 详情
"""

import sys
import numpy as np
from pathlib import Path

ROOT = Path.home() / 'Library/Mobile Documents/com~apple~CloudDocs/PycharmProjects/Lottery'
sys.path.insert(0, str(ROOT))
from lottery_data import LotteryData
from kl8_v22_utils import detect_regime, compute_v22_a_baseline

ld = LotteryData(ROOT)
df, conf = ld.load('快乐8')
red_cols = [f'红球{i}' for i in range(1, 21)]
data = df[red_cols].astype(int).values
n_periods = len(data)
periods = df['期号'].tolist()
print(f"数据: {n_periods} 期 (2020001 - {periods[0]})\n")


# ============= 自定义更鲁棒的 detect_regime =============
def detect_regime_v2(last_i, sum_sigma=2.5, zone_threshold=0.05):
    """
    修复版: 更稳定的反转期检测
      1. 和值偏离: |近5期均值 - 近200期均值| / 近200期标准差 > sum_sigma
      2. 4 区位偏离: max |近5期占比 - 近200期占比| > zone_threshold (>=5pp)
      3. lag1 重号 > 10 (P99+)
    """
    if last_i + 5 >= n_periods or last_i + 200 >= n_periods:
        return 'normal', {'reason': 'insufficient_data'}

    # 信号 1: 近 5 期和值 vs 近 200 期均值
    recent_sums = [int(data[last_i + k].sum()) for k in range(1, 6)]
    base_sums = [int(data[last_i + 5 + k].sum()) for k in range(200)]
    recent_mean = np.mean(recent_sums)
    base_mean = np.mean(base_sums)
    base_std = np.std(base_sums) + 1e-9
    sum_signal = abs(recent_mean - base_mean) / base_std

    # 信号 2: 4 区位偏离 (max |占比差|)
    recent_zone = np.zeros(4)
    for k in range(1, 6):
        for n in data[last_i + k]:
            recent_zone[(n - 1) // 20] += 1
    recent_zone_pct = recent_zone / 100  # 5期×20号=100, 直接得百分比

    base_zone = np.zeros(4)
    for k in range(5, 205):
        for n in data[last_i + k]:
            base_zone[(n - 1) // 20] += 1
    base_zone_pct = base_zone / 4000  # 200期×20号=4000

    zone_signal = float(np.max(np.abs(recent_zone_pct - base_zone_pct)))

    # 信号 3: lag1 重号
    lag1_overlap = len(set(data[last_i].tolist()) & set(data[last_i + 1].tolist()))

    signals = {
        'sum_signal': sum_signal,
        'zone_signal': zone_signal,
        'lag1_overlap': lag1_overlap,
        'recent_sums': recent_sums,
        'recent_mean': recent_mean,
        'base_mean': base_mean,
        'base_std': base_std,
        'recent_zone_pct': recent_zone_pct.tolist(),
        'base_zone_pct': base_zone_pct.tolist(),
    }

    is_extreme = (
        sum_signal > sum_sigma or
        zone_signal > zone_threshold or
        lag1_overlap > 10
    )

    return ('extreme' if is_extreme else 'normal'), signals


# ============= 基础评分函数 =============
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


# ============= 用最近 200 期 (期号最新) =============
print("="*70)
print("P2 v2 验证: detect_regime_v2 (修复版)")
print("="*70)

# 修复: START 选数据最近端 (期号最新)
# 数据降序, n_periods-1 是最早, 0 是最新
# 想用最近 200 期 = 索引 0..199 (期号最新)
# 但每期需要看过去 200 期, last_i = i+1, last_i+200 ≤ n_periods
# 所以 START ≥ 0 (期号最新), END ≤ n_periods - 201
N_TEST = 200
START = 0   # 包含最新期
END = START + N_TEST  # 测试期数 200

# 但是 last_i = i+1 + 200 需要 < n_periods, 所以 END ≤ n_periods - 201
END = min(END, n_periods - 201)
N_TEST = END - START

print(f"START={START}, END={END}, N_TEST={N_TEST}")
print(f"对应期号: {periods[START]} (最新) → {periods[END-1]} (测试终点)\n")


# 收集数据
extreme_records = []
normal_records = []

for i in range(START, END):
    last_i = i + 1
    actual = set(data[i].tolist())

    regime, signals = detect_regime_v2(last_i)

    score_v0 = compute_v22_a_baseline(last_i, c5_fn, c10_fn, c30_fn, nbr_fn)
    top4_v0 = set(np.argsort(score_v0)[::-1][:4] + 1)
    hits_v0 = len(top4_v0 & actual)

    rec = {
        'i': i,
        'period': periods[i],
        'regime': regime,
        'signals': signals,
        'hits_v0': hits_v0,
    }
    if regime == 'extreme':
        extreme_records.append(rec)
    else:
        normal_records.append(rec)


# ============= 总体统计 =============
print(f"{'='*70}")
print(f"总体: {len(extreme_records)} extreme / {len(normal_records)} normal / {N_TEST} 总")
print("="*70)

print(f"\n{'模型':<20}{'Normal 期':<25}{'Extreme 期':<25}{'差异'}")
print("-"*80)

normal_hits = sum(r['hits_v0'] for r in normal_records)
extreme_hits = sum(r['hits_v0'] for r in extreme_records)
n_total_normal = len(normal_records) * 4
n_total_extreme = len(extreme_records) * 4

if n_total_normal > 0 and n_total_extreme > 0:
    normal_pct = normal_hits / n_total_normal * 100
    extreme_pct = extreme_hits / n_total_extreme * 100
    diff = extreme_pct - normal_pct
    flag = '⚠️ 反转期更差' if diff < -1 else ('✅ 有效' if diff < -0.5 else '≈ 无差')
    print(f"{'V22.A 基线 Top 4':<20}{normal_pct:>6.2f}% ({len(normal_records)}期)   {extreme_pct:>6.2f}% ({len(extreme_records)}期)   {diff:+.2f}pp  {flag}")
elif n_total_extreme == 0:
    print(f"V22.A 基线 Top 4: 没有 extreme 期 (阈值可能过严)")


# ============= 显示所有 extreme 期 =============
print(f"\n{'='*70}")
print("所有标记为 extreme 的期:")
print("="*70)
print(f"{'期号':<10}{'sum_sig':<10}{'zone_sig':<10}{'lag1':<6}{'和值近5':<12}{'区1/2/3/4':<20}{'V0命中'}")
print("-"*85)
for r in extreme_records:
    sig = r['signals']
    rz = sig.get('recent_zone_pct', [0,0,0,0])
    rs = sig.get('recent_sums', [])
    rs_str = ','.join(str(s) for s in rs[:5])
    rz_str = '/'.join(f'{p*100:.1f}' for p in rz)
    print(f"{r['period']:<10}{sig.get('sum_signal',0):<10.2f}{sig.get('zone_signal',0):<10.3f}{sig.get('lag1_overlap',0):<6}{rs_str:<12}{rz_str:<20}{r['hits_v0']}")


# ============= 实战建议 =============
print(f"\n{'='*70}")
print("实战策略评估")
print("="*70)

print("\n策略 A: 全期用 V22.A 基线 (现状)")
all_hits = sum(r['hits_v0'] for r in normal_records) + sum(r['hits_v0'] for r in extreme_records)
all_pct = all_hits / (N_TEST * 4) * 100
print(f"  Top 4 命中率: {all_pct:.2f}%")

print("\n策略 B: 反转期不下注 (假设 0% 命中)")
if n_total_normal > 0:
    skip_pct = normal_hits / (N_TEST * 4) * 100
    print(f"  命中率: {skip_pct:.2f}% (vs 策略 A {all_pct:+.2f}pp)")
    print(f"  收益评估: {'✅ 更好' if skip_pct > all_pct else '❌ 一样/更差'}")

print("\n策略 C: 反转期切到 V22.A 基线 (本来就是基线), normal 期保持")
print("  实质同策略 A, 无变化")


# ============= 阈值敏感性测试 =============
print(f"\n{'='*70}")
print("阈值敏感性测试 (找最优阈值)")
print("="*70)

# 尝试不同阈值,看极端期的命中率
print(f"{'sum_σ':<8}{'zone_thr':<10}{'extreme期数':<14}{'extreme期Top4%':<18}{'收益差 (B-A)'}")
print("-"*70)

for sum_sigma in [1.5, 2.0, 2.5, 3.0]:
    for zone_thr in [0.03, 0.05, 0.08]:
        cnt = 0
        ex_hits = 0
        ex_n = 0
        for i in range(START, END):
            last_i = i + 1
            actual = set(data[i].tolist())
            regime, signals = detect_regime_v2(last_i, sum_sigma=sum_sigma, zone_threshold=zone_thr)
            score_v0 = compute_v22_a_baseline(last_i, c5_fn, c10_fn, c30_fn, nbr_fn)
            top4 = set(np.argsort(score_v0)[::-1][:4] + 1)
            hits = len(top4 & actual)
            if regime == 'extreme':
                cnt += 1
                ex_hits += hits
                ex_n += 4

        if ex_n > 0:
            ex_pct = ex_hits / ex_n * 100
        else:
            ex_pct = 0

        # 策略 B (反转期不下注) 的命中率
        normal_total = N_TEST - cnt
        skip_pct = (all_hits - ex_hits) / (N_TEST * 4) * 100
        diff = skip_pct - all_pct
        print(f"{sum_sigma:<8}{zone_thr:<10}{cnt:<14}{ex_pct:<18.2f}{diff:+.2f}pp")