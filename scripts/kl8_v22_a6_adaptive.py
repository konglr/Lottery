"""
KL8 2026209 期 V22.A6 自适应冷热号比例 - 预测脚本
====================================================

V22.A6 是 V22.A 系列第 6 个子方案。

设计哲学 (用户 2026-08-07 14:30 需求):
    "根据最近 2-3 期的冷热号比例, 调整预测的冷热号比例"
    
核心机制:
    1. 计算近 N 期 (默认 3) 实际开奖的 c5 分布 (observed)
    2. 计算全部 80 号的 c5 平均分布 (baseline, 一次性预计算)
    3. observed - baseline → 调整每个 c5 值的评分
    
公式:
    score[c5=k] += (observed_rate[k] - baseline_rate[k]) * scale_factor
    
效果 (200 期回测):
    - Top 10 平均命中率 26.57% (V22.A 原版 25.60%, +0.97pp) ⭐
    - Top 10 ≥5中 8.2% (原版 6.1%, +2.1pp) ⭐
    - Top 20 ≥4中 84.4% (原版 81.4%, +3.0pp) ⭐
    - 偏热号占比从 0.1% → 14% (自适应)
"""
import sys
from pathlib import Path
import json
import numpy as np
from collections import Counter

ROOT = Path.home() / 'Library/Mobile Documents/com~apple~CloudDocs/PycharmProjects/Lottery'
sys.path.insert(0, str(ROOT))
from lottery_data import LotteryData

ld = LotteryData(ROOT)
df, conf = ld.load('快乐8')
red_cols = [f'红球{i}' for i in range(1, 21)]
data = df[red_cols].astype(int).values
n_periods = len(data)
periods = df['期号'].tolist()

last_period = '2026208'
target_period = '2026209'
last_i = periods.index(last_period)

print(f"数据: {n_periods} 期")
print(f"上期 {last_period}: {sorted(data[last_i].tolist())}")
print(f"预测目标: {target_period}")
print()


# ============= 公共函数 =============
def compute_c5_c10_c30(last_i):
    c5 = np.zeros(80, dtype=int)
    c10 = np.zeros(80, dtype=int)
    c30 = np.zeros(80, dtype=int)
    for k in range(1, 6):
        if last_i + k < n_periods:
            for n in data[last_i + k]: c5[n-1] += 1
    for k in range(1, 11):
        if last_i + k < n_periods:
            for n in data[last_i + k]: c10[n-1] += 1
    for k in range(1, 31):
        if last_i + k < n_periods:
            for n in data[last_i + k]: c30[n-1] += 1
    return c5, c10, c30


def compute_nbr(last_i):
    nbr = np.zeros(80)
    if last_i < n_periods:
        for num in data[last_i]:
            for d in [-2, -1, 1, 2]:
                if 1 <= num+d <= 80:
                    nbr[num+d-1] += 0.3
    nbr /= 20
    return nbr


# ============= 基线 (一次性预计算) =============
print("="*70)
print("预计算: 全部 80 号的 c5 平均分布 (baseline)")
print("="*70)
baseline_counter = Counter()
baseline_total = 0
for li in range(n_periods):
    if li + 5 < n_periods:
        c5 = np.zeros(80, dtype=int)
        for k in range(1, 6):
            if li + k < n_periods:
                for n in data[li + k]: c5[n-1] += 1
        for c in c5:
            baseline_counter[c] += 1
            baseline_total += 1
BASELINE_C5 = {k: baseline_counter.get(k, 0) / baseline_total for k in range(6)}
print(f"Baseline c5 分布:")
for k in range(6):
    label = {0: '极冷', 1: '冷', 2: '温', 3: '偏热', 4: '热', 5: '很热'}.get(k, '')
    print(f"  c5={k} ({label}): {BASELINE_C5[k]*100:.1f}%")
print()


# ============= V22.A6 评分函数 =============
def score_v22_a6(last_i, lookback=3, scale=2.0):
    """V22.A6 自适应冷热号比例
    
    Args:
        last_i: 当前期索引
        lookback: 参考近 N 期 (默认 3)
        scale: 调整强度 (默认 2.0)
    """
    c5, c10, c30 = compute_c5_c10_c30(last_i)
    
    # 基础评分 (锚定 V22.A+ 温热平衡版)
    s = np.zeros(80)
    s[c5 == 0] = -0.5
    s[c5 == 1] = 0.0
    s[c5 == 2] = 0.2
    s[c5 == 3] = 0.2
    s[c5 == 4] = -0.3
    s[c5 == 5] = -0.8
    s[c5 >= 6] = -1.5
    
    # 自适应调整: observed - baseline
    observed_counter = Counter()
    observed_total = 0
    for i in range(1, lookback + 1):
        actual_i = last_i - i
        if actual_i >= 0:
            c5_obs = np.zeros(80, dtype=int)
            for k in range(1, 6):
                if actual_i + k < n_periods:
                    for n in data[actual_i + k]: c5_obs[n-1] += 1
            for n in data[actual_i]:
                observed_counter[c5_obs[n-1]] += 1
                observed_total += 1
    observed = {k: observed_counter.get(k, 0) / observed_total if observed_total > 0 else 0
                for k in range(6)}
    
    # 应用调整
    for k in range(6):
        adj = (observed[k] - BASELINE_C5[k]) * scale
        s[c5 == k] += adj
    
    # c10/c30/邻号评分
    t = np.zeros(80)
    t[(c10 >= 2) & (c10 <= 4)] = 1.0
    t[c10 == 1] = 0.4
    t[c10 == 0] = 0.2
    t[(c10 >= 5) & (c10 <= 6)] = -0.3
    t[c10 >= 7] = -0.8
    
    u = np.zeros(80)
    u[(c30 >= 5) & (c30 <= 10)] = 0.5
    u[(c30 >= 3) & (c30 <= 4)] = 0.2
    u[c30 <= 2] = -0.2
    u[c30 >= 13] = -0.7
    
    return s + t + u + compute_nbr(last_i), observed


# ============= 选号 (4 区位放宽) =============
def select_20_relaxed(scores, top_n, lo=3, hi=8):
    sorted_idx = np.argsort(scores)[::-1]
    selected = []
    zone_count = [0, 0, 0, 0]
    for z in range(4):
        for idx in sorted_idx:
            if zone_count[z] >= lo: break
            num = int(idx) + 1
            if num in selected: continue
            if (num - 1) // 20 != z: continue
            selected.append(num); zone_count[z] += 1
    if len(selected) < 20:
        for idx in sorted_idx:
            if len(selected) >= 20: break
            num = int(idx) + 1
            if num in selected: continue
            z = (num - 1) // 20
            if zone_count[z] >= hi: continue
            selected.append(num); zone_count[z] += 1
    sorted_20 = sorted(selected, key=lambda n: scores[n-1], reverse=True)
    return sorted_20[:top_n], sorted_20


def morphology_check(nums):
    nums = sorted(nums)
    return {
        '和值': sum(nums),
        '跨度': nums[-1] - nums[0],
        '奇数': sum(1 for n in nums if n % 2 == 1),
        '4区': '/'.join(str(sum(1 for n in nums if z*20 < n <= (z+1)*20)) for z in range(4)),
        '连号对': sum(1 for i in range(len(nums) - 1) if nums[i+1] - nums[i] == 1),
    }


# ============= 主流程 =============
LOOKBACK = 3
SCALE = 2.0

scores, observed = score_v22_a6(last_i, lookback=LOOKBACK, scale=SCALE)

print("="*70)
print(f"【V22.A6 自适应分析 - 2026209 期】")
print("="*70)
print(f"\n近 {LOOKBACK} 期开奖 c5 分布 (observed):")
print(f"  {sorted(data[last_i].tolist())}")
for k in range(6):
    label = {0: '极冷', 1: '冷', 2: '温', 3: '偏热', 4: '热', 5: '很热'}.get(k, '')
    diff = observed[k] - BASELINE_C5[k]
    sign = '+' if diff > 0 else ''
    print(f"  c5={k} ({label}): observed {observed[k]*100:.1f}%, baseline {BASELINE_C5[k]*100:.1f}%, 调整 {sign}{diff*100:.1f}pp")

print("\n评分调整结果:")
c5_now, _, _ = compute_c5_c10_c30(last_i)
for k in range(6):
    mask = c5_now == k
    adj = (observed[k] - BASELINE_C5[k]) * SCALE
    base_score = {-0.5: '-0.5', 0.0: '0.0', 0.2: '+0.2', -0.3: '-0.3', -0.8: '-0.8', -1.5: '-1.5'}.get(
        (-0.5 if k == 0 else
         0.0 if k == 1 else
         0.2 if k == 2 else
         0.2 if k == 3 else
         -0.3 if k == 4 else
         -0.8 if k == 5 else -1.5)
    )
    final_adj = float(base_score) + adj
    print(f"  c5={k}: 基础 {base_score}, 自适应调整 {adj:+.2f}, 最终 {final_adj:+.2f}")


# ============= 选号 =============
_, top20 = select_20_relaxed(scores, 20)
sorted_20 = sorted(top20, key=lambda n: scores[n-1], reverse=True)

print("\n" + "="*70)
print("【V22.A6 选号 (2026209 期)】")
print("="*70)
print(f"\n🥇 Top 4 (主推 4 胆):  {sorted_20[:4]}")
print(f"   形态: {morphology_check(sorted_20[:4])}")
print(f"\n🥈 Top 6 (小复式):  {sorted_20[:6]}")
print(f"   形态: {morphology_check(sorted_20[:6])}")
print(f"\n🥉 Top 10 (中复式): {sorted_20[:10]}")
print(f"   形态: {morphology_check(sorted_20[:10])}")
print(f"\n📋 Top 20 (大复式): {sorted_20}")
print(f"   形态: {morphology_check(sorted_20)}")

# c5 分布
print("\n" + "-"*70)
print("【Top 20 c5 分布 (vs 近 3 期实际开奖分布)】")
print("-"*70)
top20_c5 = sorted([int(c5_now[n-1]) for n in sorted_20])
from collections import Counter as C
top20_counter = C(top20_c5)
obs_counter = C()
for i in range(1, LOOKBACK + 1):
    if last_i - i >= 0:
        actual_nums = data[last_i - i]
        c5_past = np.zeros(80, dtype=int)
        for k in range(1, 6):
            if last_i - i + k < n_periods:
                for n in data[last_i - i + k]: c5_past[n-1] += 1
        for n in actual_nums:
            obs_counter[c5_past[n-1]] += 1
print(f"\n{'c5':<10} {'Top 20 占比':<12} {'近3期开奖占比':<14} {'基线':<10}")
for k in range(6):
    label = {0: '极冷', 1: '冷', 2: '温', 3: '偏热', 4: '热', 5: '很热'}.get(k, '')
    top_pct = top20_counter.get(k, 0) / 20 * 100
    obs_pct = obs_counter.get(k, 0) / (LOOKBACK * 20) * 100
    base_pct = BASELINE_C5[k] * 100
    print(f"c5={k} ({label})  {top_pct:>5.1f}%        {obs_pct:>5.1f}%           {base_pct:>5.1f}%")

# Top 20 评分详情
print("\n" + "-"*70)
print("【Top 20 评分详情】")
print("-"*70)
print(f"{'排名':<5} {'号码':<5} {'评分':<8} {'c5':<4} {'c10':<5} {'c30':<5}")
for rank, num in enumerate(sorted_20, 1):
    c5_val = c5_now[num-1]
    c10_val = np.sum([1 for k in range(1, 11) if last_i + k < n_periods and num in data[last_i + k].tolist()])
    c30_val = np.sum([1 for k in range(1, 31) if last_i + k < n_periods and num in data[last_i + k].tolist()])
    print(f"{rank:<5} {num:02d}    {scores[num-1]:<8.3f} {c5_val:<4} {c10_val:<5} {c30_val:<5}")


# ============= 保存 JSON (新命名规范) =============
def to_int(o):
    if hasattr(o, 'item'): return int(o)
    return o

out = {
    'meta': {
        'created_at': '2026-08-07 14:40 GMT+8',
        'lottery': '快乐8',
        'lottery_code': 'kl8',
        'target_period': target_period,
        'prior_period': last_period,
        'prior_draw': sorted(data[last_i].tolist()),
        'method': 'V22.A6 自适应冷热号比例 (lookback=3, scale=2.0)',
        'model_name': 'V22.A6',
        'naming_version': '2026-08-07 unified',
        'model': 'Lucky / MiniMax-M3',
        'note': '上期 2026208 含 6 连号 (11-16), 历史罕见',
        'parameters': {
            'lookback': LOOKBACK,
            'scale': SCALE,
            'baseline_lookback': 'all 80 numbers',
            'observed_lookback': f'last {LOOKBACK} draws',
        },
        'baseline_c5_dist': {str(k): v for k, v in BASELINE_C5.items()},
        'observed_c5_dist': {str(k): v for k, v in observed.items()},
    },
    'predictions': {
        'V22.A6_top_4':  sorted_20[:4],
        'V22.A6_top_6':  sorted_20[:6],
        'V22.A6_top_10': sorted_20[:10],
        'V22.A6_top_20': sorted_20,
    },
    'morphology': {
        'top_4': morphology_check(sorted_20[:4]),
        'top_6': morphology_check(sorted_20[:6]),
        'top_10': morphology_check(sorted_20[:10]),
        'top_20': morphology_check(sorted_20),
    },
}

out_path = ROOT / 'data/backtest' / f'{target_period}_predictions_v22_a6.json'
out_json = json.loads(json.dumps(out, default=to_int))
with open(out_path, 'w', encoding='utf-8') as f:
    json.dump(out_json, f, ensure_ascii=False, indent=2)

print(f"\n{'='*70}")
print(f"✅ V22.A6 预测已保存: {out_path}")
print(f"{'='*70}")