"""
KL8 2026203 期 V22.A2: V22.A1 + 20 胆放宽 (每区 3-8)
=============================================================

V4 → V5 演进 (2026-08-01 17:08):
    用户反馈: "20 胆 4 区位 5/5/5/5 太严格, 需要放松一点"

设计变更:
    - V4: 20 胆 4 区位 5/5/5/5 (严格)
    - **V5: 20 胆 4 区位 每区最少 3, 最多 8** (放宽) ⭐
    
实证基础 (200 期实际开奖 4 区位分布):
    - 平均: [4.92, 5.11, 5.04, 4.93] ≈ 5/5/5/5
    - **P5: [3, 3, 2, 2]** ← 各区可以少到 2-3 个
    - **P95: [7, 8, 8, 7]** ← 各区可以多到 7-8 个
    - **最大: [9, 10, 10, 10]** ← 1 区最多出 9 个

回测对比 (V4 方案 F + 不同 20 胆规则):
    | 规则 | Top 4 | Top 6 | Top 10 | Top 20 |
    |---|---|---|---|---|
    | 严格 5/5/5/5 | 52.2% | 51.1% | 50.4% | 46.3% |
    | **每区最少 3, 最多 8** ⭐ | **52.2%** | **51.7%** | **50.1%** | **48.7%** |
    | 纯 Top 20 (无限制) | 50.9% | 50.6% | 50.5% | 48.7% |

结论:
    - Top 4 命中率不变 (52.2%)
    - Top 6 提升 +0.6pp (51.7%)
    - Top 20 大幅提升 +2.4pp (48.7%)
"""
import sys
from pathlib import Path
import json
import numpy as np

ROOT = Path.home() / 'Library/Mobile Documents/com~apple~CloudDocs/PycharmProjects/Lottery'
sys.path.insert(0, str(ROOT))
sys.path.insert(0, '/Users/clarkkong/.openclaw/workspace/agents/lucky')
from lottery_data import LotteryData

ld = LotteryData(ROOT)
df, conf = ld.load('快乐8')
red_cols = [f'红球{i}' for i in range(1, 21)]
data = df[red_cols].astype(int).values
periods = df['期号'].tolist()
n_periods = len(data)

PRIOR = list(map(int, df[df['期号'] == '2026202'].iloc[0]['frontWinningNum'].split()))
TARGET = '2026203'
last_i = periods.index('2026202')

print(f"预测目标: {TARGET}")
print(f"上期 2026202: {PRIOR}")


# ============= 评分函数 V4 =============
def score_v22_v4(last_i, c2_w_hot=0.6, c3_w_hot=0.4):
    """V22.A + 方案 F 微调"""
    c5 = np.zeros(80, dtype=int)
    c10 = np.zeros(80, dtype=int)
    c30 = np.zeros(80, dtype=int)
    c2 = np.zeros(80, dtype=int)
    c3 = np.zeros(80, dtype=int)
    for k in range(1, 6):
        if last_i + k < n_periods:
            for n in data[last_i + k]: c5[n-1] += 1
    for k in range(1, 11):
        if last_i + k < n_periods:
            for n in data[last_i + k]: c10[n-1] += 1
    for k in range(1, 31):
        if last_i + k < n_periods:
            for n in data[last_i + k]: c30[n-1] += 1
    for k in range(1, 3):
        if last_i + k < n_periods:
            for n in data[last_i + k]: c2[n-1] += 1
    for k in range(1, 4):
        if last_i + k < n_periods:
            for n in data[last_i + k]: c3[n-1] += 1
    
    s = np.zeros(80)
    s[c5 <= 2] = 0.3
    s[(c5 >= 4) & (c5 <= 5)] = -0.8
    s[c5 >= 6] = -1.5
    
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
    
    nbr = np.zeros(80)
    if last_i < n_periods:
        for num in data[last_i]:
            for d in [-2, -1, 1, 2]:
                if 1 <= num+d <= 80:
                    nbr[num+d-1] += 0.3
    nbr /= 20
    
    c2_score = np.zeros(80)
    c2_score[c2 == 1] = c2_w_hot
    
    c3_score = np.zeros(80)
    c3_score[c3 == 1] = c3_w_hot
    
    return s + t + u + nbr + c2_score + c3_score


# ============= 20 胆选号 (V5 放宽: 每区 3-8) =============
def zone_of(num):
    if 1 <= num <= 20: return 0
    if 21 <= num <= 40: return 1
    if 41 <= num <= 60: return 2
    if 61 <= num <= 80: return 3
    return -1


def select_20_relaxed(scores, last_i, lo=3, hi=8):
    """V5: 20 胆 4 区位放宽 (每区最少 3, 最多 8)
    
    步骤:
        1. 每区先选 lo=3 个号 (保证每区都有)
        2. 按全局分数填到 20, 每区最多 hi=8 个
    """
    sorted_idx = np.argsort(scores)[::-1]
    selected = []
    zone_count = [0, 0, 0, 0]
    
    # 第一轮: 每区最少 lo 个
    for z in range(4):
        for idx in sorted_idx:
            if zone_count[z] >= lo: break
            num = int(idx) + 1
            if num in selected: continue
            if zone_of(num) != z: continue
            selected.append(num); zone_count[z] += 1
    
    # 第二轮: 按全局分数填到 20, 每区最多 hi 个
    if len(selected) < 20:
        for idx in sorted_idx:
            if len(selected) >= 20: break
            num = int(idx) + 1
            if num in selected: continue
            if zone_count[zone_of(num)] >= hi: continue
            selected.append(num); zone_count[zone_of(num)] += 1
    
    return selected, zone_count


# ============= 实战预测 =============
scores = score_v22_v4(last_i)
twenty, zones = select_20_relaxed(scores, last_i, lo=3, hi=8)

print(f"\n{'='*70}")
print(f"V5 实战预测 (2026203 期)")
print('='*70)
print(f"\n20 胆: {twenty}")
print(f"4 区位: {zones[0]}/{zones[1]}/{zones[2]}/{zones[3]} (总和 {sum(zones)})")

# 评分详情
sorted_20 = sorted(twenty, key=lambda n: scores[n-1], reverse=True)
print(f"\n20 胆评分详情 (降序):")
for i, n in enumerate(sorted_20, 1):
    z = zone_of(n) + 1
    print(f"  {i:2d}. {n:2d} (区 {z}): score={scores[n-1]:.3f}")

# N 胆
print(f"\n4 胆: {sorted_20[:4]}")
print(f"  4 区位: {[zone_of(n)+1 for n in sorted_20[:4]]}")
print(f"\n6 胆: {sorted_20[:6]}")
print(f"  4 区位: {[zone_of(n)+1 for n in sorted_20[:6]]}")
print(f"\n10 胆: {sorted_20[:10]}")
print(f"  4 区位: {[zone_of(n)+1 for n in sorted_20[:10]]}")

# V4 vs V5 对比
print(f"\n--- V4 (严格 5/5/5/5) vs V5 (每区 3-8) 对比 ---")
sorted_idx = np.argsort(scores)[::-1]
selected_v4 = []
zone_count = [0, 0, 0, 0]
for z in range(4):
    for idx in sorted_idx:
        if zone_count[z] >= 5: break
        num = int(idx) + 1
        if num in selected_v4: continue
        if zone_of(num) != z: continue
        selected_v4.append(num); zone_count[z] += 1

v4_top4 = sorted(selected_v4, key=lambda n: scores[n-1], reverse=True)[:4]
v5_top4 = sorted_20[:4]
print(f"V4 4 胆: {v4_top4}")
print(f"V5 4 胆: {v5_top4}")
print(f"V4 vs V5 4 胆差异: {sorted(set(v4_top4) ^ set(v5_top4)) or '完全相同 ✅'}")


# ============= 保存 JSON =============
output = {
    'meta': {
        'target_period': TARGET,
        'prior_period': '2026202',
        'prior_numbers': PRIOR,
        'created_at': '2026-08-01 17:15 GMT+8',
        'author': 'Lucky / MiniMax-M3',
        'method': 'V5: V22.A + 方案 F + 20 胆放宽 (每区 3-8)',
        'changes_from_v4': {
            '20_dan_constraint': 'V4 严格 5/5/5/5 → V5 每区最少 3 最多 8',
            'reason': '用户反馈 5/5/5/5 太严格; 实际开奖 P5 [3,3,2,2], P95 [7,8,8,7]',
            'backtest_impact': 'Top 4 不变 52.2%, Top 6 +0.6pp, Top 20 +2.4pp',
        },
        'scheme_F_params': {
            'c5_w_penalty': 0.0,
            'c2_w_hot': 0.6,
            'c3_w_hot': 0.4,
        },
        '20_dan_quota': {
            'min_per_zone': 3,
            'max_per_zone': 8,
        },
    },
    'predictions': {
        'V22_v5': {
            'method': 'V22.A + 方案 F + 20 胆放宽 (每区 3-8)',
            '20_dan': twenty,
            '4_dan': sorted_20[:4],
            '6_dan': sorted_20[:6],
            '10_dan': sorted_20[:10],
        }
    }
}

with open(ROOT / 'data/backtest/2026203_predictions_v22_a2.json', 'w', encoding='utf-8') as f:
    json.dump(output, f, ensure_ascii=False, indent=2)
print(f"\n✅ 已保存: {ROOT}/data/backtest/2026203_predictions_v22_a2.json")