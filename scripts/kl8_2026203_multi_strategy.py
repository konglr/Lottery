"""
KL8 2026203 期 多策略预测
=============================================================

4 个独立策略, 每个都预测 4胆/6胆/10胆/20胆:
1. V22.A (P90 约束 + 4 区位配额) - 当前最强
2. V22.E (c5 主权重 + c10/c30 减权) - 平衡
3. V22.F (偏热号策略) - 偏 c5=2 温号
4. V18.C (反向基线) - 冷号优先

输入: 2026202 期实际开奖号
输出: 4 个策略的 4胆/6胆/10胆/20胆 + 共识分析
"""
import sys
from pathlib import Path
import json
from datetime import datetime

ROOT = Path.home() / 'Library/Mobile Documents/com~apple~CloudDocs/PycharmProjects/Lottery'
sys.path.insert(0, str(ROOT))
sys.path.insert(0, '/Users/clarkkong/.openclaw/workspace/agents/lucky')

from lottery_data import LotteryData
import numpy as np

ld = LotteryData(ROOT)
df, conf = ld.load('快乐8')
red_cols = [f'红球{i}' for i in range(1, 21)]
data = df[red_cols].astype(int).values
periods = df['期号'].tolist()

# 输入: 2026202 期是最新期 (索引 0)
PRIOR = list(map(int, df[df['期号'] == '2026202'].iloc[0]['frontWinningNum'].split()))
TARGET = '2026203'
print(f"预测目标: {TARGET}")
print(f"上期 2026202 开奖: {PRIOR}")
print(f"数据总期数: {len(data)} (倒序, 索引 0 = 2026202)")


# ============= 通用工具 =============
def zone_of(num):
    if 1 <= num <= 20: return 0
    if 21 <= num <= 40: return 1
    if 41 <= num <= 60: return 2
    if 61 <= num <= 80: return 3
    return -1


def max_consecutive_in(nums):
    if len(nums) < 2: return 1
    s = sorted(nums)
    max_run = 1
    current = 1
    for i in range(1, len(s)):
        if s[i] == s[i-1] + 1:
            current += 1
            max_run = max(max_run, current)
        else:
            current = 1
    return max_run


def zone_deviation(nums):
    if not nums: return 0
    n = len(nums)
    zones = [0, 0, 0, 0]
    for num in nums:
        zones[zone_of(num)] += 1
    return max([abs(z/n - 0.25) for z in zones])


def get_quota(top_n):
    if top_n == 4: return [1, 1, 1, 1]
    if top_n == 6: return [2, 1, 2, 1]
    if top_n == 10: return [3, 3, 2, 2]
    if top_n == 20: return [5, 5, 5, 5]
    return [(top_n + 3) // 4] * 4


# ============= 评分函数 (4 个策略) =============
def score_v22_v21(last_i):
    """V22.A = V20 + 配额 + 形态约束 + 邻号加权"""
    if last_i + 30 >= len(data):
        return np.zeros(80)
    counts = {}
    for w in [5, 10, 30]:
        c = np.zeros(80, dtype=int)
        for k in range(1, w + 1):
            if last_i + k < len(data):
                for n in data[last_i + k]:
                    c[n-1] += 1
        counts[w] = c
    c5, c10, c30 = counts[5], counts[10], counts[30]
    
    c5_score = np.zeros(80)
    c5_score[c5 <= 2] = 0.3
    c5_score[(c5 >= 4) & (c5 <= 5)] = -0.8
    c5_score[c5 >= 6] = -1.5
    
    c10_score = np.zeros(80)
    c10_score[(c10 >= 2) & (c10 <= 4)] = 1.0
    c10_score[c10 == 1] = 0.4
    c10_score[c10 == 0] = 0.2
    c10_score[(c10 >= 5) & (c10 <= 6)] = -0.3
    c10_score[c10 >= 7] = -0.8
    
    c30_score = np.zeros(80)
    c30_score[(c30 >= 5) & (c30 <= 10)] = 0.5
    c30_score[(c30 >= 3) & (c30 <= 4)] = 0.2
    c30_score[c30 <= 2] = -0.2
    c30_score[c30 >= 13] = -0.7
    
    nbr_score = np.zeros(80)
    if last_i < len(data):
        for num in data[last_i]:
            for d in [-2, -1, 1, 2]:
                if 1 <= num+d <= 80:
                    nbr_score[num+d-1] += 0.3
    nbr_score /= 20
    
    return c5_score + c10_score + c30_score + nbr_score


def score_v22_v25(last_i):
    """V22.E: c5 主权重 + c10/c30 减权 (0.3 倍)
    c5=0/1: +0.5, c5=2: +0.6, c5=3: -0.5, c5=4-5: -1.5, c5≥6: -3.0
    """
    if last_i + 30 >= len(data):
        return np.zeros(80)
    counts = {}
    for w in [5, 10, 30]:
        c = np.zeros(80, dtype=int)
        for k in range(1, w + 1):
            if last_i + k < len(data):
                for n in data[last_i + k]:
                    c[n-1] += 1
        counts[w] = c
    c5, c10, c30 = counts[5], counts[10], counts[30]
    
    # v2.5 c5 主权重
    c5_score = np.zeros(80)
    c5_score[c5 == 0] = 0.5
    c5_score[c5 == 1] = 0.5
    c5_score[c5 == 2] = 0.6
    c5_score[c5 == 3] = -0.5
    c5_score[(c5 >= 4) & (c5 <= 5)] = -1.5
    c5_score[c5 >= 6] = -3.0
    
    # c10 减权 0.3 倍
    c10_score = np.zeros(80)
    c10_score[(c10 >= 2) & (c10 <= 4)] = 1.0 * 0.3
    c10_score[c10 == 1] = 0.4 * 0.3
    c10_score[c10 == 0] = 0.2 * 0.3
    c10_score[(c10 >= 5) & (c10 <= 6)] = -0.3 * 0.3
    c10_score[c10 >= 7] = -0.8 * 0.3
    
    # c30 减权 0.3 倍
    c30_score = np.zeros(80)
    c30_score[(c30 >= 5) & (c30 <= 10)] = 0.5 * 0.3
    c30_score[(c30 >= 3) & (c30 <= 4)] = 0.2 * 0.3
    c30_score[c30 <= 2] = -0.2 * 0.3
    c30_score[c30 >= 13] = -0.7 * 0.3
    
    nbr_score = np.zeros(80)
    if last_i < len(data):
        for num in data[last_i]:
            for d in [-2, -1, 1, 2]:
                if 1 <= num+d <= 80:
                    nbr_score[num+d-1] += 0.3
    nbr_score /= 20
    
    return c5_score + c10_score + c30_score + nbr_score


def score_v22_v26(last_i):
    """V22.F: 偏热号策略, 用于 4/6 胆
    c5=0: -0.5, c5=1: -0.3, c5=2: +0.8, c5=3: +0.7, c5=4-5: +0.3, c5≥6: -1.5
    """
    if last_i + 30 >= len(data):
        return np.zeros(80)
    c5 = np.zeros(80, dtype=int)
    for k in range(1, 6):
        if last_i + k < len(data):
            for n in data[last_i + k]:
                c5[n-1] += 1
    
    c5_score = np.zeros(80)
    c5_score[c5 == 0] = -0.5
    c5_score[c5 == 1] = -0.3
    c5_score[c5 == 2] = 0.8
    c5_score[c5 == 3] = 0.7
    c5_score[(c5 >= 4) & (c5 <= 5)] = 0.3
    c5_score[c5 >= 6] = -1.5
    
    # c10/c30 各 1.0 倍
    counts = {}
    for w in [10, 30]:
        c = np.zeros(80, dtype=int)
        for k in range(1, w + 1):
            if last_i + k < len(data):
                for n in data[last_i + k]:
                    c[n-1] += 1
        counts[w] = c
    c10, c30 = counts[10], counts[30]
    
    c10_score = np.zeros(80)
    c10_score[(c10 >= 2) & (c10 <= 4)] = 1.0
    c10_score[c10 == 1] = 0.4
    c10_score[c10 == 0] = 0.2
    c10_score[(c10 >= 5) & (c10 <= 6)] = -0.3
    c10_score[c10 >= 7] = -0.8
    
    c30_score = np.zeros(80)
    c30_score[(c30 >= 5) & (c30 <= 10)] = 0.5
    c30_score[(c30 >= 3) & (c30 <= 4)] = 0.2
    c30_score[c30 <= 2] = -0.2
    c30_score[c30 >= 13] = -0.7
    
    return c5_score + c10_score + c30_score


def score_v18c(last_i):
    """V18.C 反向基线: 冷号优先
    短窗口频次反向加权 + 中期重号反向 + 中期邻号反向
    """
    if last_i + 30 >= len(data):
        return np.zeros(80)
    freq = np.zeros(80, dtype=int)
    for k in range(1, 11):
        if last_i + k < len(data):
            for n in data[last_i + k]:
                freq[n-1] += 1
    repeat = np.zeros(80, dtype=int)
    for k in range(1, 4):
        if last_i + k < len(data):
            for n in data[last_i + k]:
                repeat[n-1] += 1
    nbr = np.zeros(80, dtype=int)
    for k in range(1, 6):
        if last_i + k < len(data):
            for n in data[last_i + k]:
                for d in [-1, 1]:
                    if 1 <= n+d <= 80:
                        nbr[n+d-1] += 1
    
    # V18.C 反向
    freq_score = -freq.astype(float) * 0.3
    repeat_score = -repeat.astype(float) * 0.3
    nbr_score = -nbr.astype(float) * 0.3
    
    return freq_score + repeat_score + nbr_score


# ============= 选号函数 =============
def select_with_constraints(scores, last_i, top_n, max_consec=4, max_zone_dev=0.35, max_span=78, quota=None):
    """带 4 区位配额和形态约束的选号
    **修复**: 配额内部 sub-zone 优先选 (1 区填满 → 2 区填满 → 3 区 → 4 区), 
    放宽 zone_dev 到 0.35 直到填满, 然后 max_zone_dev 用 0.25 作为最后检查
    """
    lag_sets = []
    for lag in range(1, 4):
        if last_i + lag - 1 < len(data):
            lag_sets.append((lag, set(data[last_i + lag - 1].tolist())))
    
    sorted_idx = np.argsort(scores)[::-1]
    selected = []
    lag_repeats = {lag: 0 for lag, _ in lag_sets}
    zone_count = [0, 0, 0, 0]
    
    # 第一轮: 按配额严格选
    if quota is not None:
        for z in range(4):
            for idx in sorted_idx:
                if zone_count[z] >= quota[z]:
                    break
                num = int(idx) + 1
                if num in selected:
                    continue
                if zone_of(num) != z:
                    continue
                candidate = selected + [num]
                if len(candidate) >= 3 and max_consecutive_in(candidate) > max_consec:
                    continue
                if len(candidate) >= 4 and (max(candidate) - min(candidate)) > max_span:
                    continue
                # 不检查 zone_dev (配额已经控制)
                for lag, lag_set in lag_sets:
                    if num in lag_set:
                        lag_repeats[lag] += 1
                zone_count[z] += 1
                selected.append(num)
                if len(selected) >= top_n:
                    return sorted(selected[:top_n])
    
    # 第二轮: 填满剩余位置 (不限制 4 区位, 用 zone_dev 检查)
    for idx in sorted_idx:
        if len(selected) >= top_n:
            break
        num = int(idx) + 1
        if num in selected:
            continue
        candidate = selected + [num]
        if len(candidate) >= 3 and max_consecutive_in(candidate) > max_consec:
            continue
        if len(candidate) >= 4 and (max(candidate) - min(candidate)) > max_span:
            continue
        if len(candidate) >= 4 and zone_deviation(candidate) > max_zone_dev:
            continue
        for lag, lag_set in lag_sets:
            if num in lag_set:
                lag_repeats[lag] += 1
        zone_count[zone_of(num)] += 1
        selected.append(num)
    
    return sorted(selected[:top_n])


def select_without_quota(scores, last_i, top_n, max_consec=4, max_zone_dev=0.25, max_span=78):
    """无 4 区位配额 (V18.C 用)"""
    lag_sets = []
    for lag in range(1, 4):
        if last_i + lag - 1 < len(data):
            lag_sets.append((lag, set(data[last_i + lag - 1].tolist())))
    
    sorted_idx = np.argsort(scores)[::-1]
    selected = []
    lag_repeats = {lag: 0 for lag, _ in lag_sets}
    
    for idx in sorted_idx:
        num = int(idx) + 1
        if num in selected:
            continue
        
        # 形态约束
        candidate = selected + [num]
        if len(candidate) >= 3 and max_consecutive_in(candidate) > max_consec:
            continue
        if len(candidate) >= 4 and (max(candidate) - min(candidate)) > max_span:
            continue
        if len(candidate) >= 4 and zone_deviation(candidate) > max_zone_dev:
            continue
        
        for lag, lag_set in lag_sets:
            if num in lag_set:
                lag_repeats[lag] += 1
        selected.append(num)
        if len(selected) >= top_n:
            break
    
    return sorted(selected[:top_n])


# ============= 跑 4 个策略 =============
last_i = periods.index('2026202')

results = {}

# 策略 1: V22.A
scores_v21 = score_v22_v21(last_i)
results['V22.A'] = {
    'method': 'V22.A (P90 约束 + 4 区位配额 + 邻号加权)',
    '4_dan':  select_with_constraints(scores_v21, last_i, 4,  quota=get_quota(4)),
    '6_dan':  select_with_constraints(scores_v21, last_i, 6,  quota=get_quota(6)),
    '10_dan': select_with_constraints(scores_v21, last_i, 10, quota=get_quota(10)),
    '20_dan': select_with_constraints(scores_v21, last_i, 20, quota=get_quota(20)),
}

# 策略 2: V22.E
scores_v25 = score_v22_v25(last_i)
results['V22.E'] = {
    'method': 'V22.E (c5 主权重 + c10/c30 减权)',
    '4_dan':  select_with_constraints(scores_v25, last_i, 4,  quota=get_quota(4)),
    '6_dan':  select_with_constraints(scores_v25, last_i, 6,  quota=get_quota(6)),
    '10_dan': select_with_constraints(scores_v25, last_i, 10, quota=get_quota(10)),
    '20_dan': select_with_constraints(scores_v25, last_i, 20, quota=get_quota(20)),
}

# 策略 3: V22.F 偏热号
scores_v26 = score_v22_v26(last_i)
results['V22.F'] = {
    'method': 'V22.F (偏热号策略, 用于 4/6 胆)',
    '4_dan':  select_with_constraints(scores_v26, last_i, 4,  quota=get_quota(4)),
    '6_dan':  select_with_constraints(scores_v26, last_i, 6,  quota=get_quota(6)),
    '10_dan': select_with_constraints(scores_v26, last_i, 10, quota=get_quota(10)),
    '20_dan': select_with_constraints(scores_v26, last_i, 20, quota=get_quota(20)),
}

# 策略 4: V18.C 反向基线
scores_v18c = score_v18c(last_i)
results['V18C'] = {
    'method': 'V18.C 反向基线 (冷号优先)',
    '4_dan':  select_without_quota(scores_v18c, last_i, 4),
    '6_dan':  select_without_quota(scores_v18c, last_i, 6),
    '10_dan': select_without_quota(scores_v18c, last_i, 10),
    '20_dan': select_without_quota(scores_v18c, last_i, 20),
}


# ============= 输出 =============
print("\n" + "=" * 80)
print(f"🎯 2026203 期 多策略预测 (上期 2026202: {PRIOR})")
print("=" * 80)

for name, r in results.items():
    print(f"\n━━━ {name} ━━━")
    print(f"方法: {r['method']}")
    for k in ['4_dan', '6_dan', '10_dan', '20_dan']:
        nums = r[k]
        zones = [zone_of(n) for n in nums]
        max_c = max_consecutive_in(nums)
        span = max(nums) - min(nums)
        print(f"  {k:8s}: {nums}")
        print(f"              4区位: {zones.count(0)}/{zones.count(1)}/{zones.count(2)}/{zones.count(3)}, "
              f"最大连号: {max_c}, 跨度: {span}")
        # 显示重号
        repeats = [n for n in nums if n in PRIOR]
        if repeats:
            print(f"              重号(上期): {repeats}")

# 共识分析
print("\n" + "=" * 80)
print("🤝 共识分析 (4 策略交集)")
print("=" * 80)
for k in ['4_dan', '6_dan', '10_dan']:
    sets = [set(r[k]) for r in results.values()]
    consensus = set.intersection(*sets)
    union = set.union(*sets)
    print(f"\n{k}:")
    print(f"  4 策略交集: {sorted(consensus) if consensus else '❌ 无'}")
    print(f"  4 策略并集 ({len(union)}): {sorted(union)}")
    # 出现 3+ 次的号
    from collections import Counter
    cnt = Counter()
    for r in results.values():
        for n in r[k]:
            cnt[n] += 1
    top = [(n, c) for n, c in cnt.most_common() if c >= 3]
    print(f"  出现 3+ 次的号: {top}")

# 保存 JSON
output = {
    'meta': {
        'target_period': TARGET,
        'prior_period': '2026202',
        'prior_numbers': PRIOR,
        'created_at': '2026-08-01 15:30 GMT+8',
        'method': '多策略独立预测 (V22.A / V22.E / V22.F + V18.C)',
        'author': 'Lucky / MiniMax-M3',
    },
    'predictions': {
        name: {
            '4_dan': r['4_dan'],
            '6_dan': r['6_dan'],
            '10_dan': r['10_dan'],
            '20_dan': r['20_dan'],
        }
        for name, r in results.items()
    }
}

out_path = ROOT / 'data/backtest/2026203_predictions_multi_strategy.json'
with open(out_path, 'w', encoding='utf-8') as f:
    json.dump(output, f, ensure_ascii=False, indent=2)
print(f"\n✅ 预测已保存: {out_path}")
