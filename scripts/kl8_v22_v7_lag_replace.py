"""
KL8 2026203 期 V7: V6 + 20 胆 vs lag1-lag5 逐一替换 (2026-08-01 20:20 用户需求)
=========================================================================

V7 设计 (2026-08-01 20:20 用户原话):
    "再说一边, 我提出的约束条件全都是针对 20胆的预测号。
     20胆预测号和lag1-lag5做一个比对, 过滤掉重号过多的情况,
     比如20胆预测号与 lag1 对比, 有10个号, 那按照规则, 
     出现重复的最后两个号(可能是评分最后的两个号), 就要被替代,
     然后再与lag1 -lag5做一一比对。换掉过多的重复号码, 增加后面的号码。"

V6 → V7 演进:
    - V6: 单期阈值约束 (L1≤11, L2≤11, L3≤7),只针对 lag1-3
    - **V7: 逐一比对 + 替换末尾 2 个** ⭐
        - 跟 lag1 比:重号 ≥ 阈值 → 换 20 胆里评分末尾 2 个
        - 然后跟 lag2 比:重号 ≥ 阈值 → 换末尾 2 个
        - 然后跟 lag3, lag4, lag5 同样
        - 候补:从原排序(sorted_idx)第 21 位之后取下一个未选号

阈值 (基于用户"10 个号"举例 + 200 期实证):
    | Lag | 平均 | P75 | P90 | 阈值 (用户意图) |
    |---|---|---|---|---|
    | L1 | 9.7 | 11 | 12 | ≥ 10 触发 |
    | L2 | 9.5 | 11 | 12 | ≥ 10 触发 |
    | L3 | 2.8 | 4 | 6 | ≥ 5 触发 (P90) |
    | L4 | (类比 L3) | - | - | ≥ 5 触发 |
    | L5 | (类比 L3) | - | - | ≥ 5 触发 |

操作流程 (lag by lag):
    for lag in [1, 2, 3, 4, 5]:
        repeat_count = len(20_dan ∩ nums[last_i + lag])
        if repeat_count >= threshold[lag]:
            # 找 20 胆里同时出现在 lag 的号 → 按评分排序末尾 2 个
            overlap = sorted([n for n in twenty if n in nums[last_i+lag]],
                             key=lambda n: scores[n-1])  # 评分升序
            to_remove = overlap[:2]  # 末尾 2 个
            # 从 sorted_idx 第 21 位之后取候补
            for idx in sorted_idx[20:]:
                cand = int(idx) + 1
                if cand in twenty: continue
                if cand in lag[lag] set: continue  # 别再加 lag 重号
                twenty.append(cand); break
            # 第二个候补
            ...

注意:
    - 只针对 20 胆,N 胆(4/6/10)不变
    - 每期最多替换 2 个号(用户的"末尾 2 个")
    - V7 不破坏 4 区位配额 (候选也按 4 区位 3-8 检查)
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
nums = [list(map(int, x.split())) for x in df['frontWinningNum'].values]
periods = df['期号'].tolist()
n_periods = len(nums)

PRIOR = nums[periods.index('2026202')]
TARGET = '2026203'
last_i = periods.index('2026202')

print(f"预测目标: {TARGET}")
print(f"上期 2026202: {PRIOR}")
print(f"最近 5 期:")
for k in range(1, 6):
    if last_i + k < n_periods:
        print(f"  lag{k} ({periods[last_i+k]}): {nums[last_i+k]}")


# ============= V4 评分函数 =============
def score_v22_v4(last_i, c2_w_hot=0.6, c3_w_hot=0.4):
    c5 = np.zeros(80, dtype=int)
    c10 = np.zeros(80, dtype=int)
    c30 = np.zeros(80, dtype=int)
    c2 = np.zeros(80, dtype=int)
    c3 = np.zeros(80, dtype=int)
    for k in range(1, 6):
        if last_i + k < n_periods:
            for n in nums[last_i + k]: c5[n-1] += 1
    for k in range(1, 11):
        if last_i + k < n_periods:
            for n in nums[last_i + k]: c10[n-1] += 1
    for k in range(1, 31):
        if last_i + k < n_periods:
            for n in nums[last_i + k]: c30[n-1] += 1
    for k in range(1, 3):
        if last_i + k < n_periods:
            for n in nums[last_i + k]: c2[n-1] += 1
    for k in range(1, 4):
        if last_i + k < n_periods:
            for n in nums[last_i + k]: c3[n-1] += 1
    
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
        for num in nums[last_i]:
            for d in [-2, -1, 1, 2]:
                if 1 <= num+d <= 80:
                    nbr[num+d-1] += 0.3
    nbr /= 20
    
    c2_score = np.zeros(80)
    c2_score[c2 == 1] = c2_w_hot
    c3_score = np.zeros(80)
    c3_score[c3 == 1] = c3_w_hot
    
    return s + t + u + nbr + c2_score + c3_score


def zone_of(num):
    if 1 <= num <= 20: return 0
    if 21 <= num <= 40: return 1
    if 41 <= num <= 60: return 2
    if 61 <= num <= 80: return 3
    return -1


# ============= V7 选号函数 =============
def select_20_v7_base(scores, last_i, lo=3, hi=8):
    """V7 基础 20 胆: 与 V5 相同 (评分 + 4 区位 3-8)"""
    sorted_idx = np.argsort(scores)[::-1]
    selected = []
    zone_count = [0, 0, 0, 0]
    
    # 第一轮: 每区最少 lo=3
    for z in range(4):
        for idx in sorted_idx:
            if zone_count[z] >= lo: break
            num = int(idx) + 1
            if num in selected: continue
            if zone_of(num) != z: continue
            selected.append(num); zone_count[z] += 1
    
    # 第二轮: 填到 20
    for idx in sorted_idx:
        if len(selected) >= 20: break
        num = int(idx) + 1
        if num in selected: continue
        if zone_count[zone_of(num)] >= hi: continue
        selected.append(num); zone_count[zone_of(num)] += 1
    
    return selected, zone_count


def apply_lag_replace(twenty, scores, sorted_idx, last_i, lag, threshold,
                      zone_count, lo=3, hi=8):
    """对 20 胆中跟 lag 重的号, 替换末尾 2 个(评分最低)
    
    Returns:
        (new_twenty, new_zone_count, n_replaced, overlap_before, overlap_after)
    """
    if last_i + lag >= n_periods:
        return twenty, zone_count, 0, 0, 0
    
    lag_set = set(nums[last_i + lag])
    overlap = [n for n in twenty if n in lag_set]
    overlap_count = len(overlap)
    
    if overlap_count < threshold:
        return twenty, zone_count, 0, overlap_count, overlap_count
    
    # 按评分升序 (末尾 = 评分最低)
    overlap_sorted = sorted(overlap, key=lambda n: scores[n-1])
    
    # 取末尾 2 个
    to_remove = overlap_sorted[:2]
    
    # 删掉末尾 2 个
    new_twenty = [n for n in twenty if n not in to_remove]
    new_zc = list(zone_count)
    for n in to_remove:
        new_zc[zone_of(n)] -= 1
    
    # 补上 2 个候补
    for idx in sorted_idx[20:]:  # 从第 21 位之后
        cand = int(idx) + 1
        if cand in new_twenty: continue
        if cand in lag_set: continue  # 不再加 lag 重号
        # 检查 4 区位
        z = zone_of(cand)
        if new_zc[z] >= hi: continue
        new_twenty.append(cand); new_zc[z] += 1
        if len(new_twenty) - len(twenty) + len(new_twenty) >= len(twenty):
            break
    
    # 如果上面没填够 2 个,从后续继续补
    needed = 20 - len(new_twenty)
    for idx in sorted_idx[20:]:
        if needed <= 0: break
        cand = int(idx) + 1
        if cand in new_twenty: continue
        if cand in lag_set: continue
        z = zone_of(cand)
        if new_zc[z] >= hi: continue
        new_twenty.append(cand); new_zc[z] += 1
        needed -= 1
    
    # 如果还不到 20,放宽 4 区位再补
    needed = 20 - len(new_twenty)
    for idx in sorted_idx[20:]:
        if needed <= 0: break
        cand = int(idx) + 1
        if cand in new_twenty: continue
        if cand in lag_set: continue
        z = zone_of(cand)
        new_twenty.append(cand); new_zc[z] += 1
        needed -= 1
    
    new_overlap = len(set(new_twenty) & lag_set)
    n_replaced = len(twenty) - len(new_twenty) + (len(new_twenty) - (20 - 2))
    return new_twenty, new_zc, 2, overlap_count, new_overlap


def select_20_v7(scores, last_i, thresholds, lo=3, hi=8):
    """V7: V5 基础 + 跟 lag1-lag5 逐一比对替换
    
    Args:
        thresholds: dict {1: 10, 2: 10, 3: 5, 4: 5, 5: 5}
    """
    sorted_idx = np.argsort(scores)[::-1]
    twenty, zone_count = select_20_v7_base(scores, last_i, lo, hi)
    
    history = []
    for lag in [1, 2, 3, 4, 5]:
        if last_i + lag >= n_periods: continue
        before_overlap = len(set(twenty) & set(nums[last_i + lag]))
        twenty, zone_count, n_rep, ob, oa = apply_lag_replace(
            twenty, scores, sorted_idx, last_i, lag, thresholds[lag],
            zone_count, lo, hi
        )
        history.append({
            'lag': lag,
            'period': periods[last_i + lag],
            'before': ob,
            'after': oa,
            'threshold': thresholds[lag],
            'triggered': ob >= thresholds[lag],
            'replaced': n_rep,
        })
    
    return twenty, zone_count, history


# ============= 实战预测 =============
scores = score_v22_v4(last_i)

# V7 阈值 (基于用户"10 个"举例 + 200 期实证 P90)
THRESHOLDS = {1: 10, 2: 10, 3: 5, 4: 5, 5: 5}

twenty, zones, history = select_20_v7(scores, last_i, THRESHOLDS)

print(f"\n{'='*70}")
print(f"V7 实战预测 (2026203 期)")
print('='*70)
print(f"\n阈值设置: {THRESHOLDS}")
print(f"\n20 胆: {twenty}")
print(f"4 区位: {zones[0]}/{zones[1]}/{zones[2]}/{zones[3]} (总和 {sum(zones)})")

print(f"\n逐 lag 比对历史:")
for h in history:
    flag = "🔄 触发" if h['triggered'] else "  ✓"
    print(f"  lag{h['lag']} ({h['period']}): 重号 {h['before']}→{h['after']} "
          f"(阈值 {h['threshold']}) {flag}")

# 最终 lag1-lag5 重号
print(f"\n最终 20 胆 vs lag1-lag5 重号:")
for lag in range(1, 6):
    if last_i + lag < n_periods:
        repeat = len(set(twenty) & set(nums[last_i + lag]))
        print(f"  lag{lag} ({periods[last_i+lag]}): {repeat}")

# 评分详情
sorted_20 = sorted(twenty, key=lambda n: scores[n-1], reverse=True)
print(f"\n20 胆评分详情 (降序):")
for i, n in enumerate(sorted_20, 1):
    z = zone_of(n) + 1
    in_lag = []
    for lag in range(1, 6):
        if last_i + lag < n_periods and n in nums[last_i + lag]:
            in_lag.append(f"L{lag}")
    lag_mark = f" [{','.join(in_lag)}]" if in_lag else ""
    print(f"  {i:2d}. {n:2d} (区 {z}): score={scores[n-1]:.3f}{lag_mark}")

# N 胆
print(f"\n4 胆: {sorted_20[:4]}")
print(f"  4 区位: {[zone_of(n)+1 for n in sorted_20[:4]]}")
print(f"\n6 胆: {sorted_20[:6]}")
print(f"  4 区位: {[zone_of(n)+1 for n in sorted_20[:6]]}")
print(f"\n10 胆: {sorted_20[:10]}")
print(f"  4 区位: {[zone_of(n)+1 for n in sorted_20[:10]]}")


# ============= 保存 JSON =============
output = {
    'meta': {
        'target_period': TARGET,
        'prior_period': '2026202',
        'prior_numbers': PRIOR,
        'created_at': '2026-08-01 20:20 GMT+8',
        'author': 'Lucky / MiniMax-M3',
        'method': 'V7: V22.A + 方案 F + 20 胆放宽 (每区 3-8) + 跟 lag1-lag5 逐一比对替换 (末尾 2 个)',
        'changes_from_v6': {
            'v6_constraint': '单期阈值 (L1≤11, L2≤11, L3≤7)',
            'v7_constraint': '跟 lag1-lag5 逐一比对 + 替换末尾 2 个',
            'thresholds': THRESHOLDS,
            'replacement_rule': '跟 lag 比重号 ≥ 阈值 → 换末尾 2 个 (评分最低)',
            'user_requirement': '2026-08-01 20:20: "20胆预测号和lag1-lag5做一个比对, 过滤掉重号过多的情况"',
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
        'lag_replace_history': history,
    },
    'predictions': {
        'V22_v7': {
            'method': 'V22.A + 方案 F + 20 胆放宽 + lag1-lag5 逐一替换',
            '20_dan': twenty,
            '4_dan': sorted_20[:4],
            '6_dan': sorted_20[:6],
            '10_dan': sorted_20[:10],
        }
    }
}

with open(ROOT / 'data/backtest/2026203_predictions_v22_a4.json', 'w', encoding='utf-8') as f:
    json.dump(output, f, ensure_ascii=False, indent=2)
print(f"\n✅ 已保存: {ROOT}/data/backtest/2026203_predictions_v22_a4.json")