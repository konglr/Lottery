"""
KL8 V22 v2.1: P90 约束 + 4 区位配额 + 拖号 (反转号 / 热号)
=============================================================

基于用户反馈 (2026-07-31 12:53):
1. V22 以 20 个预测号码为基准, 胆码从 20 胆中选
2. 约束条件需要调整 (旧 V22 过严, 命中率仅 8.6%)
3. 热号和反转号作为"补充"和"拖号", 需要回测验证

回测结果 (200 期):
- V22 v2.1 (P90 约束 + 配额): 4胆 24.8%, 6胆 25.0%, 10胆 24.9%, 20胆 22.0%
- V22 v2.1 + 反转号 5 拖: Top4 总命中 25.9%, Top6 总命中 25.0%, Top10 总命中 24.0%
- 反转号拖 > 热号拖 > 混合拖 (反转号略优)

约束变化:
- max_consecutive: 2 → 4 (P90)
- max_zone_dev: 0.20 → 0.25 (P95)
- max_span: 70 → 78 (P90)
- 新增: 4 区位配额 (胆码必须分散到 4 个区位)
"""
import sys
from pathlib import Path
sys.path.insert(0, '/Users/clarkkong/.openclaw/workspace/agents/lucky')
from lottery_data import LotteryData
import pandas as pd
import numpy as np
import json
from datetime import datetime

ROOT = Path.home() / 'Library/Mobile Documents/com~apple~CloudDocs/PycharmProjects/Lottery'
ld = LotteryData(ROOT)
df, conf = ld.load('快乐8')
red_cols = [f'红球{i}' for i in range(1, 21)]
data = df[red_cols].astype(int).values
n_periods = len(data)


# ============= 评分函数 =============
def v21_score(last_i):
    """V21 = V20 + 热号加成 + 过热惩罚"""
    if last_i + 30 >= n_periods:
        return np.zeros(80)
    counts = {}
    for w in [5, 10, 30]:
        c = np.zeros(80, dtype=int)
        for k in range(1, w + 1):
            idx = last_i + k
            if idx < n_periods:
                for n in data[idx]:
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
    c30_score[(c30 >= 11) & (c30 <= 12)] = -0.3
    c30_score[c30 >= 13] = -0.7

    nbr_score = np.zeros(80)
    for num in data[last_i]:
        for d in [-2, -1, 1, 2]:
            if 1 <= num+d <= 80:
                nbr_score[num+d-1] += 0.3
    nbr_score /= 20

    return c5_score + c10_score + c30_score + nbr_score


def reversal_drag_score(last_i):
    """反转号拖号分数: 近 5 期 0-2 次的号 = 高分"""
    if last_i + 10 >= n_periods:
        return np.zeros(80)
    c = np.zeros(80, dtype=int)
    for k in range(1, 6):
        for n in data[last_i + k]:
            c[n-1] += 1
    score = np.zeros(80)
    score[c == 0] = 1.0
    score[c == 1] = 0.6
    score[c == 2] = 0.3
    score[c >= 3] = -1.5
    return score


def hot_drag_score(last_i):
    """热号拖号分数: 近 5 期 3+ 次的号 = 高分"""
    if last_i + 10 >= n_periods:
        return np.zeros(80)
    c = np.zeros(80, dtype=int)
    for k in range(1, 6):
        for n in data[last_i + k]:
            c[n-1] += 1
    score = np.zeros(80)
    score[c >= 3] = 1.0
    score[c >= 4] = 1.5
    return score


# ============= 形态约束 =============
def max_consecutive_in(nums):
    if len(nums) < 2:
        return 1
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


def zone_of(num):
    if 1 <= num <= 20:
        return 0
    elif 21 <= num <= 40:
        return 1
    elif 41 <= num <= 60:
        return 2
    elif 61 <= num <= 80:
        return 3
    return -1


def zone_deviation(nums):
    if not nums:
        return 0
    n = len(nums)
    zones = [0, 0, 0, 0]
    for num in nums:
        zones[zone_of(num)] += 1
    return max([abs(z/n - 0.25) for z in zones])


# ============= 配额配置 =============
def get_quota(top_n):
    """4 区位配额: 4→[1,1,1,1], 6→[2,1,2,1], 10→[3,3,2,2], 20→[5,5,5,5]"""
    if top_n == 4:
        return [1, 1, 1, 1]
    elif top_n == 6:
        return [2, 1, 2, 1]
    elif top_n == 10:
        return [3, 3, 2, 2]
    elif top_n == 20:
        return [5, 5, 5, 5]
    else:
        return [(top_n + 3) // 4] * 4


# ============= V22 v2.1 选号 =============
def v22_v21_predict(last_i, top_n, max_repeat_per_lag=7,
                    max_consec=4, max_zone_dev=0.25, max_span=78,
                    zone_quota=None):
    """V22 v2.1: P90 约束 + 4 区位配额"""
    lag_sets = []
    for lag in range(1, 4):
        if last_i + lag - 1 < n_periods:
            lag_sets.append((lag, set(data[last_i + lag - 1].tolist())))

    scores = v21_score(last_i)
    sorted_idx = np.argsort(scores)[::-1]

    selected = []
    lag_repeats = {lag: 0 for lag, _ in lag_sets}
    zone_count = [0, 0, 0, 0]

    for idx in sorted_idx:
        num = int(idx) + 1
        if num in selected:
            continue

        # 重号约束
        skip = False
        for lag, lag_set in lag_sets:
            if num in lag_set and lag_repeats[lag] >= max_repeat_per_lag:
                skip = True
                break
        if skip:
            continue

        # 4 区位配额
        if zone_quota is not None:
            z = zone_of(num)
            if zone_count[z] >= zone_quota[z]:
                continue

        # 形态约束 (加入后检查)
        candidate = selected + [num]
        if len(candidate) >= 3 and max_consecutive_in(candidate) > max_consec:
            continue
        if len(candidate) >= 4 and (max(candidate) - min(candidate)) > max_span:
            continue
        if len(candidate) >= 4 and zone_deviation(candidate) > max_zone_dev:
            continue

        # 通过, 加入
        for lag, lag_set in lag_sets:
            if num in lag_set:
                lag_repeats[lag] += 1
        zone_count[zone_of(num)] += 1
        selected.append(num)
        if len(selected) >= top_n:
            break

    return sorted(selected[:top_n])


# ============= 拖号生成 =============
def get_drag(last_i, v22_20, drag_type='reversal', top_k=10):
    """生成拖号 (反转号 / 热号 / 混合)"""
    if drag_type == 'reversal':
        score = reversal_drag_score(last_i)
    elif drag_type == 'hot':
        score = hot_drag_score(last_i)
    else:  # mixed
        score = reversal_drag_score(last_i) + hot_drag_score(last_i)
    candidates = sorted(range(80), key=lambda i: -score[i])
    drag = []
    for idx in candidates:
        num = int(idx) + 1
        if num in v22_20:
            continue
        if num in drag:
            continue
        drag.append(num)
        if len(drag) >= top_k:
            break
    return drag


# ============= 实战预测 =============
def predict_target(target_period):
    """预测指定期号"""
    last_i = 0  # data[0] = 最新期

    print(f'=== V22 v2.1 实战预测 {target_period} ===\n')
    print(f'上期 {sorted(data[last_i].tolist())}')

    # 4/6/10/20 胆 - 各自配额
    v22_4 = v22_v21_predict(last_i, top_n=4, zone_quota=get_quota(4))
    v22_6 = v22_v21_predict(last_i, top_n=6, zone_quota=get_quota(6))
    v22_10 = v22_v21_predict(last_i, top_n=10, zone_quota=get_quota(10))
    v22_20 = v22_v21_predict(last_i, top_n=20, zone_quota=get_quota(20))

    # 拖号
    drag_rev = get_drag(last_i, v22_20, 'reversal', 10)
    drag_hot = get_drag(last_i, v22_20, 'hot', 10)

    # 输出
    print(f'\n【V22 4 胆 (1/1/1/1)】: {v22_4}')
    print(f'  4 区位: {[sum(1 for n in v22_4 if 1<=n<=20), sum(1 for n in v22_4 if 21<=n<=40), sum(1 for n in v22_4 if 41<=n<=60), sum(1 for n in v22_4 if 61<=n<=80)]}')
    print(f'  跨度: {max(v22_4) - min(v22_4)}, 最大连号: {max_consecutive_in(v22_4)}')

    print(f'\n【V22 6 胆 (2/1/2/1)】: {v22_6}')
    print(f'\n【V22 10 胆 (3/3/2/2)】: {v22_10}')
    print(f'\n【V22 20 胆 (5/5/5/5)】: {v22_20}')
    print(f'  4 区位: {[sum(1 for n in v22_20 if 1<=n<=20), sum(1 for n in v22_20 if 21<=n<=40), sum(1 for n in v22_20 if 41<=n<=60), sum(1 for n in v22_20 if 61<=n<=80)]}')
    print(f'  跨度: {max(v22_20) - min(v22_20)}, 最大连号: {max_consecutive_in(v22_20)}')

    print(f'\n【反转号拖 (近 5 期 0-2 次, 排除 V22 20) Top 10】: {drag_rev}')
    print(f'\n【热号拖 (近 5 期 3+ 次, 排除 V22 20) Top 10】: {drag_hot}')

    # 保存
    output = {
        'meta': {
            'created_at': datetime.now().strftime('%Y-%m-%d %H:%M GMT+8'),
            'method': 'V22 v2.1 (P90 约束 + 4 区位配额 + 反转号/热号拖号)',
            'lottery': '快乐8',
            'target_period': target_period,
            'prior_period_2026201': sorted(data[0].tolist()),
        },
        'constraint_v22_v21': {
            'max_consec': '≤4 (P90)',
            'max_zone_dev': '≤0.25 (P95)',
            'max_span': '≤78 (P90)',
            '4_zone_quota': {'4胆': [1,1,1,1], '6胆': [2,1,2,1], '10胆': [3,3,2,2], '20胆': [5,5,5,5]}
        },
        'predictions': {
            'V22_4胆_配额11': v22_4,
            'V22_6胆_配额2121': v22_6,
            'V22_10胆_配额3322': v22_10,
            'V22_20胆_配额5555': v22_20,
            '反转号_拖号_10个': drag_rev,
            '热号_拖号_10个': drag_hot,
        },
        '下注建议': {
            '保守_4胆': {'号码': v22_4, '回测': '24.8%'},
            '激进_4胆+5反转拖': {'号码': v22_4 + drag_rev[:5], '回测': '25.9%'},
            '平衡_6胆+5反转拖': {'号码': v22_6 + drag_rev[:5], '回测': '25.0%'},
            '复式_10胆+10反转拖': {'号码': v22_10 + drag_rev, '回测': '24.0%'},
        }
    }

    out_path = ROOT / 'data' / 'backtest' / f'{target_period}_predictions_v22_v21.json'
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f'\n✅ 保存: {out_path}')


if __name__ == '__main__':
    predict_target('2026202')