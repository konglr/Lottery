"""
KL8 2026203 期 多策略预测 (V3: 严格 20 胆基准 + 纯评分驱动)
=============================================================

约束 (用户 2026-08-01 15:37 + 15:53 确认):
    1. 所有 N 胆 (4/6/10) 都从 20 个预测号码中获取
    2. **小胆号选择, 4 区位配额/跨度/连号 限制都不重要**
       — 只看 20 胆中评分最高的 N 个

实现方式:
    1. 每个策略先选 20 个基础号码 (固定 4 区位 5/5/5/5)
    2. 4 胆 / 6 胆 / 10 胆 都从 20 胆中按分数排序取前 N 个
    3. **没有 4 区位配额/跨度/连号约束** (V2 → V3 变更)
    4. 20 胆 4 区位 5/5/5/5: 保持

V1 → V2 → V3 变更历史:
    V1: N 胆独立选, 有 4 区位配额
    V2: N 胆从 20 胆派生, 仍受 4 区位配额限制
    V3 (current): N 胆从 20 胆派生, 纯评分驱动

4 个策略:
    A. V22.A (P90 约束 + 4 区位配额 + 邻号加权) - 主力
    B. V22.E (c5 主权重 + c10/c30 减权) - 平衡
    C. V22.F (偏热号策略) - 偏 c5=2 温号
    D. V18.C 反向基线 - 冷号优先
"""
import sys
from pathlib import Path
import json
from datetime import datetime
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

PRIOR = list(map(int, df[df['期号'] == '2026202'].iloc[0]['frontWinningNum'].split()))
TARGET = '2026203'
last_i = periods.index('2026202')

print(f"预测目标: {TARGET}")
print(f"上期 2026202 开奖: {PRIOR}")


# ============= 工具 =============
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


# ============= 4 个评分函数 =============
def score_v22_v21(last_i):
    """V22.A: c5/c10/c30 + 邻号 (基线)"""
    c5 = np.zeros(80, dtype=int)
    for k in range(1, 6):
        for n in data[last_i + k]:
            c5[n-1] += 1
    c10 = np.zeros(80, dtype=int)
    for k in range(1, 11):
        for n in data[last_i + k]:
            c10[n-1] += 1
    c30 = np.zeros(80, dtype=int)
    for k in range(1, 31):
        for n in data[last_i + k]:
            c30[n-1] += 1
    
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
    for num in data[last_i]:
        for d in [-2, -1, 1, 2]:
            if 1 <= num+d <= 80:
                nbr[num+d-1] += 0.3
    nbr /= 20
    
    return s + t + u + nbr


def score_v22_v25(last_i):
    """V22.E: c5 主权重 + c10/c30 减权 0.3 倍"""
    c5 = np.zeros(80, dtype=int)
    for k in range(1, 6):
        for n in data[last_i + k]:
            c5[n-1] += 1
    c10 = np.zeros(80, dtype=int)
    for k in range(1, 11):
        for n in data[last_i + k]:
            c10[n-1] += 1
    c30 = np.zeros(80, dtype=int)
    for k in range(1, 31):
        for n in data[last_i + k]:
            c30[n-1] += 1
    
    s = np.zeros(80)
    s[c5 == 0] = 0.5
    s[c5 == 1] = 0.5
    s[c5 == 2] = 0.6
    s[c5 == 3] = -0.5
    s[(c5 >= 4) & (c5 <= 5)] = -1.5
    s[c5 >= 6] = -3.0
    
    t = np.zeros(80)
    t[(c10 >= 2) & (c10 <= 4)] = 0.3
    t[c10 == 1] = 0.12
    t[c10 == 0] = 0.06
    t[(c10 >= 5) & (c10 <= 6)] = -0.09
    t[c10 >= 7] = -0.24
    
    u = np.zeros(80)
    u[(c30 >= 5) & (c30 <= 10)] = 0.15
    u[(c30 >= 3) & (c30 <= 4)] = 0.06
    u[c30 <= 2] = -0.06
    u[c30 >= 13] = -0.21
    
    nbr = np.zeros(80)
    for num in data[last_i]:
        for d in [-2, -1, 1, 2]:
            if 1 <= num+d <= 80:
                nbr[num+d-1] += 0.3
    nbr /= 20
    
    return s + t + u + nbr


def score_v22_v26(last_i):
    """V22.F: 偏热号策略 (c5=2 高加权)"""
    c5 = np.zeros(80, dtype=int)
    for k in range(1, 6):
        for n in data[last_i + k]:
            c5[n-1] += 1
    c10 = np.zeros(80, dtype=int)
    for k in range(1, 11):
        for n in data[last_i + k]:
            c10[n-1] += 1
    c30 = np.zeros(80, dtype=int)
    for k in range(1, 31):
        for n in data[last_i + k]:
            c30[n-1] += 1
    
    s = np.zeros(80)
    s[c5 == 0] = -0.5
    s[c5 == 1] = -0.3
    s[c5 == 2] = 0.8
    s[c5 == 3] = 0.7
    s[(c5 >= 4) & (c5 <= 5)] = 0.3
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
    
    return s + t + u


def score_v18c(last_i):
    """V18.C 反向基线: 冷号优先"""
    freq = np.zeros(80, dtype=int)
    for k in range(1, 11):
        for n in data[last_i + k]:
            freq[n-1] += 1
    repeat = np.zeros(80, dtype=int)
    for k in range(1, 4):
        for n in data[last_i + k]:
            repeat[n-1] += 1
    nbr = np.zeros(80, dtype=int)
    for k in range(1, 6):
        for n in data[last_i + k]:
            for d in [-1, 1]:
                if 1 <= n+d <= 80:
                    nbr[n+d-1] += 1
    
    return -freq * 0.3 - repeat * 0.3 - nbr * 0.3


# ============= 20 胆选号 (4 区位 5/5/5/5) =============
def select_20_base(scores, last_i, max_consec=4, max_span=78, max_zone_dev=0.05):
    """20 胆选号: 4 区位 5/5/5/5, 形态约束
    
    严格 4 区位 5/5/5/5: 1 区填满 5 个 → 2 区填满 5 个 → 3 区 → 4 区
    zone_deviation 限制放宽到 0.05 (允许 1 区填满前偏差 5/20 = 0.0, 加完仍 0.0)
    """
    sorted_idx = np.argsort(scores)[::-1]
    selected = []
    zone_count = [0, 0, 0, 0]
    zone_quota = [5, 5, 5, 5]
    
    # 第一轮: 按区位优先级, 严格 5/5/5/5
    for z in range(4):
        for idx in sorted_idx:
            if zone_count[z] >= zone_quota[z]:
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
            # 不检查 zone_dev (区位优先级已经控制)
            zone_count[z] += 1
            selected.append(num)
            if len(selected) >= 20:
                return sorted(selected), zone_count
    
    return sorted(selected), zone_count


# ============= 派生 N 胆 (从 20 胆中取, 纯评分驱动) =============
def derive_n_from_20(twenty_picks, scores, top_n):
    """从 20 胆中按分数取前 N 个 (V2: 纯评分驱动, 无区位/跨度/连号约束)
    
    哲学 (用户 2026-08-01 15:53 确认):
        - 小胆号选择 上, 4 区位配额 / 跨度 / 连号 / 最大连号 这些限制都不重要
        - 仍是以 20 胆为基准, 取评分最高的 N 个
    
    Args:
        twenty_picks: 20 个号码 (list)
        scores: 80 个号的全评分
        top_n: 4/6/10
    """
    # 按分数从 20 胆中排序 (高分优先)
    sorted_20 = sorted(twenty_picks, key=lambda n: scores[n-1], reverse=True)
    
    # 取前 N 个 (不限区位)
    selected = sorted_20[:top_n]
    zone_count = [0, 0, 0, 0]
    for n in selected:
        zone_count[zone_of(n)] += 1
    
    return sorted(selected), zone_count


# ============= 跑 4 个策略 =============
strategies = {
    'V22.A': {
        'method': 'V22.A (P90 约束 + 4 区位配额 + 邻号加权)',
        'scorer': score_v22_v21,
    },
    'V22.E': {
        'method': 'V22.E (c5 主权重 + c10/c30 减权)',
        'scorer': score_v22_v25,
    },
    'V22.F': {
        'method': 'V22.F (偏热号策略)',
        'scorer': score_v22_v26,
    },
    'V18.C': {
        'method': 'V18.C 反向基线 (冷号优先)',
        'scorer': score_v18c,
    },
}

QUOTAS = {
    4:  [1, 1, 1, 1],
    6:  [2, 1, 2, 1],
    10: [3, 3, 2, 2],
    20: [5, 5, 5, 5],
}

results = {}

for name, cfg in strategies.items():
    print(f"\n{'='*70}")
    print(f"策略: {name} - {cfg['method']}")
    print('='*70)
    
    scores = cfg['scorer'](last_i)
    
    # 1. 选 20 胆
    twenty, twenty_zones = select_20_base(scores, last_i)
    print(f"\n20 胆 ({len(twenty)} 个): {twenty}")
    print(f"  4 区位: {twenty_zones[0]}/{twenty_zones[1]}/{twenty_zones[2]}/{twenty_zones[3]}")
    print(f"  跨度: {max(twenty)-min(twenty)}, 最大连号: {max_consecutive_in(twenty)}")
    
    # 2. 派生 4 胆 / 6 胆 / 10 胆 (V3: 纯评分驱动, 不受区位/跨度/连号约束)
    picks = {'20_dan': twenty}
    for n in [4, 6, 10]:
        sub, sub_zones = derive_n_from_20(twenty, scores, n)
        picks[f'{n}_dan'] = sub
        print(f"\n{n} 胆 ({len(sub)} 个): {sub}")
        print(f"  4 区位: {sub_zones[0]}/{sub_zones[1]}/{sub_zones[2]}/{sub_zones[3]}")
        print(f"  跨度: {max(sub)-min(sub)}, 最大连号: {max_consecutive_in(sub)}")
        
        # 验证包含关系
        assert set(sub).issubset(set(twenty)), f"{n} 胆不在 20 胆中!"
    
    # 3. 重号统计
    repeats = [n for n in twenty if n in PRIOR]
    print(f"\n20 胆重号(上期): {repeats}")
    
    results[name] = {
        'method': cfg['method'],
        '20_dan': picks['20_dan'],
        '4_dan': picks['4_dan'],
        '6_dan': picks['6_dan'],
        '10_dan': picks['10_dan'],
        '20_dan_zones': twenty_zones,
    }


# ============= 4 战略对比 =============
print("\n" + "=" * 70)
print("🤝 4 战略对比")
print("=" * 70)

for n in [4, 6, 10, 20]:
    print(f"\n--- {n} 胆 ---")
    for name, r in results.items():
        nums = r[f'{n}_dan']
        print(f"  {name:12s}: {nums}")
    
    # 共识
    from collections import Counter
    cnt = Counter()
    for r in results.values():
        for num in r[f'{n}_dan']:
            cnt[num] += 1
    
    consensus_4 = sorted([num for num, c in cnt.items() if c == 4])
    consensus_3 = sorted([num for num, c in cnt.items() if c >= 3])
    consensus_2 = sorted([num for num, c in cnt.items() if c >= 2])
    
    print(f"  4 策略全共识: {consensus_4 if consensus_4 else '❌ 无'}")
    print(f"  3+ 策略共识: {consensus_3 if consensus_3 else '❌ 无'}")
    print(f"  2+ 策略共识: {consensus_2 if consensus_2 else '❌ 无'}")


# ============= 保存 JSON =============
output = {
    'meta': {
        'target_period': TARGET,
        'prior_period': '2026202',
        'prior_numbers': PRIOR,
        'created_at': '2026-08-01 15:55 GMT+8',
        'author': 'Lucky / MiniMax-M3',
        'method': 'V3: 严格 20 胆基准 + 4 战咯纯评分驱动',
        'constraint': '20 胆 4 区位 5/5/5/5; N 胆 (4/6/10) 都是 20 胆中评分最高的 N 个, 不受区位/跨度/连号约束',
        'philosophy': '评分驱动 > 形态约束, 胆码从 20 胆中取前 N 高分号',
        'constraint_changes': {
            'V1': 'N 胆独立选, 有 4 区位配额',
            'V2': 'N 胆从 20 胆派, 仍有 4 区位配额',
            'V3 (current)': 'N 胆从 20 胆派, 纯评分驱动, 无区位/跨度/连号约束',
        },
        '20_dan_zones': [5, 5, 5, 5],
    },
    'predictions': {
        name: {
            'method': r['method'],
            '20_dan': r['20_dan'],
            '4_dan': r['4_dan'],
            '6_dan': r['6_dan'],
            '10_dan': r['10_dan'],
        }
        for name, r in results.items()
    }
}

with open(ROOT / 'data/backtest/2026203_predictions_v3_score_only.json', 'w', encoding='utf-8') as f:
    json.dump(output, f, ensure_ascii=False, indent=2)
print(f"\n✅ 已保存: {ROOT}/data/backtest/2026203_predictions_v3_score_only.json")
