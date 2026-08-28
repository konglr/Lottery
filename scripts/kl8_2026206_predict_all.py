"""
KL8 2026206 期 全模型预测
=============================================================

集成模型:
  V22 主线: V22.A / V22.E / V22.F
  V22.A 子方案: V22.A1 / V22.A2 / V22.A3 / V22.A4
  V18 系列: V18.C
  V21 系列: V21.A
  Consensus (跨模型投票)

选号规则 (V3 风格):
  1. 每个模型先选 20 胆 (各模型独立选号规则)
  2. 4 胆 / 6 胆 / 10 胆 = 20 胆中评分最高的 N 个 (纯评分驱动)
  3. 无 4 区位配额 / 跨度 / 连号约束 (小胆码只看分数)

数据状态: 2026205 为最新已开奖期 (上期)
"""
import sys
from pathlib import Path
import json
from datetime import datetime
import numpy as np
from collections import Counter

ROOT = Path.home() / 'Library/Mobile Documents/com~apple~CloudDocs/PycharmProjects/Lottery'
sys.path.insert(0, str(ROOT))
sys.path.insert(0, '/Users/clarkkong/.openclaw/workspace/agents/lucky')
from lottery_data import LotteryData

ld = LotteryData(ROOT)
df, conf = ld.load('快乐8')
red_cols = [f'红球{i}' for i in range(1, 21)]
data = df[red_cols].astype(int).values
periods = df['期号'].tolist()

# 2026206 期: 上期 = 2026205
PRIOR_PERIOD = '2026205'
TARGET = '2026206'
last_i = periods.index(PRIOR_PERIOD)

PRIOR = list(map(int, df[df['期号'] == PRIOR_PERIOD].iloc[0]['frontWinningNum'].split()))

print(f"预测目标: {TARGET}")
print(f"上期 {PRIOR_PERIOD} 开奖: {PRIOR}")
print(f"数据总期数: {len(df)}, last_i={last_i}")


# ============= 工具函数 =============
def zone_of(num):
    if 1 <= num <= 20: return 0
    if 21 <= num <= 40: return 1
    if 41 <= num <= 60: return 2
    return 3


def max_consecutive_in(nums):
    if len(nums) < 2: return 1
    s = sorted(nums)
    max_run = 1; current = 1
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


# ============= V22.A / V22.E / V22.F 评分 =============
def score_v22_a(last_i):
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


def score_v22_e(last_i):
    """V22.E: c5 主权重 + c10/c30 减权"""
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
    s[c5 == 0] = 0.5; s[c5 == 1] = 0.5; s[c5 == 2] = 0.6
    s[c5 == 3] = -0.5
    s[(c5 >= 4) & (c5 <= 5)] = -1.5
    s[c5 >= 6] = -3.0
    
    t = np.zeros(80)
    t[(c10 >= 2) & (c10 <= 4)] = 0.3
    t[c10 == 1] = 0.12; t[c10 == 0] = 0.06
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


def score_v22_f(last_i):
    """V22.F: 偏热号 (c5=2-3 加分)"""
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
    s[c5 == 0] = -0.5; s[c5 == 1] = -0.3
    s[c5 == 2] = 0.8; s[c5 == 3] = 0.7
    s[(c5 >= 4) & (c5 <= 5)] = 0.3
    s[c5 >= 6] = -1.5
    
    t = np.zeros(80)
    t[(c10 >= 2) & (c10 <= 4)] = 1.0
    t[c10 == 1] = 0.4; t[c10 == 0] = 0.2
    t[(c10 >= 5) & (c10 <= 6)] = -0.3
    t[c10 >= 7] = -0.8
    
    u = np.zeros(80)
    u[(c30 >= 5) & (c30 <= 10)] = 0.5
    u[(c30 >= 3) & (c30 <= 4)] = 0.2
    u[c30 <= 2] = -0.2
    u[c30 >= 13] = -0.7
    
    return s + t + u


# ============= V22.A1-A4 评分 (方案 F 变种) =============
def score_v22_a1(last_i):
    """V22.A1 (方案 F): c2_hot=0.6, c3_hot=0.4"""
    base = score_v22_a(last_i)
    c2 = np.zeros(80, dtype=int)
    for k in range(1, 3):  # 近 2 期
        for n in data[last_i + k]:
            c2[n-1] += 1
    c3 = np.zeros(80, dtype=int)
    for k in range(1, 4):  # 近 3 期
        for n in data[last_i + k]:
            c3[n-1] += 1
    c5 = np.zeros(80, dtype=int)
    for k in range(1, 6):
        for n in data[last_i + k]:
            c5[n-1] += 1
    
    s = base.copy()
    s[c2 == 1] += 0.6
    s[c3 == 1] += 0.4
    s[(c5 >= 3) & (c5 <= 4)] -= 0.3
    s[c5 >= 5] -= 0.5
    return s


# ============= V18.C 反向基线 =============
def score_v18c(last_i):
    """V18.C: 反向特征 (冷号优先)"""
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


# ============= V21.A 评分 (V20 + 30% 热号混入) =============
def score_v21_a(last_i):
    """V21.A: V20 多窗口频次 + 热号加权"""
    # V20 评分 = 多窗口频次
    c5 = np.zeros(80, dtype=int)
    for k in range(1, 6):
        for n in data[last_i + k]:
            c5[n-1] += 1
    c10 = np.zeros(80, dtype=int)
    for k in range(1, 11):
        for n in data[last_i + k]:
            c10[n-1] += 1
    c20 = np.zeros(80, dtype=int)
    for k in range(1, 21):
        for n in data[last_i + k]:
            c20[n-1] += 1
    
    s = np.zeros(80)
    # V20 基线
    s += c5 * 0.3 + c10 * 0.2 + c20 * 0.1
    # V21 热号加权 (c5 >= 2 加分)
    s[c5 == 2] += 0.3
    s[c5 == 3] += 0.5
    s[(c5 >= 4) & (c5 <= 5)] += 0.2
    s[c5 >= 6] -= 0.5
    return s


# ============= 选号工具 =============
def select_20_strict_5perzone(scores, last_i):
    """V22.A / V22.E: 严格 4 区位 5/5/5/5"""
    sorted_idx = np.argsort(scores)[::-1]
    selected = []
    zone_count = [0, 0, 0, 0]
    zone_quota = [5, 5, 5, 5]
    for z in range(4):
        for idx in sorted_idx:
            if zone_count[z] >= zone_quota[z]: break
            num = int(idx) + 1
            if num in selected: continue
            if zone_of(num) != z: continue
            candidate = selected + [num]
            if len(candidate) >= 3 and max_consecutive_in(candidate) > 4:
                continue
            if len(candidate) >= 4 and (max(candidate) - min(candidate)) > 78:
                continue
            zone_count[z] += 1
            selected.append(num)
            if len(selected) >= 20:
                return sorted(selected), zone_count
    return sorted(selected), zone_count


def select_20_relaxed(scores, last_i, lo=3, hi=8):
    """V22.A2 / V22.A4: 4 区位放宽 3-8"""
    sorted_idx = np.argsort(scores)[::-1]
    selected = []
    zone_count = [0, 0, 0, 0]
    for idx in sorted_idx:
        if len(selected) >= 20: break
        num = int(idx) + 1
        if num in selected: continue
        z = zone_of(num)
        if zone_count[z] >= hi: continue
        # 检查 4 区位最低
        if len(selected) >= 20 - 4 * lo:
            zones_left = [hi - zone_count[zi] for zi in range(4)]
            if max(zones_left) == 0: continue
            # 还有至少 lo 个号要填
            remaining = 20 - len(selected)
            min_needed = sum(max(0, lo - zc) for zc in zone_count)
            if min_needed > remaining:
                # 必须填到 lo
                if zone_count[z] >= lo: continue
        candidate = selected + [num]
        if len(candidate) >= 3 and max_consecutive_in(candidate) > 4: continue
        if len(candidate) >= 4 and (max(candidate) - min(candidate)) > 78: continue
        zone_count[z] += 1
        selected.append(num)
    return sorted(selected), zone_count


def select_20_a4(scores, last_i):
    """V22.A4: 4 区位 3-8 + lag1-5 替换约束"""
    base_20, base_zones = select_20_relaxed(scores, last_i)
    sorted_idx = np.argsort(scores)[::-1]
    
    # 对 lag1-5 逐一替换
    thresholds = {1: 10, 2: 10, 3: 5, 4: 5, 5: 5}
    cur = list(base_20)
    for lag, th in thresholds.items():
        if last_i + lag >= len(data): continue
        lag_set = set(data[last_i + lag])
        overlap = len(set(cur) & lag_set)
        if overlap < th: continue
        # 替换末尾 2 个
        # 找 cur 里与 lag 重的号,按评分升序排序
        cur_with_score = [(n, scores[n-1]) for n in cur]
        cur_with_score.sort(key=lambda x: x[1])
        to_remove = [n for n, s in cur_with_score if n in lag_set][:2]
        if len(to_remove) < 2:
            # 不够 2 个重的,就取评分最低的 2 个
            to_remove = [n for n, s in cur_with_score[:2]]
        for n in to_remove:
            cur.remove(n)
        # 补 2 个新号(从 sorted_idx 后续中找不与 lag 重的,4 区位 ≤ 8)
        cur_set = set(cur)
        cur_zones = [0, 0, 0, 0]
        for n in cur:
            cur_zones[zone_of(n)] += 1
        added = 0
        for idx in sorted_idx:
            if added >= 2: break
            num = int(idx) + 1
            if num in cur_set or num in lag_set: continue
            if cur_zones[zone_of(num)] >= 8: continue
            cur.append(num)
            cur_set.add(num)
            cur_zones[zone_of(num)] += 1
            added += 1
    return sorted(cur), [0,0,0,0]


def derive_n(twenty_picks, scores, top_n):
    """从 20 胆中按分数取前 N 个 (V3 风格,纯评分)"""
    sorted_20 = sorted(twenty_picks, key=lambda n: scores[n-1], reverse=True)
    selected = sorted_20[:top_n]
    return sorted(selected), [0,0,0,0]


# ============= 主预测流程 =============
strategies = [
    ('V22.A', score_v22_a, select_20_strict_5perzone, 'P90 + 4 区位配额 + 邻号'),
    ('V22.E', score_v22_e, select_20_strict_5perzone, 'c5 主权重 + c10/c30 减权'),
    ('V22.F', score_v22_f, select_20_strict_5perzone, '偏热号 c5=2-3 加分'),
    ('V22.A1', score_v22_a1, select_20_strict_5perzone, 'V22.A + 方案 F (c2=0.6, c3=0.4) + 严格 5/5/5/5'),
    ('V22.A2', score_v22_a1, select_20_relaxed, 'V22.A1 + 4 区位放宽 3-8'),
    ('V22.A4', score_v22_a1, select_20_a4, 'V22.A2 + lag1-5 替换 (阈值 10/10/5/5/5)'),
    ('V18.C', score_v18c, select_20_relaxed, '反向基线 (冷号优先)'),
    ('V21.A', score_v21_a, select_20_relaxed, 'V20 多窗口 + 30% 热号混入'),
]

results = {}

print("\n" + "=" * 70)
print("🤖 KL8 2026206 期 多模型预测")
print("=" * 70)

for name, scorer, selector, desc in strategies:
    scores = scorer(last_i)
    twenty, zones = selector(scores, last_i)
    
    picks = {'20_dan': twenty}
    for n in [4, 6, 10]:
        sub, _ = derive_n(twenty, scores, n)
        picks[f'{n}_dan'] = sub
    
    results[name] = {
        'method': desc,
        **picks,
        '20_dan_zones': zones,
    }
    
    print(f"\n--- {name} ({desc}) ---")
    print(f"  20 胆: {twenty} ({zones[0]}/{zones[1]}/{zones[2]}/{zones[3]})")
    print(f"  4 胆: {picks['4_dan']}")
    print(f"  6 胆: {picks['6_dan']}")
    print(f"  10 胆: {picks['10_dan']}")
    print(f"  跨度: {max(twenty)-min(twenty)}, 最大连号: {max_consecutive_in(twenty)}")


# ============= 共识 (Consensus) =============
print("\n" + "=" * 70)
print("🤝 模型共识 (跨所有 8 个模型)")
print("=" * 70)

consensus = {}
for n in [4, 6, 10, 20]:
    cnt = Counter()
    for name, r in results.items():
        for num in r[f'{n}_dan']:
            cnt[num] += 1
    
    consensus_3 = sorted([num for num, c in cnt.items() if c >= 3])
    consensus_5 = sorted([num for num, c in cnt.items() if c >= 5])
    consensus_6 = sorted([num for num, c in cnt.items() if c >= 6])
    
    consensus[f'{n}_dan_3plus'] = consensus_3
    consensus[f'{n}_dan_5plus'] = consensus_5
    consensus[f'{n}_dan_6plus'] = consensus_6
    
    print(f"\n{n} 胆:")
    print(f"  ≥3 模型共识 ({len(consensus_3)} 个): {consensus_3}")
    print(f"  ≥5 模型共识 ({len(consensus_5)} 个): {consensus_5}")
    print(f"  ≥6 模型共识 ({len(consensus_6)} 个): {consensus_6}")


# ============= 保存 JSON =============
output = {
    'meta': {
        'created_at': datetime.now().strftime('%Y-%m-%d %H:%M GMT+8'),
        'lottery': '快乐8',
        'target_period': TARGET,
        'prior_period': PRIOR_PERIOD,
        'prior_numbers': PRIOR,
        'open_time_predicted': '2026-08-04 21:30 (预计)',
        'method': '集成 8 模型: V22.A / V22.E / V22.F / V22.A1 / V22.A2 / V22.A4 / V18.C / V21.A',
        'data_periods': len(df),
        'author': 'Lucky / MiniMax-M3',
        'selection_rules': '20 胆 4 区位配额; 4/6/10 胆从 20 胆中评分前 N (纯评分驱动)',
    },
    'predictions': {
        name: {
            'method': r['method'],
            '4_dan': r['4_dan'],
            '6_dan': r['6_dan'],
            '10_dan': r['10_dan'],
            '20_dan': r['20_dan'],
        }
        for name, r in results.items()
    },
    'consensus_recommendation': consensus,
}

out_path = ROOT / 'data' / 'backtest' / f'{TARGET}_predictions_multi_model.json'
with open(out_path, 'w', encoding='utf-8') as f:
    json.dump(output, f, ensure_ascii=False, indent=2)

print(f"\n✅ 已保存: {out_path}")
print(f"\n📊 总览: 8 个模型 × 4 胆/6 胆/10 胆/20 胆 = 32 条独立预测")