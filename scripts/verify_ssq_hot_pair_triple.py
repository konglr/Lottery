"""
SSQ 热门号码对 + 三元组挖掘 — 验证脚本
========================================

改进时间: 2026-09-18 13:45 GMT+8
作者: Lucky / MiniMax-M3 (执行任务)
依据: 用户请求 "在双色球选号策略中, 增加热门号码对 + 热门三元组"

数据: 3505 期 SSQ (2003001 - 2026108)
红球: 6 个 / 1-33

核心思路:
  1. 计算每个 (a,b) 对的 z-score (相对独立假设)
  2. 计算每个 (a,b,c) 三元组的 z-score
  3. 时间加权: 近 500 期权重高
  4. 把"参与高 z 对/三元组"的号叠加到 V22.A 基线评分
  5. 验证新评分 vs V22.A 基线 vs 随机

⚠️ 注意: V22.A 在 SSQ 上不直接叫 V22.A, 是 SSQ-FSW
   这里用 V22.A 的精神 (c5/c10/c30/邻号) 作为基线, 新增 "对/三元组" 项
"""

import sys
import numpy as np
from pathlib import Path
from itertools import combinations
from collections import Counter

ROOT = Path.home() / 'Library/Mobile Documents/com~apple~CloudDocs/PycharmProjects/Lottery'
sys.path.insert(0, str(ROOT))
from lottery_data import LotteryData

ld = LotteryData(ROOT)
df, conf = ld.load('双色球')
red_cols = [f'红球{i}' for i in range(1, 7)]
data = df[red_cols].astype(int).values  # shape: (n_periods, 6)
n_periods = len(data)
periods = df['期号'].tolist()
print(f"数据: {n_periods} 期 (最新 {periods[0]} → 最老 {periods[-1]})\n")

N_RED = 33
N_PER_DRAW = 6


# ============================================================
# 1. 频率统计 + z-score (全期)
# ============================================================
def count_pairs(data):
    """统计所有 (a,b) 对的出现次数, a<b"""
    counter = Counter()
    for row in data:
        for a, b in combinations(sorted(row.tolist()), 2):
            counter[(a, b)] += 1
    return counter


def count_triples(data):
    """统计所有 (a,b,c) 三元组的出现次数, a<b<c"""
    counter = Counter()
    for row in data:
        for a, b, c in combinations(sorted(row.tolist()), 3):
            counter[(a, b, c)] += 1
    return counter


print("="*70)
print("1. 全期频率统计 + z-score")
print("="*70)

pair_counter = count_pairs(data)
triple_counter = count_triples(data)

# 期望基线: 独立假设下, 一对 (a,b) 出现的期望次数
# E = N × C(6,2) / C(33,2) = N × 15/528 ≈ N × 0.0181 (对所有对都一样)
n_pairs_total = N_PER_DRAW * (N_PER_DRAW - 1) // 2  # = 15
n_pairs_possible = N_RED * (N_RED - 1) // 2  # = 528
E_pair = n_periods * n_pairs_total / n_pairs_possible
Var_pair = E_pair * (1 - n_pairs_total / n_pairs_possible)

n_triples_total = N_PER_DRAW * (N_PER_DRAW - 1) * (N_PER_DRAW - 2) // 6  # = 20
n_triples_possible = N_RED * (N_RED - 1) * (N_RED - 2) // 6  # = 5456
E_triple = n_periods * n_triples_total / n_triples_possible

print(f"\n对基线:")
print(f"  总期数 N = {n_periods}")
print(f"  每期对数 C(6,2) = {n_pairs_total}")
print(f"  全部可能对数 C(33,2) = {n_pairs_possible}")
print(f"  每对期望次数 E_pair = {E_pair:.2f}")
print(f"  每对标准差 = {Var_pair**0.5:.2f}")

print(f"\n三元组基线:")
print(f"  每期三元组数 C(6,3) = {n_triples_total}")
print(f"  全部可能三元组 C(33,3) = {n_triples_possible}")
print(f"  每三元组期望次数 E_triple = {E_triple:.2f}")


# 计算 z-score (对所有对/三元组)
pair_z = {pair: (count - E_pair) / (Var_pair ** 0.5)
          for pair, count in pair_counter.items()}
triple_z = {triple: (count - E_triple) / (E_triple ** 0.5)
            for triple, count in triple_counter.items()}

# 排序: 找最显著的对/三元组
top_pairs = sorted(pair_z.items(), key=lambda x: -x[1])[:15]
top_triples = sorted(triple_z.items(), key=lambda x: -x[1])[:15]

print(f"\n🔥 最高 z-score 的 15 个对 (全期):")
for (a, b), z in top_pairs[:15]:
    actual = pair_counter[(a, b)]
    print(f"  ({a:2d}, {b:2d})  实际 {actual:3d} 期望 {E_pair:.1f}  z={z:+.2f}")

print(f"\n🔥 最高 z-score 的 15 个三元组 (全期):")
for (a, b, c), z in top_triples[:15]:
    actual = triple_counter[(a, b, c)]
    print(f"  ({a:2d}, {b:2d}, {c:2d})  实际 {actual:3d} 期望 {E_triple:.1f}  z={z:+.2f}")

# 多少对/三元组显著 (|z| > 2)?
n_sig_pair = sum(1 for z in pair_z.values() if z > 2)
n_sig_triple = sum(1 for z in triple_z.values() if z > 2)
print(f"\n显著对 (z>2): {n_sig_pair} / {len(pair_z)}")
print(f"显著三元组 (z>2): {n_sig_triple} / {len(triple_z)}")


# ============================================================
# 2. 时间加权 z-score (近 500 期)
# ============================================================
print("\n" + "="*70)
print("2. 时间加权 z-score (近 500 期, EMA 半衰期 200)")
print("="*70)

WINDOW = 500  # 用近 500 期算 z-score

# 用最近 500 期算 z-score
recent_data = data[:WINDOW]
recent_pair_counter = count_pairs(recent_data)
recent_triple_counter = count_triples(recent_data)

E_pair_recent = WINDOW * n_pairs_total / n_pairs_possible
Var_pair_recent = E_pair_recent * (1 - n_pairs_total / n_pairs_possible)
E_triple_recent = WINDOW * n_triples_total / n_triples_possible

recent_pair_z = {pair: (count - E_pair_recent) / (Var_pair_recent ** 0.5)
                 for pair, count in recent_pair_counter.items()}
recent_triple_z = {triple: (count - E_triple_recent) / (E_triple_recent ** 0.5)
                   for triple, count in recent_triple_counter.items()}

top_recent_pairs = sorted(recent_pair_z.items(), key=lambda x: -x[1])[:10]
top_recent_triples = sorted(recent_triple_z.items(), key=lambda x: -x[1])[:10]

print(f"\n🔥 近 500 期最高 z-score 的 10 个对:")
for (a, b), z in top_recent_pairs:
    actual = recent_pair_counter[(a, b)]
    print(f"  ({a:2d}, {b:2d})  实际 {actual:3d} 期望 {E_pair_recent:.1f}  z={z:+.2f}")

print(f"\n🔥 近 500 期最高 z-score 的 10 个三元组:")
for (a, b, c), z in top_recent_triples:
    actual = recent_triple_counter[(a, b, c)]
    print(f"  ({a:2d}, {b:2d}, {c:2d})  实际 {actual:3d} 期望 {E_triple_recent:.1f}  z={z:+.2f}")


# ============================================================
# 3. 评分函数: V22.A 基线 + 对/三元组加分
# ============================================================
def c5_ssq(last_i):
    """SSQ c5: 最近 5 期每个号出现次数 (0-5)"""
    c = np.zeros(N_RED, dtype=int)
    for k in range(1, 6):
        if last_i + k < n_periods:
            for n in data[last_i + k]: c[n-1] += 1
    return c


def c10_ssq(last_i):
    c = np.zeros(N_RED, dtype=int)
    for k in range(1, 11):
        if last_i + k < n_periods:
            for n in data[last_i + k]: c[n-1] += 1
    return c


def c30_ssq(last_i):
    c = np.zeros(N_RED, dtype=int)
    for k in range(1, 31):
        if last_i + k < n_periods:
            for n in data[last_i + k]: c[n-1] += 1
    return c


def nbr_ssq(last_i):
    """邻号: 上期 ±1, ±2"""
    nbr = np.zeros(N_RED)
    if last_i < n_periods:
        for num in data[last_i]:
            for d in [-2, -1, 1, 2]:
                if 1 <= num+d <= N_RED:
                    nbr[num+d-1] += 0.3
    nbr /= 6  # 归一化
    return nbr


def baseline_ssq_score(last_i):
    """V22.A 精神基线 (c5/c10/c30/邻号), 33 号版本"""
    c5 = c5_ssq(last_i)
    s = np.zeros(N_RED)
    s[c5 == 0] = -0.5
    s[c5 == 1] = 0.0
    s[c5 == 2] = +0.2
    s[c5 == 3] = +0.2
    s[c5 == 4] = -0.3
    s[c5 == 5] = -0.8

    c10 = c10_ssq(last_i)
    t = np.zeros(N_RED)
    t[(c10 >= 2) & (c10 <= 4)] = 1.0
    t[c10 == 1] = 0.4
    t[c10 == 0] = 0.2
    t[(c10 >= 5) & (c10 <= 6)] = -0.3
    t[c10 >= 7] = -0.8

    c30 = c30_ssq(last_i)
    u = np.zeros(N_RED)
    u[(c30 >= 4) & (c30 <= 8)] = 0.5
    u[(c30 >= 2) & (c30 <= 3)] = 0.2
    u[c30 <= 1] = -0.2
    u[c30 >= 11] = -0.7

    return s + t + u + nbr_ssq(last_i)


def pair_score(last_i, window=WINDOW):
    """
    对每个号 n, 计算它参与的"高 z 对"的总加分
    公式: score[n] = sum_{(n, m) in top_pairs} max(0, z(n,m) - 2)
    """
    # 重新算近 window 期的对 z-score
    start = max(0, last_i + 1)
    end = min(n_periods, last_i + window + 1)
    recent = data[start:end]
    if len(recent) < 30:
        return np.zeros(N_RED), {}

    cnt = count_pairs(recent)
    n = len(recent)
    E = n * n_pairs_total / n_pairs_possible
    V = E * (1 - n_pairs_total / n_pairs_possible)

    z = {pair: (c - E) / (V ** 0.5) for pair, c in cnt.items()}

    # 对每个号, 累计它参与的显著对的 z (z > 2)
    score = np.zeros(N_RED)
    pair_contributions = {n: [] for n in range(1, N_RED + 1)}
    for (a, b), zval in z.items():
        if zval > 2.0:
            score[a-1] += zval - 2.0
            score[b-1] += zval - 2.0
            pair_contributions[a].append((b, zval))
            pair_contributions[b].append((a, zval))
    return score, pair_contributions


def triple_score(last_i, window=WINDOW, z_thresh=2.0):
    """
    对每个号 n, 计算它参与的"高 z 三元组"的总加分
    """
    start = max(0, last_i + 1)
    end = min(n_periods, last_i + window + 1)
    recent = data[start:end]
    if len(recent) < 30:
        return np.zeros(N_RED), {}

    cnt = count_triples(recent)
    n = len(recent)
    E = n * n_triples_total / n_triples_possible

    z = {triple: (c - E) / (E ** 0.5) for triple, c in cnt.items()}

    score = np.zeros(N_RED)
    triple_contributions = {n: [] for n in range(1, N_RED + 1)}
    for (a, b, c), zval in z.items():
        if zval > z_thresh:
            for n_ in [a, b, c]:
                score[n_-1] += zval - z_thresh
                triple_contributions[n_].append(((a, b, c), zval))
    return score, triple_contributions


# ============================================================
# 4. 回测: 多种方案对比
# ============================================================
print("\n" + "="*70)
print("3. 回测: 多种方案对比 (滚动窗口 200 期)")
print("="*70)

N_TEST = 200
# 数据降序: data[0] 最新, data[n-1] 最老
# last_i = i+1, actual = data[i]
# 最近 200 期 = i in [0, 200)
START = 0
END = min(N_TEST, n_periods - 30)

assert END - START == N_TEST, f"预期 {N_TEST}, got {END - START}"

print(f"回测范围: i = {START}..{END-1} (期号 {periods[START]} → {periods[END-1]})\n")


def select_top_n_ssq(scores, n):
    return (np.argsort(scores)[::-1][:n] + 1).tolist()


# 收集结果
results = {
    'V0: V22.A 基线':            [],
    'V1: + 热门对 (w=0.1)':       [],
    'V2: + 热门对 (w=0.3)':       [],
    'V3: + 热门三元组 (w=0.1)':   [],
    'V4: + 对 + 三元组 (w=0.1/0.1)': [],
    'V5: + 对 + 三元组 (w=0.3/0.3)': [],
}

# Top N 测试点 (SSQ 主推就是 6 胆)
TOP_NS = [4, 6, 8, 12]

# 初始化命中计数器
hits = {v_name: {n: 0 for n in TOP_NS} for v_name in results}

for i in range(START, END):
    last_i = i + 1
    actual = set(data[i].tolist())  # 当前期 6 个号

    # V0: 基线
    s0 = baseline_ssq_score(last_i)

    # V1: + 对 (w=0.1)
    ps1, _ = pair_score(last_i)
    s1 = s0 + ps1 * 0.1

    # V2: + 对 (w=0.3)
    ps2, _ = pair_score(last_i)
    s2 = s0 + ps2 * 0.3

    # V3: + 三元组 (w=0.1)
    ts3, _ = triple_score(last_i, z_thresh=2.0)
    s3 = s0 + ts3 * 0.1

    # V4: + 对 (0.1) + 三元组 (0.1)
    ps4, _ = pair_score(last_i)
    ts4, _ = triple_score(last_i, z_thresh=2.0)
    s4 = s0 + ps4 * 0.1 + ts4 * 0.1

    # V5: + 对 (0.3) + 三元组 (0.3)
    ps5, _ = pair_score(last_i)
    ts5, _ = triple_score(last_i, z_thresh=2.0)
    s5 = s0 + ps5 * 0.3 + ts5 * 0.3

    score_map = {
        'V0: V22.A 基线':            s0,
        'V1: + 热门对 (w=0.1)':       s1,
        'V2: + 热门对 (w=0.3)':       s2,
        'V3: + 热门三元组 (w=0.1)':   s3,
        'V4: + 对 + 三元组 (w=0.1/0.1)': s4,
        'V5: + 对 + 三元组 (w=0.3/0.3)': s5,
    }

    for v_name, scores in score_map.items():
        for n in TOP_NS:
            top_n = set(select_top_n_ssq(scores, n))
            hits[v_name][n] += len(top_n & actual)


# 输出结果
print(f"{'版本':<40}{'Top 4':<12}{'Top 6':<12}{'Top 8':<12}{'Top 12'}")
print("-"*85)

# 排序: 按 Top 6 命中率从高到低
sorted_versions = sorted(results.keys(),
                         key=lambda v: -hits[v][6] / (N_TEST * 6))

for rank, v_name in enumerate(sorted_versions, 1):
    row = f"#{rank} {v_name}"
    for n in TOP_NS:
        pct = hits[v_name][n] / (N_TEST * n) * 100
        row += f"{pct:>6.2f}%      "
    print(row)

# V0 在最后单独打一遍 (对照)
v0_pct = hits['V0: V22.A 基线'][6] / (N_TEST * 6) * 100
print(f"\n基线 V0 Top 6 命中率: {v0_pct:.2f}%")
print(f"基线理论: 6/33 = 18.18% (随机)")
print(f"统计期数: {N_TEST} 期")


# ============================================================
# 5. 最佳方案的贡献详情
# ============================================================
best_v = sorted_versions[0]
print(f"\n{'='*70}")
print(f"最佳方案: {best_v}")
print("="*70)

# 看最佳方案在某期的具体选号示例 (最近一期)
i = 0  # 最近一期
last_i = i + 1
actual = data[i].tolist()
print(f"\n最近一期 (i=0, {periods[i]}, 实际={sorted(actual)}):")

if best_v == 'V0: V22.A 基线':
    scores = baseline_ssq_score(last_i)
elif best_v == 'V1: + 热门对 (w=0.1)':
    s0 = baseline_ssq_score(last_i)
    ps, _ = pair_score(last_i)
    scores = s0 + ps * 0.1
elif best_v == 'V2: + 热门对 (w=0.3)':
    s0 = baseline_ssq_score(last_i)
    ps, _ = pair_score(last_i)
    scores = s0 + ps * 0.3
elif best_v == 'V3: + 热门三元组 (w=0.1)':
    s0 = baseline_ssq_score(last_i)
    ts, _ = triple_score(last_i)
    scores = s0 + ts * 0.1
elif best_v == 'V4: + 对 + 三元组 (w=0.1/0.1)':
    s0 = baseline_ssq_score(last_i)
    ps, _ = pair_score(last_i)
    ts, _ = triple_score(last_i)
    scores = s0 + ps * 0.1 + ts * 0.1
elif best_v == 'V5: + 对 + 三元组 (w=0.3/0.3)':
    s0 = baseline_ssq_score(last_i)
    ps, _ = pair_score(last_i)
    ts, _ = triple_score(last_i)
    scores = s0 + ps * 0.3 + ts * 0.3

top_6 = select_top_n_ssq(scores, 6)
print(f"  Top 6 (主推): {sorted(top_6)}")
print(f"  命中: {len(set(top_6) & set(actual))}/6 = {[n for n in top_6 if n in actual]}")