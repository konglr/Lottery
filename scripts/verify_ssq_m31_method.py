"""
SSQ 预测方法 200 期回测 — 验证"形态反转加成"是否有价值
=====================================================

时间: 2026-09-30 12:15 GMT+8
作者: Lucky / MiniMax-M3.1-flash
背景: 2026113 期实战预测失败 (Top4 形态 0/0/4 全压 3 区)

对比 4 个版本 (200 期, 严格时间方向):
  V0: V22.A 基线 (c5/c10/c30/邻号)
  V1: V0 + FSW.SSQ1.F (span_3/5/10 窗口)
  V2: V0 + FSW + 形态反转加成  ← 2026113 实战用的方法
  V3: V0 + 形态平衡约束 (4 区位至少 1 个)

输出: Top 4 / 6 / 12 命中率对比 + 4 区位分布
"""

import sys
import numpy as np
from pathlib import Path

ROOT = Path.home() / 'Library/Mobile Documents/com~apple~CloudDocs/PycharmProjects/Lottery'
sys.path.insert(0, str(ROOT))
from lottery_data import LotteryData

ld = LotteryData(ROOT)
df, conf = ld.load('双色球')
red_cols = [f'红球{i}' for i in range(1, 7)]
data = df[red_cols].astype(int).values
n_periods = len(data)
periods = df['期号'].tolist()
print(f"数据: {n_periods} 期 (最新 {periods[0]})\n")


# ---------- 基础统计函数 ----------
def c5(li):
    c = np.zeros(33, dtype=int)
    for k in range(1, 6):
        if li + k < n_periods:
            for n in data[li + k]: c[n-1] += 1
    return c


def c10(li):
    c = np.zeros(33, dtype=int)
    for k in range(1, 11):
        if li + k < n_periods:
            for n in data[li + k]: c[n-1] += 1
    return c


def c30(li):
    c = np.zeros(33, dtype=int)
    for k in range(1, 31):
        if li + k < n_periods:
            for n in data[li + k]: c[n-1] += 1
    return c


def nbr(li):
    v = np.zeros(33)
    if li < n_periods:
        for num in data[li]:
            for d in [-2, -1, 1, 2]:
                if 1 <= num + d <= 33:
                    v[num + d - 1] += 0.3
    return v / 6


def zones(nums):
    return [sum(1 for n in nums if n <= 11),
            sum(1 for n in nums if 12 <= n <= 22),
            sum(1 for n in nums if n >= 23)]


# ---------- V22.A 基线评分 ----------
def score_base(li):
    c5v, c10v, c30v = c5(li), c10(li), c30(li)
    s = np.zeros(33)
    s[c5v == 0] = -0.5
    s[c5v == 1] = 0.0
    s[c5v == 2] = 0.2
    s[c5v == 3] = 0.2
    s[c5v == 4] = -0.3
    s[c5v == 5] = -0.8
    t = np.zeros(33)
    t[(c10v >= 2) & (c10v <= 4)] = 1.0
    t[c10v == 1] = 0.4
    t[c10v == 0] = 0.2
    t[(c10v >= 5) & (c10v <= 6)] = -0.3
    t[c10v >= 7] = -0.8
    u = np.zeros(33)
    u[(c30v >= 4) & (c30v <= 8)] = 0.5
    u[(c30v >= 2) & (c30v <= 3)] = 0.2
    u[c30v <= 1] = -0.2
    u[c30v >= 11] = -0.7
    return s + t + u + nbr(li)


# ---------- FSW.SSQ1.F: span_3 + span_5 + span_10 ----------
def score_fsw(li, w=(0.5, 0.3, 0.3)):
    out = np.zeros(33)
    for span, weight in zip((3, 5, 10), w):
        c = np.zeros(33)
        for k in range(1, span + 1):
            if li + k < n_periods:
                for n in data[li + k]: c[n-1] += 1
        out += c / span * weight
    return out


# ---------- 形态反转加成 (2026113 实战用的) ----------
def reversal_boost(li, amount=0.3):
    """上一期 1 区 >= 3 个 -> 中大号加分 (实战用的逻辑)"""
    z = zones(data[li])
    boost = np.zeros(33)
    if z[0] >= 3:
        for n in range(12, 34):
            boost[n - 1] += amount
    return boost


# ---------- 形态平衡选号 (4 区位至少 1 个) ----------
def select_balanced(scores, n):
    """按评分排序, 但保证 4 区位每个区至少 1 个"""
    order = np.argsort(scores)[::-1]
    picked = []
    zc = [0, 0, 0, 0]
    # Step 1: 每区先取评分最高的 1 个
    for z in range(3):
        for idx in order:
            num = int(idx) + 1
            if zc[z] > 0:
                break
            if num in picked:
                continue
            if (num - 1) // 11 != z:
                continue
            picked.append(num)
            zc[z] += 1
    # Step 2: 补足到 n
    for idx in order:
        if len(picked) >= n:
            break
        num = int(idx) + 1
        if num in picked:
            continue
        picked.append(num)
    return sorted(picked, key=lambda x: scores[x - 1], reverse=True)


def select_plain(scores, n):
    return sorted((np.argsort(scores)[::-1][:n] + 1).tolist())


# ============================================================
# 200 期回测
# ============================================================
N_TEST = 200
# 数据降序: data[0] 最新
# last_i = i+1 (上一期), actual = data[i]
START = 0
END = min(N_TEST, n_periods - 40)
n_test = END - START
assert n_test == N_TEST, f"实际 {n_test} 期"

versions = ['V0 基线', 'V1 +FSW', 'V2 +FSW+反转', 'V3 形态平衡']
hits = {v: {4: 0, 6: 0, 12: 0} for v in versions}
zone_dist = {v: [] for v in versions}

for i in range(START, END):
    last_i = i + 1
    actual = set(data[i].tolist())

    s0 = score_base(last_i)
    s1 = s0 + score_fsw(last_i) * 0.4
    s2 = s1 + reversal_boost(last_i, 0.3)
    s3 = s0  # 形态平衡用同一评分, 只改选号方式

    for tn in [4, 6, 12]:
        sel0 = select_plain(s0, tn)
        sel1 = select_plain(s1, tn)
        sel2 = select_plain(s2, tn)
        sel3 = select_balanced(s3, tn)

        hits['V0 基线'][tn] += len(set(sel0) & actual)
        hits['V1 +FSW'][tn] += len(set(sel1) & actual)
        hits['V2 +FSW+反转'][tn] += len(set(sel2) & actual)
        hits['V3 形态平衡'][tn] += len(set(sel3) & actual)

    zone_dist['V0 基线'].append(zones(select_plain(s0, 12)))
    zone_dist['V2 +FSW+反转'].append(zones(select_plain(s2, 12)))
    zone_dist['V3 形态平衡'].append(zones(select_balanced(s3, 12)))

print("=" * 72)
print(f"SSQ {n_test} 期回测 (期号 {periods[START]} → {periods[END-1]})")
print("=" * 72)
print(f"{'版本':<18}{'Top 4':<14}{'Top 6':<14}{'Top 12'}")
print("-" * 72)
for v in versions:
    row = f"{v:<18}"
    for tn in [4, 6, 12]:
        pct = hits[v][tn] / (n_test * tn) * 100
        row += f"{pct:>6.2f}%       "
    print(row)

print(f"\n基线 (随机): 18.18%")

# 4 区位分布 (Top 12)
print("\n" + "=" * 72)
print("Top 12 的 4 区位分布 (近 3 期) — 看形态是否极端")
print("=" * 72)
for v in ['V0 基线', 'V2 +FSW+反转', 'V3 形态平衡']:
    z = np.mean(zone_dist[v], axis=0)
    print(f"{v:<18} 1区 {z[0]:.2f}  2区 {z[1]:.2f}  3区 {z[2]:.2f}  (理论 2.00/2.00/2.00)")

# 实际开奖的 4 区位分布
act_z = [zones(data[i]) for i in range(n_test)]
az = np.mean(act_z, axis=0)
print(f"{'实际开奖':<18} 1区 {az[0]:.2f}  2区 {az[1]:.2f}  3区 {az[2]:.2f}")

# 统计显著性
print("\n" + "=" * 72)
print("统计显著性 (vs 18.18% 基线)")
print("=" * 72)
for v in versions:
    for tn in [4, 6]:
        h = hits[v][tn]
        expected = n_test * tn * 6 / 33
        var = n_test * tn * (6/33) * (1 - 6/33)
        z = (h - expected) / (var ** 0.5)
        sig = "显著" if abs(z) > 1.96 else "不显著"
        print(f"{v:<18} Top {tn:<3} 命中 {h:>4}  期望 {expected:>6.1f}  z={z:+.2f}  {sig}")
