"""
P1 验证: V22.A6 主脚本 bug 修复后, Top 4 命中率是否回归 27%+

数据: 快乐8 全量
时间: 2026-09-11 14:00 GMT+8
作者: Lucky / MiniMax-M3

回测范围: 跳过前 50 期预热, 在剩余数据上滑动测试最后 200 期
"""

import sys
import numpy as np
from pathlib import Path

ROOT = Path.home() / 'Library/Mobile Documents/com~apple~CloudDocs/PycharmProjects/Lottery'
sys.path.insert(0, str(ROOT))
from lottery_data import LotteryData

ld = LotteryData(ROOT)
df, conf = ld.load('快乐8')
red_cols = [f'红球{i}' for i in range(1, 21)]
data = df[red_cols].astype(int).values
n_periods = len(data)

# ============= 复用 V22.A6 评分函数 =============
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


def score_v22_a6_fixed(last_i, lookback=3, scale=2.0):
    """修复后版本: actual_i = last_i + i (数据降序, +i 是过去期)"""
    c5, c10, c30 = compute_c5_c10_c30(last_i)
    s = np.zeros(80)
    s[c5 == 0] = -0.5
    s[c5 == 1] = 0.0
    s[c5 == 2] = 0.2
    s[c5 == 3] = 0.2
    s[c5 == 4] = -0.3
    s[c5 == 5] = -0.8
    s[c5 >= 6] = -1.5

    observed_counter = {}
    observed_total = 0
    for i in range(1, lookback + 1):
        actual_i = last_i + i  # ✅ 修复: 看过去期
        if actual_i < n_periods:
            c5_obs = np.zeros(80, dtype=int)
            for k in range(1, 6):
                if actual_i + k < n_periods:
                    for n in data[actual_i + k]: c5_obs[n-1] += 1
            for n in data[actual_i]:
                cnt = int(c5_obs[n-1])
                observed_counter[cnt] = observed_counter.get(cnt, 0) + 1
                observed_total += 1
    observed = {k: observed_counter.get(k, 0) / observed_total if observed_total > 0 else 0
                for k in range(6)}

    for k in range(6):
        adj = (observed.get(k, 0) - BASELINE_C5.get(k, 0)) * scale
        s[c5 == k] += adj

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

    return s + t + u + compute_nbr(last_i)


def select_top_n(scores, n):
    return (np.argsort(scores)[::-1][:n] + 1).tolist()


# ============= 计算 baseline (全部 80 号的 c5 平均分布) =============
print("预计算 baseline c5 分布...")
baseline_counter = {}
baseline_total = 0
for li in range(n_periods):
    if li + 5 < n_periods:
        c5 = np.zeros(80, dtype=int)
        for k in range(1, 6):
            for n in data[li + k]: c5[n-1] += 1
        for c in c5:
            baseline_counter[int(c)] = baseline_counter.get(int(c), 0) + 1
            baseline_total += 1
BASELINE_C5 = {k: baseline_counter.get(k, 0) / baseline_total for k in range(6)}
print(f"Baseline c5 分布: {['%.1f%%' % (v*100) for v in BASELINE_C5.values()]}")
print()

# ============= 200 期回测 =============
print("="*60)
print("回测: V22.A6 修复后 vs V22.A 基线 (200 期)")
print("="*60)

# 修正: 用最后 200 期做回测 (range 长度 = 200)
N_TEST = 200
START = n_periods - 201  # 从最后 200 期开始 (range 是 [START, n_periods-1), 长度 = 200)
hits_a6 = {4: 0, 6: 0, 10: 0, 20: 0}
hits_a = {4: 0, 6: 0, 10: 0, 20: 0}

print(f"n_periods = {n_periods}, START = {START}, 测试期数 = {n_periods - START - 1}")
assert n_periods - START - 1 == N_TEST, f"预期 {N_TEST} 期, 实际 {n_periods - START - 1} 期"

# 数据降序: data[0] 最新, data[n_periods-1] 最老
# last_i = i+1: 用 data[i+1] 看过去, actual = data[i] (data[i] 比 data[i+1] 更新)
# range 长度 = n_periods - 1 - START, 需要 200 期: START = n_periods - 201
for i in range(START, n_periods - 1):
    last_i = i + 1  # 上一期索引 (已知开奖, 用于 c5/c10/c30)
    actual = set(data[i].tolist())  # 实际开奖 (data[i] 是 last_i 的下一期, 因为降序)

    # V22.A6 修复版
    scores_a6 = score_v22_a6_fixed(last_i)
    # V22.A 基线 (用同样公式但去掉自适应部分)
    c5, c10, c30 = compute_c5_c10_c30(last_i)
    sa = np.zeros(80)
    sa[c5 == 0] = -0.5; sa[c5 == 1] = 0.0; sa[c5 == 2] = 0.2; sa[c5 == 3] = 0.2
    sa[c5 == 4] = -0.3; sa[c5 == 5] = -0.8; sa[c5 >= 6] = -1.5
    ta = np.zeros(80)
    ta[(c10 >= 2) & (c10 <= 4)] = 1.0; ta[c10 == 1] = 0.4; ta[c10 == 0] = 0.2
    ta[(c10 >= 5) & (c10 <= 6)] = -0.3; ta[c10 >= 7] = -0.8
    ua = np.zeros(80)
    ua[(c30 >= 5) & (c30 <= 10)] = 0.5; ua[(c30 >= 3) & (c30 <= 4)] = 0.2
    ua[c30 <= 2] = -0.2; ua[c30 >= 13] = -0.7
    scores_a = sa + ta + ua + compute_nbr(last_i)

    for top_n in [4, 6, 10, 20]:
        top_a6 = set(select_top_n(scores_a6, top_n))
        top_a = set(select_top_n(scores_a, top_n))
        hits_a6[top_n] += len(top_a6 & actual)
        hits_a[top_n] += len(top_a & actual)

print(f"\n{'TopN':<6}{'V22.A6 修复版':<24}{'V22.A 基线':<24}{'差 (pp)'}")
print("-"*70)
for n in [4, 6, 10, 20]:
    pct_a6 = hits_a6[n] / (N_TEST * n) * 100
    pct_a = hits_a[n] / (N_TEST * n) * 100
    diff = pct_a6 - pct_a
    print(f"{n:<6}{pct_a6:>7.2f}% ({hits_a6[n]:>4} hits)    {pct_a:>7.2f}% ({hits_a[n]:>4} hits)    {diff:+.2f}")

print(f"\n基线理论: 25.00% (随机)")
print(f"V22.A6 修复后 Top 4 目标: ≥27.28% (修复前数据)")