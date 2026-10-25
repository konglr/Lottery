"""
双色球近期开奖规律分析
================================

时间: 2026-09-28 22:41 GMT+8
作者: Lucky / MiniMax-M3

分析维度:
  1. 最近 N 期基本统计 (和值、跨度、奇偶、连号)
  2. 4 区位分布 (1-11 / 12-22 / 23-33)
  3. 冷热号 (近 5/10/30/100 期的 c5/c10/c30/c100)
  4. 重号趋势 (跟 lag1-lag5 的重号数变化)
  5. 邻号 (跟 lag1 的 ±1 命中率)
  6. 同尾号分布 (尾数 1-9 各出了几个)
  7. AC 值 (算术复杂度)
  8. 区间比、012 路比 等
  9. 历史 vs 近期对比 (找反转信号)

数据: data/双色球_lottery_data.csv (最新 ~3500+ 期)
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter

ROOT = Path.home() / 'Library/Mobile Documents/com~apple~CloudDocs/PycharmProjects/Lottery'
sys.path.insert(0, str(ROOT))
from lottery_data import LotteryData

ld = LotteryData(ROOT)
df, conf = ld.load('双色球')
red_cols = [f'红球{i}' for i in range(1, 7)]
data = df[red_cols].astype(int).values  # (n_periods, 6), 降序
periods = df['期号'].tolist()
n_periods = len(data)

print(f"数据: {n_periods} 期 (最新 {periods[0]} → 最老 {periods[-1]})\n")


def ac_value(nums):
    """AC 值: 算术复杂度 = 不同差值数 - (n-1)"""
    nums = sorted(nums)
    diffs = set()
    for i in range(len(nums)):
        for j in range(i+1, len(nums)):
            diffs.add(nums[j] - nums[i])
    return len(diffs) - (len(nums) - 1)


def stats(nums):
    """计算一期的所有统计量"""
    s = sorted(nums)
    return {
        '和值': sum(s),
        '跨度': s[-1] - s[0],
        '奇数': sum(1 for n in s if n % 2 == 1),
        'AC': ac_value(s),
        '4区(1-11/12-22/23-33)': f'{sum(1 for n in s if n<=11)}/{sum(1 for n in s if 12<=n<=22)}/{sum(1 for n in s if n>=23)}',
        '连号对': sum(1 for i in range(len(s)-1) if s[i+1]-s[i]==1),
        '尾数分布': dict(Counter(n % 10 for n in s)),
    }


# ============================================================
# 1. 最近 30 期一览
# ============================================================
print("=" * 70)
print("1. 最近 30 期基本统计")
print("=" * 70)
print(f"{'期号':<10}{'号码':<24}{'和值':<6}{'跨度':<6}{'奇':<4}{'AC':<4}{'4区':<10}{'连号'}")
print("-" * 70)
for i in range(30):
    nums = data[i].tolist()
    s = stats(nums)
    period = periods[i]
    nums_str = ' '.join(f'{n:02d}' for n in sorted(nums))
    print(f"{period:<10}{nums_str:<24}{s['和值']:<6}{s['跨度']:<6}{s['奇数']:<4}{s['AC']:<4}{s['4区(1-11/12-22/23-33)']:<10}{s['连号对']}")


# ============================================================
# 2. 近期均值 / P50 / 标准差 (跟历史对比)
# ============================================================
print("\n" + "=" * 70)
print("2. 近期 vs 历史均值对比 (找异常 / 反转信号)")
print("=" * 70)

metrics = ['和值', '跨度', '奇数', 'AC', '连号对']

# 不同窗口
windows = [5, 10, 20, 30, 100, 500]

# 历史基准 (跳过最近 N_TEST 期以避免泄漏)
hist_data = data[500:]  # 用早期 3000+ 期做基准

print(f"\n{'指标':<8}", end='')
for w in windows:
    print(f"{'近' + str(w) + '期':<10}", end='')
print(f"{'历史基准(早期)':<14}{'z-score'}")
print("-" * 70)

for metric in metrics:
    print(f"{metric:<8}", end='')
    # 算每个窗口的均值
    window_means = {}
    for w in windows:
        if metric == '4区':
            # 4 区分别算
            z1 = np.mean([sum(1 for n in data[i] if n<=11) for i in range(w)])
            z2 = np.mean([sum(1 for n in data[i] if 12<=n<=22) for i in range(w)])
            z3 = np.mean([sum(1 for n in data[i] if n>=23) for i in range(w)])
            window_means[w] = f'{z1:.1f}/{z2:.1f}/{z3:.1f}'
        elif metric == '奇数':
            window_means[w] = f'{np.mean([stats(data[i].tolist())["奇数"] for i in range(w)]):.1f}'
        elif metric == 'AC':
            window_means[w] = f'{np.mean([stats(data[i].tolist())["AC"] for i in range(w)]):.1f}'
        elif metric == '连号对':
            window_means[w] = f'{np.mean([stats(data[i].tolist())["连号对"] for i in range(w)]):.2f}'
        elif metric == '和值':
            window_means[w] = f'{np.mean([stats(data[i].tolist())["和值"] for i in range(w)]):.1f}'
        elif metric == '跨度':
            window_means[w] = f'{np.mean([stats(data[i].tolist())["跨度"] for i in range(w)]):.1f}'

    for w in windows:
        print(f"{window_means[w]:<10}", end='')

    # 历史基准 (跳过最近 N_TEST 期)
    if metric == '4区':
        h1 = np.mean([sum(1 for n in hist_data[i] if n<=11) for i in range(len(hist_data))])
        h2 = np.mean([sum(1 for n in hist_data[i] if 12<=n<=22) for i in range(len(hist_data))])
        h3 = np.mean([sum(1 for n in hist_data[i] if n>=23) for i in range(len(hist_data))])
        hist_str = f'{h1:.1f}/{h2:.1f}/{h3:.1f}'
        # z-score 用近期 30 期 vs 历史
        recent = data[:30]
        z_total = abs((h1+h2+h3) - (np.mean([sum(1 for n in recent[i] if n<=11) for i in range(30)]) +
                                     np.mean([sum(1 for n in recent[i] if 12<=n<=22) for i in range(30)]) +
                                     np.mean([sum(1 for n in recent[i] if n>=23) for i in range(30)])))
    else:
        hist_arr = [stats(hist_data[i].tolist())[metric] for i in range(len(hist_data))]
        recent_arr = [stats(data[i].tolist())[metric] for i in range(30)]
        hist_mean = np.mean(hist_arr)
        hist_std = np.std(hist_arr) + 1e-9
        recent_mean = np.mean(recent_arr)
        z = (recent_mean - hist_mean) / hist_std
        hist_str = f'{hist_mean:.1f}±{hist_std:.1f}'
        # 重新格式化
        print(f"{hist_str:<14}{z:+.2f}")
        continue
    print(f"{hist_str:<14}{'-'}")


# ============================================================
# 3. 冷热号 (c5/c10/c30/c100)
# ============================================================
print("\n" + "=" * 70)
print("3. 冷热号 (近 5/10/30/100 期)")
print("=" * 70)

windows = [5, 10, 30, 100]

# 每个窗口, 每个号出现次数
print(f"\n{'号':<4}", end='')
for w in windows:
    print(f"{f'近{w}期':<10}", end='')
print(f"{'热门度评级'}")
print("-" * 60)

hot_cold_score = {}
for n in range(1, 34):
    counts = []
    for w in windows:
        c = sum(1 for i in range(w) if n in data[i].tolist())
        counts.append(c)
    # 评级: c30>=3 = 热, c30=2 = 温, c30=1 = 冷, c30=0 = 极冷
    rating = ''
    if counts[2] >= 3: rating = '🔥 热'
    elif counts[2] == 2: rating = '温'
    elif counts[2] == 1: rating = '❄️ 冷'
    else: rating = '❄️❄️ 极冷'
    hot_cold_score[n] = (counts, rating)
    print(f"{n:<4}", end='')
    for c in counts:
        print(f"{c:<10}", end='')
    print(rating)

# 统计
print(f"\n近 30 期评级分布:")
rating_counter = Counter(v[1] for v in hot_cold_score.values())
for r, c in rating_counter.most_common():
    print(f"  {r}: {c} 个号")


# ============================================================
# 4. 重号趋势 (lag1-lag10)
# ============================================================
print("\n" + "=" * 70)
print("4. 重号趋势 (lag1-lag10, 最近 30 期平均)")
print("=" * 70)

print(f"{'Lag':<6}{'近5期':<8}{'近10期':<8}{'近20期':<8}{'近30期':<8}{'近100期':<8}{'历史均值'}")
print("-" * 70)

for lag in range(1, 11):
    recent5 = []
    recent10 = []
    recent20 = []
    recent30 = []
    recent100 = []
    hist = []

    for i in range(min(5000, n_periods - lag)):
        overlap = len(set(data[i].tolist()) & set(data[i+lag].tolist()))
        if i < 5: recent5.append(overlap)
        if i < 10: recent10.append(overlap)
        if i < 20: recent20.append(overlap)
        if i < 30: recent30.append(overlap)
        if i < 100: recent100.append(overlap)
        hist.append(overlap)

    print(f"lag{lag:<4}", end='')
    print(f"{np.mean(recent5):<8.2f}", end='')
    print(f"{np.mean(recent10):<8.2f}", end='')
    print(f"{np.mean(recent20):<8.2f}", end='')
    print(f"{np.mean(recent30):<8.2f}", end='')
    print(f"{np.mean(recent100):<8.2f}", end='')
    print(f"{np.mean(hist):<8.2f}")


# ============================================================
# 5. 邻号分析 (上期 ±1, ±2 的号下期出现概率)
# ============================================================
print("\n" + "=" * 70)
print("5. 邻号分析 (上期 ±1, ±2 的号下期出现概率)")
print("=" * 70)

N_LAG = 200
neighbor_hits = {1: 0, 2: 0}
neighbor_total = {1: 0, 2: 0}

for i in range(N_LAG):
    prev = set(data[i].tolist())
    actual = set(data[i+1].tolist())
    for d in [1, 2]:
        for p in prev:
            for delta in [-d, d]:
                nbr = p + delta
                if 1 <= nbr <= 33:
                    neighbor_total[d] += 1
                    if nbr in actual:
                        neighbor_hits[d] += 1

print(f"\n{'邻距':<6}{'总出现次数':<12}{'命中次数':<12}{'命中率':<10}{'基线(6/33)'}")
print("-" * 50)
for d in [1, 2]:
    rate = neighbor_hits[d] / neighbor_total[d] * 100 if neighbor_total[d] > 0 else 0
    print(f"±{d:<5}{neighbor_total[d]:<12}{neighbor_hits[d]:<12}{rate:<10.2f}{18.18:.2f}%")


# ============================================================
# 6. 尾数分布
# ============================================================
print("\n" + "=" * 70)
print("6. 尾数分布 (近 30 期)")
print("=" * 70)

tail_count = Counter()
for i in range(30):
    for n in data[i]:
        tail_count[n % 10] += 1

print(f"理论: 6 个号, 10 个尾数, 每个尾数期望 6*30/10 = {6*30/10:.1f} 次")
print(f"\n{'尾数':<6}{'次数':<8}{'占比':<8}{'热/冷'}")
print("-" * 40)
expected = 6 * 30 / 10
for t in range(10):
    c = tail_count[t]
    pct = c / (6 * 30) * 100
    z = (c - expected) / (expected ** 0.5)
    rating = '🔥' if z > 1.5 else ('❄️' if z < -1.5 else '·')
    print(f"{t:<6}{c:<8}{pct:<8.2f}{rating} (z={z:+.2f})")


# ============================================================
# 7. 蓝球分析
# ============================================================
print("\n" + "=" * 70)
print("7. 蓝球 (最近 30 期冷热)")
print("=" * 70)

blue_col = '蓝球'
if blue_col in df.columns:
    blues = df[blue_col].iloc[:30].astype(int).tolist()
    blue_counter = Counter(blues)
    print(f"\n{'号码':<5}{'近30期出现':<12}{'评级'}")
    print("-" * 40)
    for n in range(1, 17):
        c = blue_counter.get(n, 0)
        if c >= 4: rating = '🔥 热'
        elif c == 0: rating = '❄️❄️ 极冷'
        elif c == 1: rating = '❄️ 冷'
        else: rating = '温'
        print(f"{n:<5}{c:<12}{rating}")
else:
    print("未找到蓝球列")


# ============================================================
# 8. 综合判断 + 给下一期的预测建议
# ============================================================
print("\n" + "=" * 70)
print("8. 综合判断 + 下一期(2026113)预测建议")
print("=" * 70)

# 最近一期
last_nums = sorted(data[0].tolist())
last_period = periods[0]
last_stats = stats(last_nums)

print(f"\n最近一期: {last_period}  红球 {last_nums}")
print(f"  和值 {last_stats['和值']}, 跨度 {last_stats['跨度']}, 奇数 {last_stats['奇数']}, "
      f"4区 {last_stats['4区(1-11/12-22/23-33)']}, 连号 {last_stats['连号对']}")

# 找近期最热的号 (c30 >= 3) 作为 4 胆主推候选
hot_nums = [n for n, (counts, rating) in hot_cold_score.items() if '🔥' in rating]
cold_nums = [n for n, (counts, rating) in hot_cold_score.items() if '极冷' in rating]

print(f"\n🔥 近 30 期热号 (c30 >= 3): {sorted(hot_nums)}")
print(f"❄️ 近 30 期冷号 (c30 = 0):  {sorted(cold_nums)}")

# 趋势信号
recent30_和值 = np.mean([stats(data[i].tolist())['和值'] for i in range(30)])
recent100_和值 = np.mean([stats(data[i].tolist())['和值'] for i in range(100)])
all_和值 = np.mean([stats(data[i].tolist())['和值'] for i in range(n_periods)])

print(f"\n和值趋势:")
print(f"  近 30 期均值:  {recent30_和值:.1f}")
print(f"  近 100 期均值: {recent100_和值:.1f}")
print(f"  全部均值:      {all_和值:.1f}")

if recent30_和值 > all_和值 + 5:
    print("  ⚠️ 近期和值偏高, 下一期可能回归中位")
elif recent30_和值 < all_和值 - 5:
    print("  ⚠️ 近期和值偏低, 下一期可能回升")
else:
    print("  ✅ 近期和值在正常范围")

print("\n下一期(2026113)预测建议:")
print(f"  和值区间: {int(all_和值)-15}~{int(all_和值)+15}")
print(f"  4 区位推荐: 2/2/2 (均衡) 或 1/2/3 (偏大号)")
print(f"  4 胆主推:   从 {sorted(hot_nums)[:6]} 中选 4 个")
print(f"  冷号关注:   {sorted(cold_nums)[:3]} (反转可能)")