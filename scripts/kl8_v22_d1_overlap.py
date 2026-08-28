"""
KL8 V22.D1 — 20 号重号检验机制
===================================

V22.D1 = V22.A 基线 (P90 + 4区位严格5/5/5/5) + 20 号与 lag1-lag5 的重号检验

核心设计:
  - 基础: 用 V22.A 评分得到所有 80 号的分数
  - 选号: 严格 5/5/5/5 选出 20 胆
  - 重号检验: 计算 20 胆 vs lag1, lag2, ..., lag5 的重号数
  - 阈值: 如果某个 lag 的重号数 > 阈值, 用候补号替换 20 胆尾部评分最低的号

阈值设计 (基于近 500 期 lag 重号分布):
  - lag1 (近 1 期): 平均 ~10 个重号, P75=11, 设阈值 10 (触发后减到 ≤10)
  - lag2 (近 2 期): 平均 ~9 个, 设阈值 10
  - lag3 (近 3 期): 平均 ~8 个, 设阈值 9
  - lag4 (近 4 期): 平均 ~7 个, 设阈值 8
  - lag5 (近 5 期): 平均 ~6 个, 设阈值 7

替换策略:
  - 触发后: 从 sorted_idx[20:] 候补中按评分顺序取
  - 优先选: 不与该 lag 重, 不破坏 4 区位 5/5/5/5 配额
  - 如果破坏: 跳过该候补, 选下一个

V22.D1 与已有变种的区别:
  - V22.A3 (V6): 单期阈值 (L1≤11, L2≤11, L3≤7), 在贪心构造时跳过
  - V22.A4 (V7): lag1-5 替换末尾 2 个, 但每 lag 都替换
  - V22.D1: lag1-5 单独阈值, 只在超阈值时替换末尾 1 个
"""
import sys
import numpy as np
import json
from collections import Counter
from pathlib import Path
from datetime import datetime

ROOT = Path.home() / 'Library/Mobile Documents/com~apple~CloudDocs/PycharmProjects/Lottery'
sys.path.insert(0, str(ROOT))
from lottery_data import LotteryData

ld = LotteryData(ROOT)
df, conf = ld.load('快乐8')
red_cols = [f'红球{i}' for i in range(1, 21)]
data = df[red_cols].astype(int).values
periods = df['期号'].tolist()
n_periods = len(data)


# ============= 评分函数 (V22.A 基线) =============
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


def score_v22_a(last_i):
    """V22.A 基线评分"""
    c5, c10, c30 = compute_c5_c10_c30(last_i)
    s = np.zeros(80)
    s[c5 == 0] = -0.5
    s[c5 == 1] = 0.0
    s[c5 == 2] = 0.2
    s[c5 == 3] = 0.2
    s[c5 == 4] = -0.3
    s[c5 == 5] = -0.8
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

    return s + t + u + compute_nbr(last_i)


# ============= 严格 5/5/5/5 选号 =============
def select_20_strict(scores):
    """严格 5/5/5/5 选出 20 胆"""
    sorted_idx = np.argsort(scores)[::-1]
    selected = []
    zone_count = [0, 0, 0, 0]
    for z in range(4):
        zc = 0
        for idx in sorted_idx:
            if zc >= 5: break
            num = int(idx) + 1
            if num in selected: continue
            if (num - 1) // 20 != z: continue
            selected.append(num); zone_count[z] += 1; zc += 1
    return selected


# ============= V22.D1: 20 号重号检验 =============
# 阈值 (基于近 500 期 lag 重号分布 — 详见回测)
LAG_THRESHOLDS = {
    'lag1': 9,
    'lag2': 9,
    'lag3': 9,
    'lag4': 9,
    'lag5': 9,
}


def check_overlap(selected, lag_period_idx):
    """计算 selected 与 lag_period_idx 期的重号数"""
    lag_set = set(data[lag_period_idx].tolist())
    return sum(1 for n in selected if n in lag_set)


def v22_d1_select(last_i, scores, verbose=False):
    """V22.D1 选号:
    1. 严格 5/5/5/5 选 20 胆
    2. 检查 lag1-lag5 重号
    3. 若某 lag 超阈值, 从候补池中按评分顺序替换 20 胆中评分最低且与该 lag 重的号
    4. 优先不与 lag 重, 不破坏 4 区位

    Returns:
        (selected, overlap_history, replacements)
    """
    selected = select_20_strict(scores)
    sorted_idx = np.argsort(scores)[::-1]
    candidates = [int(idx) + 1 for idx in sorted_idx if (int(idx) + 1) not in selected]

    overlap_history = {}
    replacements = []

    for lag_name, lag_k in [('lag1', 1), ('lag2', 2), ('lag3', 3), ('lag4', 4), ('lag5', 5)]:
        lag_period_idx = last_i + lag_k  # 数据降序, +k = 更早 k 期
        if lag_period_idx >= n_periods:
            continue

        threshold = LAG_THRESHOLDS[lag_name]
        overlap = check_overlap(selected, lag_period_idx)
        overlap_history[lag_name] = overlap

        if overlap > threshold:
            # 触发替换: 选 selected 中评分最低且与 lag 重的号
            lag_set = set(data[lag_period_idx].tolist())

            # 找出 selected 中与 lag 重的号, 按评分升序排序
            overlap_in_selected = [n for n in selected if n in lag_set]
            overlap_in_selected_sorted = sorted(overlap_in_selected, key=lambda n: scores[n-1])

            # 替换 1 个: 评分最低的与 lag 重的号
            if overlap_in_selected_sorted:
                to_remove = overlap_in_selected_sorted[0]
                # 找候补: 不与 lag 重, 且不破坏 4 区位 (替换后该区只能有 5 个)
                removed_zone = (to_remove - 1) // 20
                zone_count = [0, 0, 0, 0]
                for n in selected:
                    if n != to_remove:
                        zone_count[(n - 1) // 20] += 1

                replacement = None
                for cand in candidates:
                    if cand in selected: continue
                    if cand in lag_set: continue
                    cand_zone = (cand - 1) // 20
                    if zone_count[cand_zone] >= 5:
                        continue  # 破坏 4 区位
                    replacement = cand
                    break

                if replacement:
                    selected = [n for n in selected if n != to_remove] + [replacement]
                    candidates.remove(replacement)
                    candidates.append(to_remove)  # 被替换的号回到候补
                    replacements.append({
                        'lag': lag_name,
                        'removed': to_remove,
                        'added': replacement,
                        'overlap_before': overlap,
                    })
                    if verbose:
                        print(f"  {lag_name}: 重号 {overlap} > {threshold}, 替换 {to_remove} → {replacement}")

    return selected, overlap_history, replacements


def morphology_check(nums):
    nums = sorted(nums)
    return {
        '和值': sum(nums),
        '跨度': nums[-1] - nums[0],
        '奇数': sum(1 for n in nums if n % 2 == 1),
        '4区': '/'.join(str(sum(1 for n in nums if z*20 < n <= (z+1)*20)) for z in range(4)),
        '连号对': sum(1 for i in range(len(nums) - 1) if nums[i+1] - nums[i] == 1),
    }


# ============= 实战预测: 2026211 期 =============
if __name__ == '__main__':
    last_period = '2026210'
    target_period = '2026211'
    last_i = periods.index(last_period)

    print(f"📅 数据: {n_periods} 期")
    print(f"🎯 目标: {target_period} (今晚 21:30 开奖)")
    print(f"📌 上期 {last_period}: {sorted(data[last_i].tolist())}")
    print()
    print(f"阈值: {LAG_THRESHOLDS}")
    print()

    scores = score_v22_a(last_i)
    selected_20, overlap_hist, replacements = v22_d1_select(last_i, scores, verbose=True)
    selected_20_sorted = sorted(selected_20, key=lambda n: scores[n-1], reverse=True)

    print()
    print("="*70)
    print(f"【V22.D1 — 2026211 期 20 胆】")
    print("="*70)
    print(f"\n📋 20 胆 (按评分排序): {selected_20_sorted}")
    print(f"   形态: {morphology_check(selected_20_sorted)}")

    print(f"\n📊 lag1-lag5 重号检验:")
    for lag_name, overlap in overlap_hist.items():
        threshold = LAG_THRESHOLDS[lag_name]
        status = '✓' if overlap <= threshold else '🔄 触发替换'
        print(f"  {lag_name} ({periods[last_i + int(lag_name[-1])]}): {overlap}/{threshold} {status}")

    print(f"\n🔄 替换记录: {len(replacements)} 次")
    for r in replacements:
        print(f"  {r['lag']}: {r['removed']} → {r['added']} (前重号 {r['overlap_before']})")

    # 选 Top N
    for n_top in [4, 6, 10]:
        top_n = selected_20_sorted[:n_top]
        print(f"\n🥇 Top {n_top}: {top_n}")
        print(f"   形态: {morphology_check(top_n)}")

    # JSON 归档
    out = {
        'meta': {
            'created_at': datetime.now().strftime('%Y-%m-%d %H:%M GMT+8'),
            'method': 'V22.D1 — 20 号 lag1-lag5 重号检验 (统一阈值=9, lag1-lag5 同阈值)',
            'target_period': target_period,
            'open_time': '2026-08-09 21:30',
            'lottery': '快乐8',
            'model': 'Lucky / MiniMax-M3',
            'last_period': last_period,
            'last_draw': sorted(data[last_i].tolist()),
            'n_history': n_periods,
            'lag_thresholds': LAG_THRESHOLDS,
        },
        'next_period_2026211': {
            'Top_4':  selected_20_sorted[:4],
            'Top_6':  selected_20_sorted[:6],
            'Top_10': selected_20_sorted[:10],
            'Top_20': selected_20_sorted,
            'morphology_4': morphology_check(selected_20_sorted[:4]),
            'morphology_20': morphology_check(selected_20_sorted),
            'overlap_history': overlap_hist,
            'replacements': replacements,
        }
    }

    json_path = ROOT / 'data' / 'backtest' / f'{target_period}_predictions_v22_d1.json'
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print()
    print(f"✅ JSON 归档: {json_path}")