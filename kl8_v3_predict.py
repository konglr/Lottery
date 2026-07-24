"""
kl8_v3_predict.py — 快乐8 v3 多特征加权评分预测器
- SW=10 + SW=5 双窗口并行
- F5(遗漏压力)单特征策略保留作对照
- 目标期:2026193 (2026-07-22 21:30)
"""
import pandas as pd
import numpy as np
from collections import Counter
from itertools import combinations
import json
import sys

CSV_PATH = '/Users/clarkkong/Library/Mobile Documents/com~apple~CloudDocs/PycharmProjects/Lottery/data/快乐8_lottery_data.csv'

def load_data():
    df = pd.read_csv(CSV_PATH)
    return df.head(2000)  # 近 2000 期

# ==================== F1 冷热度 ====================
def f1_cold_hot(df, num, recent_n=30):
    recent = df.head(recent_n)
    red_cols = [f'红球{i}' for i in range(1, 21)]
    nums = recent[red_cols].values.flatten().astype(int)
    freq = Counter(nums)
    counts = sorted(freq.values(), reverse=True)
    # rank
    all_freq = Counter(nums)
    # 80 号频次
    full_freq = Counter(df.head(recent_n)[red_cols].values.flatten().astype(int))
    rank = sorted(range(1, 81), key=lambda x: -full_freq.get(x, 0)).index(num)
    base = (1 - rank/80) * 5
    # 加速项
    recent10 = df.head(10)[red_cols].values.flatten().astype(int)
    freq10 = Counter(recent10).get(num, 0)
    avg_freq = recent_n * 20 / 80  # = recent_n * 0.25
    if freq10 > avg_freq * 1.5: base += 0.5
    elif freq10 < avg_freq * 0.5: base -= 0.5
    return max(0, min(5, base))

# ==================== F2 区域集中度 ====================
def f2_zone_density(df, num, recent_n=30):
    zone = (num - 1) // 20  # 0,1,2,3
    red_cols = [f'红球{i}' for i in range(1, 21)]
    recent = df.head(recent_n)
    zone_density_per_period = []
    for _, row in recent.iterrows():
        nums = row[red_cols].astype(int).tolist()
        zone_count = sum(1 for n in nums if (n-1)//20 == zone)
        zone_density_per_period.append(zone_count)
    avg_density = np.mean(zone_density_per_period)
    expected = 5  # 80 选 20,每区期望 5
    full_freq = Counter(df.head(recent_n)[red_cols].values.flatten().astype(int))
    rank_in_zone = sorted([n for n in range(zone*20+1, (zone+1)*20+1)],
                          key=lambda x: -full_freq.get(x, 0)).index(num)
    if avg_density > expected * 1.3:  # 区热
        return 4 + min(1, rank_in_zone / 20)
    elif avg_density < expected * 0.7:  # 区冷
        in_recent = num in recent[red_cols].values.flatten()
        return 3 + (1 if in_recent else 0)
    return 2.5

# ==================== F3 形态适应度 ====================
def f3_morphology(df, num, recent_n=10):
    red_cols = [f'红球{i}' for i in range(1, 21)]
    recent = df.head(recent_n)
    score = 0
    # 3.1 连号倾向
    consec_count = 0
    for _, row in recent.iterrows():
        nums = sorted(set(row[red_cols].astype(int).tolist()))
        for i in range(len(nums)-1):
            if nums[i+1] - nums[i] == 1:
                consec_count += 1
    avg_consec = consec_count / recent_n
    # 检查 num 是否与已知高频连号
    if avg_consec > 2.0:
        if num - 1 in recent[red_cols].values.flatten() or num + 1 in recent[red_cols].values.flatten():
            score += 1.25
    elif avg_consec < 0.5:
        if num - 1 not in recent[red_cols].values.flatten() and num + 1 not in recent[red_cols].values.flatten():
            score += 1.25
    else:
        score += 0.625
    # 3.2 大小比
    big_count = 0
    total = 0
    for _, row in recent.iterrows():
        for n in row[red_cols].astype(int):
            total += 1
            if n > 40: big_count += 1
    big_ratio = big_count / total if total else 0.5
    if big_ratio > 0.55:
        if num > 40: score += 1.25
    elif big_ratio < 0.45:
        if num <= 40: score += 1.25
    else:
        score += 0.625
    # 3.3 奇偶比
    odd_count = 0
    total = 0
    for _, row in recent.iterrows():
        for n in row[red_cols].astype(int):
            total += 1
            if n % 2 == 1: odd_count += 1
    odd_ratio = odd_count / total if total else 0.5
    if odd_ratio > 0.55:
        if num % 2 == 1: score += 1.25
    elif odd_ratio < 0.45:
        if num % 2 == 0: score += 1.25
    else:
        score += 0.625
    return max(0, min(5, score))

# ==================== F4 重号邻号势能 ====================
def f4_repeat_neighbor(df, num):
    red_cols = [f'红球{i}' for i in range(1, 21)]
    last_set = set(df.iloc[0][red_cols].astype(int).tolist())
    last_last_set = set(df.iloc[1][red_cols].astype(int).tolist())
    neighbors = set()
    for n in last_set:
        for d in [-2, -1, 1, 2]:
            if 1 <= n + d <= 80: neighbors.add(n + d)
    neighbors -= last_set
    if num in last_set: return 5.0
    if num in neighbors: return 4.0
    if num in last_last_set: return 3.0
    return 1.0

# ==================== F5 遗漏压力 ====================
def f5_miss_pressure(df, num, recent_n=50):
    red_cols = [f'红球{i}' for i in range(1, 21)]
    recent = df.head(recent_n)
    # 当前遗漏(连续多少期未出)
    miss = 0
    for _, row in recent.iterrows():
        nums = row[red_cols].astype(int).tolist()
        if num not in nums: miss += 1
        else: break
    # 平均间隔(近 recent_n 期)
    appear_count = sum(1 for _, row in recent.iterrows() if num in row[red_cols].astype(int).tolist())
    avg_interval = recent_n / max(appear_count, 1)
    miss_ratio = miss / avg_interval
    if miss_ratio < 0.5: return 1.0
    elif miss_ratio < 1.0: return 3.0
    elif miss_ratio < 1.5: return 5.0
    elif miss_ratio < 2.0: return 4.5
    elif miss_ratio < 3.0: return 3.0
    else: return 2.0

# ==================== F6 同尾聚集 ====================
def f6_tail(df, num, recent_n=10):
    tail = num % 10
    red_cols = [f'红球{i}' for i in range(1, 21)]
    recent = df.head(recent_n)
    tail_count = 0
    num_appear = False
    for _, row in recent.iterrows():
        for n in row[red_cols].astype(int):
            if n % 10 == tail:
                tail_count += 1
            if n == num:
                num_appear = True
    density = tail_count / recent_n / 20  # 期望 0.1
    if density > 0.15:  # 密集
        return 4 + (1 if num_appear else 0)
    elif density < 0.05:  # 稀薄
        return 2 + (2 if num_appear else 0)
    return 3

# ==================== 动态权重 ====================
def compute_weight(df, target_n, recent_n, feat_func, current_i):
    """评估特征 feat 在 recent_n 期内的判别力"""
    red_cols = [f'红球{i}' for i in range(1, 21)]
    total_hits = 0
    valid = 0
    for j in range(1, recent_n + 1):
        if j >= len(df): break
        sub = df.iloc[j:].copy()
        if len(sub) < 2: continue
        # 评分该期前的 80 个号
        scores = {}
        for num in range(1, 81):
            scores[num] = feat_func(sub, num)
        # Top 9
        top9 = set(sorted(scores, key=lambda x: -scores[x])[:9])
        # 实际开奖(下一期,即 j-1)
        actual = set(df.iloc[j-1][red_cols].astype(int).tolist())
        hits = len(top9 & actual)
        total_hits += hits
        valid += 1
    if valid == 0: return 1.0
    avg_hits = total_hits / valid
    baseline = 2.25  # 9 * 20/80
    ratio = avg_hits / baseline
    # 权重映射
    if ratio >= 1.2: return 1.5
    elif ratio >= 1.0: return 1.0
    elif ratio >= 0.8: return 0.7
    else: return 0.7

# ==================== 主流程 ====================
def score_all_numbers(df, recent_n_for_weight):
    """返回每号的 6 特征 + 综合得分(SW=recent_n_for_weight)"""
    red_cols = [f'红球{i}' for i in range(1, 21)]
    # 计算权重(每个特征评估 recent_n_for_weight 期)
    weights = {}
    weights['F1'] = compute_weight(df, 1, recent_n_for_weight, f1_cold_hot, 1)
    weights['F2'] = compute_weight(df, 2, recent_n_for_weight, f2_zone_density, 1)
    weights['F3'] = compute_weight(df, 3, recent_n_for_weight, f3_morphology, 1)
    weights['F4'] = compute_weight(df, 4, recent_n_for_weight, f4_repeat_neighbor, 1)
    weights['F5'] = compute_weight(df, 5, recent_n_for_weight, f5_miss_pressure, 1)
    weights['F6'] = compute_weight(df, 6, recent_n_for_weight, f6_tail, 1)
    
    # 计算本期每号得分
    feat_funcs = {
        'F1': f1_cold_hot, 'F2': f2_zone_density, 'F3': f3_morphology,
        'F4': f4_repeat_neighbor, 'F5': f5_miss_pressure, 'F6': f6_tail
    }
    results = []
    for num in range(1, 81):
        feats = {k: f(df, num) for k, f in feat_funcs.items()}
        total = sum(feats[k] * weights[k] for k in feats)
        results.append({
            'num': num,
            'F1': feats['F1'], 'F2': feats['F2'], 'F3': feats['F3'],
            'F4': feats['F4'], 'F5': feats['F5'], 'F6': feats['F6'],
            'w_F1': weights['F1'], 'w_F2': weights['F2'], 'w_F3': weights['F3'],
            'w_F4': weights['F4'], 'w_F5': weights['F5'], 'w_F6': weights['F6'],
            'total': total
        })
    return results, weights

def select_top9(results, last_set):
    """选 Top 9 + 校验"""
    sorted_r = sorted(results, key=lambda x: -x['total'])
    top9 = sorted([r['num'] for r in sorted_r[:9]])
    tuo1 = sorted_r[9]['num']
    # 校验
    zone1 = sum(1 for n in top9 if n <= 27)
    zone2 = sum(1 for n in top9 if 28 <= n <= 53)
    zone3 = sum(1 for n in top9 if n >= 54)
    return {
        'dan_9': top9,
        'tuo_1': tuo1,
        'sum': sum(top9),
        'span': max(top9) - min(top9),
        'zone_dist': f"{zone1}:{zone2}:{zone3}",
        'repeats': [n for n in top9 if n in last_set],
        'repeat_count': sum(1 for n in top9 if n in last_set)
    }

if __name__ == '__main__':
    df = load_data()
    print(f"数据范围: {df['issue'].iloc[-1]} → {df['issue'].iloc[0]},共 {len(df)} 期")
    print(f"目标期: 2026193 (2026-07-22 21:30)")
    print(f"上期(2026192)开奖号: {sorted(df.iloc[0][[f'红球{i}' for i in range(1, 21)]].astype(int).tolist())}")
    print()
    
    last_set = set(df.iloc[0][[f'红球{i}' for i in range(1, 21)]].astype(int).tolist())
    
    # === 方案 A: SW=10 ===
    print("=" * 60)
    print("方案 A: SW=10 (10 期评估动态权重)")
    print("=" * 60)
    results_10, weights_10 = score_all_numbers(df, recent_n_for_weight=10)
    pick_10 = select_top9(results_10, last_set)
    print(f"权重: F1={weights_10['F1']:.2f} F2={weights_10['F2']:.2f} F3={weights_10['F3']:.2f} "
          f"F4={weights_10['F4']:.2f} F5={weights_10['F5']:.2f} F6={weights_10['F6']:.2f}")
    print(f"9 胆: {pick_10['dan_9']}")
    print(f"拖码: {pick_10['tuo_1']}")
    print(f"和值: {pick_10['sum']}  跨度: {pick_10['span']}  三区: {pick_10['zone_dist']}")
    print(f"上期重号({pick_10['repeat_count']}): {pick_10['repeats']}")
    
    # === 方案 B: SW=5 ===
    print()
    print("=" * 60)
    print("方案 B: SW=5 (5 期评估动态权重)")
    print("=" * 60)
    results_5, weights_5 = score_all_numbers(df, recent_n_for_weight=5)
    pick_5 = select_top9(results_5, last_set)
    print(f"权重: F1={weights_5['F1']:.2f} F2={weights_5['F2']:.2f} F3={weights_5['F3']:.2f} "
          f"F4={weights_5['F4']:.2f} F5={weights_5['F5']:.2f} F6={weights_5['F6']:.2f}")
    print(f"9 胆: {pick_5['dan_9']}")
    print(f"拖码: {pick_5['tuo_1']}")
    print(f"和值: {pick_5['sum']}  跨度: {pick_5['span']}  三区: {pick_5['zone_dist']}")
    print(f"上期重号({pick_5['repeat_count']}): {pick_5['repeats']}")
    
    # === 方案 C: 仅 F5(遗漏压力)单特征 ===
    print()
    print("=" * 60)
    print("方案 C: 仅 F5 遗漏压力 (单特征)")
    print("=" * 60)
    sorted_f5 = sorted(results_10, key=lambda x: -x['F5'])  # 用最近算的 F5 值
    top9_f5 = sorted([r['num'] for r in sorted_f5[:9]])
    tuo_f5 = sorted_f5[9]['num']
    zone1 = sum(1 for n in top9_f5 if n <= 27)
    zone2 = sum(1 for n in top9_f5 if 28 <= n <= 53)
    zone3 = sum(1 for n in top9_f5 if n >= 54)
    print(f"9 胆: {top9_f5}")
    print(f"拖码: {tuo_f5}")
    print(f"和值: {sum(top9_f5)}  跨度: {max(top9_f5)-min(top9_f5)}  三区: {zone1}:{zone2}:{zone3}")
    print(f"上期重号: {[n for n in top9_f5 if n in last_set]}")
    
    # 共识度
    set_a = set(pick_10['dan_9'])
    set_b = set(pick_5['dan_9'])
    set_c = set(top9_f5)
    print()
    print("=" * 60)
    print("三方案共识度")
    print("=" * 60)
    print(f"A ∩ B: {sorted(set_a & set_b)} ({len(set_a & set_b)}/9)")
    print(f"A ∩ C: {sorted(set_a & set_c)} ({len(set_a & set_c)}/9)")
    print(f"B ∩ C: {sorted(set_b & set_c)} ({len(set_b & set_c)}/9)")
    print(f"A ∩ B ∩ C: {sorted(set_a & set_b & set_c)} ({len(set_a & set_b & set_c)}/9)")