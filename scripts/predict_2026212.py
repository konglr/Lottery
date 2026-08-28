"""
KL8 2026212 期 多模型预测综合脚本
====================================

使用模型:
- V22.A      基线 (P90 + 4 区位配额 + 邻号)
- V22.A1     + 方案 F (c2/c3 hot 加分)
- V22.A6     自适应冷热号比例
- V22.F      偏热号 (c5=2-3 加分)
- V18.C      共识 (0.5*V22 + 0.3*V3 + 0.2*V10)
- V22.D1     20 号 lag1-lag5 重号检验

输出:
- data/backtest/2026212_predictions_multi_model.json
- data/backtest/2026212_predictions_v22_d1.json
"""
import json
import numpy as np
from collections import Counter
from pathlib import Path

ROOT = Path.home() / 'Library/Mobile Documents/com~apple~CloudDocs/PycharmProjects/Lottery'
import sys
sys.path.insert(0, str(ROOT))
from lottery_data import LotteryData

ld = LotteryData(ROOT)
df, conf = ld.load('快乐8')
red_cols = [f'红球{i}' for i in range(1, 21)]
data = df[red_cols].astype(int).values
n_periods = len(data)
periods = df['期号'].tolist()

LAST_PERIOD = '2026211'
TARGET_PERIOD = '2026212'
last_i = periods.index(LAST_PERIOD)

print(f'数据: {n_periods} 期')
print(f'上期 {LAST_PERIOD}: {sorted(data[last_i].tolist())}')
print(f'预测目标: {TARGET_PERIOD}')
print()

# ============= 公共特征 =============
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


# ============= baseline c5 =============
baseline_counter = Counter()
baseline_total = 0
for li in range(n_periods):
    if li + 5 < n_periods:
        c5 = np.zeros(80, dtype=int)
        for k in range(1, 6):
            if li + k < n_periods:
                for n in data[li + k]: c5[n-1] += 1
        for c in c5:
            baseline_counter[c] += 1
            baseline_total += 1
BASELINE_C5 = {k: baseline_counter.get(k, 0) / baseline_total for k in range(6)}


# ============= V22.A (基线) =============
def score_v22_a(last_i):
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


# ============= V22.A1 (方案 F) =============
def score_v22_a1(last_i):
    base = score_v22_a(last_i)
    c5 = compute_c5_c10_c30(last_i)[0]
    
    # c5_extra (惩罚过热)
    c5_extra = np.zeros(80)
    c5_extra[c5 == 3] = -0.5 * 0.5
    c5_extra[c5 == 4] = -0.5 * 0.8
    c5_extra[c5 >= 5] = -0.5 * 1.5
    
    # c2/c3 hot 加分 (方案 F)
    c2 = np.zeros(80, dtype=int)
    c3 = np.zeros(80, dtype=int)
    for k in range(1, 3):
        if last_i + k < n_periods:
            for n in data[last_i + k]: c2[n-1] += 1
    for k in range(1, 4):
        if last_i + k < n_periods:
            for n in data[last_i + k]: c3[n-1] += 1
    
    c2_score = np.zeros(80)
    c2_score[c2 == 1] = 0.6
    c3_score = np.zeros(80)
    c3_score[c3 == 1] = 0.4
    
    return base + c5_extra + c2_score + c3_score


# ============= V22.A6 (自适应) =============
def score_v22_a6(last_i, lookback=3, scale=2.0):
    base = score_v22_a(last_i)
    c5 = compute_c5_c10_c30(last_i)[0]
    
    observed_counter = Counter()
    observed_total = 0
    for i in range(1, lookback + 1):
        actual_i = last_i - i
        if actual_i >= 0:
            c5_obs = np.zeros(80, dtype=int)
            for k in range(1, 6):
                if actual_i + k < n_periods:
                    for n in data[actual_i + k]: c5_obs[n-1] += 1
            for n in data[actual_i]:
                observed_counter[c5_obs[n-1]] += 1
                observed_total += 1
    observed = {k: observed_counter.get(k, 0) / observed_total if observed_total > 0 else 0
                for k in range(6)}
    
    adj = np.zeros(80)
    for k in range(6):
        adj[c5 == k] += (observed[k] - BASELINE_C5[k]) * scale
    
    return base + adj


# ============= V22.F (偏热号) =============
def score_v22_f(last_i):
    base = score_v22_a(last_i)
    c5 = compute_c5_c10_c30(last_i)[0]
    boost = np.zeros(80)
    boost[c5 == 2] = 0.15
    boost[c5 == 3] = 0.25
    return base + boost


# ============= V22.D1 (重号检验) =============
def score_v22_d1(last_i, lag_thresholds={1:9, 2:9, 3:9, 4:9, 5:9}):
    """V22.D1 = V22.A 基线 + 20 号 lag1-lag5 重号检验"""
    scores = score_v22_a(last_i).copy()
    sorted_idx = np.argsort(scores)[::-1]
    
    # 先选 20 号 (4 区位严格 5/5/5/5)
    selected = []
    zone_count = [0, 0, 0, 0]
    for z in range(4):
        for idx in sorted_idx:
            if zone_count[z] >= 5: break
            num = int(idx) + 1
            if num in selected: continue
            if (num - 1) // 20 != z: continue
            selected.append(num); zone_count[z] += 1
    
    replacements = []
    
    # lag1-lag5 重号检验
    for lag, threshold in lag_thresholds.items():
        if last_i + lag >= n_periods: continue
        lag_draw = set(data[last_i + lag].tolist())
        overlap = len(set(selected) & lag_draw)
        if overlap >= threshold:
            # 找与 lag 重的号,按评分升序排序
            overlap_nums = [n for n in selected if n in lag_draw]
            sorted_overlap = sorted(overlap_nums, key=lambda n: scores[n-1])
            # 末尾 1 个删除
            to_remove = sorted_overlap[0]
            selected.remove(to_remove)
            # 从 sorted_idx[20:] 候补里按评分取下一个不与 lag 重且不在 selected 的
            for idx in sorted_idx[20:]:
                num = int(idx) + 1
                if num in selected: continue
                if num in lag_draw: continue
                z = (num - 1) // 20
                if zone_count[z] >= 8: continue
                selected.append(num); zone_count[z] += 1
                replacements.append({
                    'lag': f'lag{lag}',
                    'removed': to_remove,
                    'added': num,
                    'overlap_before': overlap,
                })
                break
    
    # 4/6/10/20 派生
    sorted_20 = sorted(selected, key=lambda n: scores[n-1], reverse=True)
    return scores, sorted_20[:4], sorted_20[:6], sorted_20[:10], sorted_20, replacements


# ============= V18.C (共识: 0.5*V22 + 0.3*V3 + 0.2*V10) =============
def score_v3_simple(last_i):
    """V3 朴素频次加权"""
    freq = np.zeros(80)
    for k in range(1, 11):
        if last_i + k < n_periods:
            for n in data[last_i + k]:
                freq[n-1] += 1
    return freq


def score_v10_simple(last_i):
    """V10 反向频次"""
    freq = score_v3_simple(last_i)
    return -freq  # 反向


def score_v18_c(last_i):
    s22 = score_v22_a(last_i)
    s3 = score_v3_simple(last_i)
    s10 = score_v10_simple(last_i)
    # 归一化到 0-1 范围
    def norm(x):
        rng = x.max() - x.min()
        return (x - x.min()) / rng if rng > 0 else np.zeros_like(x)
    return 0.5 * norm(s22) + 0.3 * norm(s3) + 0.2 * norm(s10)


# ============= 选号工具 =============
def select_top_n(scores, top_n):
    sorted_idx = np.argsort(scores)[::-1]
    return [int(idx) + 1 for idx in sorted_idx[:top_n]]


def morphology(nums):
    nums = sorted(nums)
    return {
        '和值': sum(nums),
        '跨度': nums[-1] - nums[0],
        '奇数': sum(1 for n in nums if n % 2 == 1),
        '4区': '/'.join(str(sum(1 for n in nums if z*20 < n <= (z+1)*20)) for z in range(4)),
        '连号对': sum(1 for i in range(len(nums) - 1) if nums[i+1] - nums[i] == 1),
    }


# ============= 主流程 =============
print('=' * 70)
print('计算各模型评分...')
print('=' * 70)

results = {}
results['V22.A'] = score_v22_a(last_i)
results['V22.A1'] = score_v22_a1(last_i)
results['V22.A6'] = score_v22_a6(last_i)
results['V22.F'] = score_v22_f(last_i)
results['V18.C'] = score_v18_c(last_i)
_, _, _, _, v22d1_20, v22d1_repl = score_v22_d1(last_i)
v22d1_scores = score_v22_a(last_i)

# ============= 输出结果 =============
print()
print('=' * 70)
print('【多模型综合预测】')
print('=' * 70)

multi_model = {
    'meta': {
        'created_at': '2026-08-10 15:24 GMT+8',
        'method': '6 模型集成: V22.A + V22.A1 + V22.A6 + V22.F + V18.C + V22.D1',
        'target_period': TARGET_PERIOD,
        'open_time': '2026-08-10 21:30',
        'lottery': '快乐8',
        'model': 'Lucky / MiniMax-M3',
        'last_period': LAST_PERIOD,
        'last_draw': sorted(data[last_i].tolist()),
        'n_history': n_periods,
    },
    f'next_period_{TARGET_PERIOD}': {},
    'consensus': {},
}

model_meta = {
    'V22.A': 'V22.A 基线 (P90 + 4区位配额 + 邻号)',
    'V22.A1': 'V22.A1 + 方案 F (c2/c3 hot 加分)',
    'V22.A6': 'V22.A6 自适应冷热号 (lookback=3, scale=2.0)',
    'V22.F': 'V22.F 偏热号 (c5=2-3 加分)',
    'V18.C': 'V18.C 共识 (0.5*V22 + 0.3*V3 + 0.2*V10)',
    'V22.D1': 'V22.D1 重号检验 (lag1-5 统一阈值=9)',
}

for model_name, scores in [('V22.A', results['V22.A']),
                            ('V22.A1', results['V22.A1']),
                            ('V22.A6', results['V22.A6']),
                            ('V22.F', results['V22.F']),
                            ('V18.C', results['V18.C'])]:
    top_4 = select_top_n(scores, 4)
    top_6 = select_top_n(scores, 6)
    top_10 = select_top_n(scores, 10)
    top_20 = select_top_n(scores, 20)
    
    multi_model[f'next_period_{TARGET_PERIOD}'][model_name] = {
        'method': model_meta[model_name],
        'Top_4': top_4,
        'Top_6': top_6,
        'Top_10': top_10,
        'Top_20': top_20,
        'morphology_4': morphology(top_4),
        'morphology_20': morphology(top_20),
    }
    
    print(f'\n【{model_name}】 {model_meta[model_name]}')
    print(f'  Top_4:  {top_4}  ({morphology(top_4)})')
    print(f'  Top_10: {top_10}')
    print(f'  Top_20: {top_20}')

# V22.D1
top_4_d1 = select_top_n(v22d1_scores, 4)
top_6_d1 = select_top_n(v22d1_scores, 6)
top_10_d1 = select_top_n(v22d1_scores, 10)
multi_model[f'next_period_{TARGET_PERIOD}']['V22.D1'] = {
    'method': model_meta['V22.D1'],
    'Top_4': top_4_d1,
    'Top_6': top_6_d1,
    'Top_10': top_10_d1,
    'Top_20': v22d1_20,
    'morphology_4': morphology(top_4_d1),
    'morphology_20': morphology(v22d1_20),
    'overlap_history': {f'lag{i}': len(set(v22d1_20) & set(data[last_i+i].tolist())) 
                        for i in range(1, 6) if last_i + i < n_periods},
    'replacements': v22d1_repl,
}
print(f'\n【V22.D1】 {model_meta["V22.D1"]}')
print(f'  Top_4:  {top_4_d1}')
print(f'  Top_20: {v22d1_20}')
print(f'  Replacements: {v22d1_repl}')

# ============= 共识 =============
print()
print('=' * 70)
print('【共识统计】')
print('=' * 70)

all_top4 = []
all_top10 = []
all_top20 = []
for model_name, scores in [('V22.A', results['V22.A']),
                            ('V22.A1', results['V22.A1']),
                            ('V22.A6', results['V22.A6']),
                            ('V22.F', results['V22.F']),
                            ('V18.C', results['V18.C'])]:
    all_top4 += select_top_n(scores, 4)
    all_top10 += select_top_n(scores, 10)
    all_top20 += select_top_n(scores, 20)

# Top 4 共识 (出现 ≥3 次)
c4 = Counter(all_top4)
top4_consensus = [(k, v) for k, v in c4.items() if v >= 3]
top4_consensus.sort(key=lambda x: -x[1])
print(f'\nTop 4 共识 (出现 ≥3 次 / 5 模型):')
for k, v in top4_consensus:
    print(f'  {k}: {v} 次')

# 推荐 4 胆 (取共识最高分)
recommended_4 = [k for k, v in top4_consensus[:4]] if len(top4_consensus) >= 4 else \
                [k for k, v in c4.most_common(4)]
print(f'\n推荐 4 胆: {recommended_4}')

# Top 10 共识 (≥3 次)
c10 = Counter(all_top10)
top10_consensus = [(k, v) for k, v in c10.items() if v >= 3]
top10_consensus.sort(key=lambda x: -x[1])
top10_nums = [k for k, v in top10_consensus[:10]]
print(f'\nTop 10 共识 (≥3 次): {top10_nums}')

# Top 20 共识 (≥4 次)
c20 = Counter(all_top20)
top20_consensus = [(k, v) for k, v in c20.items() if v >= 4]
top20_consensus.sort(key=lambda x: -x[1])
top20_nums = [k for k, v in top20_consensus[:20]]
print(f'\nTop 20 共识 (≥4 次): {top20_nums}')

multi_model['consensus'] = {
    'top4_consensus_count': dict(top4_consensus),
    'top10_consensus_count': dict(top10_consensus),
    'top20_consensus_count': dict(top20_consensus),
    'recommended_4_dan': recommended_4,
    'top10_consensus': top10_nums,
    'top20_consensus': top20_nums,
}

# ============= 写入 JSON =============
out_path = ROOT / 'data' / 'backtest' / f'{TARGET_PERIOD}_predictions_multi_model.json'
with open(out_path, 'w', encoding='utf-8') as f:
    json.dump(multi_model, f, ensure_ascii=False, indent=2)
print(f'\n✓ 已写入 {out_path.name}')

# V22.D1 单独文件
d1_out = {
    'meta': multi_model['meta'],
    f'next_period_{TARGET_PERIOD}': multi_model[f'next_period_{TARGET_PERIOD}']['V22.D1'],
}
out_path_d1 = ROOT / 'data' / 'backtest' / f'{TARGET_PERIOD}_predictions_v22_d1.json'
with open(out_path_d1, 'w', encoding='utf-8') as f:
    json.dump(d1_out, f, ensure_ascii=False, indent=2)
print(f'✓ 已写入 {out_path_d1.name}')