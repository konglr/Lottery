"""
KL8 V16.E: 条件特征追随 (Conditional Trend Follower)
======================================================
V16 发现的洞察: 反转胜出 (B 模式 51% 胜 C)
但 A 模式 (25% 胜 C) 也很差 — 不是所有特征都该反转
本实验: 对每个特征单独学"上期信号 → 下期结果"的方向
  - 训练: 200 期, 用上期每个特征值, 看哪些特征预测下期相同方向 vs 反方向更准
  - 测试: 200 期 (滚动, 防过拟合)
  - 推断: 根据学到的方向, 对每个特征用对应的 mult (±1)

预期: 比 A 全追 / B 全反 都要好
"""
import sys
from pathlib import Path
sys.path.insert(0, '/Users/clarkkong/.openclaw/workspace/agents/lucky')
from lottery_data import LotteryData
import pandas as pd
import numpy as np
import json
from collections import Counter

ROOT = Path.home() / 'Library/Mobile Documents/com~apple~CloudDocs/PycharmProjects/Lottery'
ld = LotteryData(ROOT)
df, conf = ld.load('快乐8')
red_cols = [f'红球{i}' for i in range(1, 21)]
data = df[red_cols].astype(int).values
n_periods = len(data)
print(f'快乐8 数据: {n_periods} 期')

# 复用 v16 的特征函数 (拷贝过来)
def feat_freq(i, window):
    if i + window > n_periods: window = n_periods - i
    if window <= 0: return np.zeros(80, dtype=float)
    block = data[i+1:i+1+window]
    return np.bincount(block.flatten(), minlength=81)[1:81].astype(float)

def feat_repeat(i, span):
    counts = np.zeros(80, dtype=float)
    for j in range(1, span+1):
        if i + j >= n_periods: break
        nums = data[i+j]
        counts[nums-1] += 1
    return np.clip(counts, 0, 1)

def feat_neighbor(i, span, distance=1):
    covered = np.zeros(80, dtype=float)
    for j in range(1, span+1):
        if i + j >= n_periods: break
        for n in data[i+j]:
            for d in range(-distance, distance+1):
                if d == 0: continue
                if 1 <= n+d <= 80: covered[n+d-1] += 1
    return covered

def feat_miss(i, window):
    miss = np.zeros(80, dtype=float)
    for j in range(1, window+1):
        if i + j >= n_periods: break
        in_period = np.zeros(80, dtype=bool)
        in_period[data[i+j]-1] = True
        miss = np.where(in_period, 0, miss + 1)
    return miss

# 预计算
N_TEST = 200
TEST_INDICES = list(range(0, N_TEST))
print('预计算特征...')
F_FREQ_10 = np.array([feat_freq(i, 10) for i in TEST_INDICES])
F_FREQ_5 = np.array([feat_freq(i, 5) for i in TEST_INDICES])
F_REPEAT_3 = np.array([feat_repeat(i, 3) for i in TEST_INDICES])
F_REPEAT_1 = np.array([feat_repeat(i, 1) for i in TEST_INDICES])
F_NEIGHBOR_5 = np.array([feat_neighbor(i, 5, 1) for i in TEST_INDICES])
F_NEIGHBOR_3 = np.array([feat_neighbor(i, 3, 1) for i in TEST_INDICES])
F_MISS_30 = np.array([feat_miss(i, 30) for i in TEST_INDICES])

# 特征检测函数 (同 v16)
def detect_features(i):
    last = data[i]
    f = {}
    if i + 3 < n_periods:
        prev_3 = set(data[i+1].tolist()) | set(data[i+2].tolist()) | set(data[i+3].tolist())
        f['F1_repeat_density'] = len(set(last.tolist()) & prev_3) / 20.0
    else:
        f['F1_repeat_density'] = 0.25
    if i + 3 < n_periods:
        nbrs = set()
        for n in last:
            for d in [-2, -1, 1, 2]:
                if 1 <= n+d <= 80: nbrs.add(n+d)
        prev_3 = set(data[i+1].tolist()) | set(data[i+2].tolist()) | set(data[i+3].tolist())
        f['F2_neighbor_density'] = len(nbrs & prev_3) / 80.0
    else:
        f['F2_neighbor_density'] = 0.2
    f['F3_consec_count'] = sum(1 for n in last if n+1 in set(last.tolist())) / 20.0
    f['F4_big_ratio'] = sum(1 for n in last if n > 40) / 20.0
    f['F5_small_ratio'] = sum(1 for n in last if n <= 20) / 20.0
    f['F6_odd_ratio'] = sum(1 for n in last if n % 2 == 1) / 20.0
    f['F7_sum_norm'] = (last.sum() - 800) / 80.0
    f['F8_span_norm'] = (last.max() - last.min() - 55) / 15.0
    if i + 5 < n_periods:
        prev_5 = set(data[i+5].tolist())
        f['F9_rebound'] = len(set(last.tolist()) & prev_5) / 20.0
    else:
        f['F9_rebound'] = 0.25
    if i + 30 < n_periods:
        cold_count = 0
        for n in last:
            miss_n = sum(1 for j in range(i+1, i+31) if n not in data[j])
            if miss_n >= 20: cold_count += 1
        f['F10_cold_recover'] = cold_count / 20.0
    else:
        f['F10_cold_recover'] = 0.05
    z = [0, 0, 0, 0]
    for n in last:
        if n <= 20: z[0] += 1
        elif n <= 40: z[1] += 1
        elif n <= 60: z[2] += 1
        else: z[3] += 1
    f['F11_zone_max'] = max(z) / 20.0
    f['F11_zone_min'] = min(z) / 20.0
    f['F11_zone_idx'] = z.index(max(z))
    sorted_last = sorted(last.tolist())
    gaps = [sorted_last[j+1] - sorted_last[j] for j in range(19)]
    f['F12_avg_gap'] = np.mean(gaps) / 4.0
    if i + 30 < n_periods:
        freq30 = feat_freq(i, 30)
        f['F13_freq_concentration'] = freq30.max() / freq30.mean()
    else:
        f['F13_freq_concentration'] = 1.0
    return f

print('缓存特征...')
FEATURES = [detect_features(i) for i in TEST_INDICES]

# =================== 关键: 学习每个特征的"方向" ===================
# 对每个特征: 用训练集 (前 100 期) 看上期特征高/低,下期命中率是延续还是反转
# 训练: indices [0..100)
# 测试: indices [100..200)

FEATURE_KEYS = ['F1_repeat_density', 'F2_neighbor_density', 'F3_consec_count',
                'F4_big_ratio', 'F5_small_ratio', 'F6_odd_ratio',
                'F9_rebound', 'F10_cold_recover',
                'F11_zone_max', 'F12_avg_gap', 'F13_freq_concentration']

def get_avg(key):
    vals = [f[key] for f in FEATURES]
    return np.mean(vals), np.std(vals)

def direction(val, avg, std, thresh=0.5):
    if val > avg + thresh * std: return 1
    if val < avg - thresh * std: return -1
    return 0

# 对每个特征, 学"高信号 + A方向权" vs "高信号 + B方向权" 的 Top20 命中率
print()
print('=' * 80)
print('学习每个特征的最优方向 (前 100 期训练, 后 100 期测试)')
print('=' * 80)

train_indices = list(range(0, 100))
test_indices = list(range(100, 200))

def base_score(i):
    """V10 反向基线分数"""
    return (F_FREQ_10[i] * -3.0 +
            F_REPEAT_3[i] * -3.0 +
            F_NEIGHBOR_5[i] * -3.0)

def apply_feature_w(f, key, mult):
    """根据特征值方向, 对 base 分数调整"""
    w = {'freq': 1.0, 'repeat': 1.0, 'neighbor': 1.0, 'miss': 1.0}
    zone_mask = np.ones(80, dtype=float)

    avg, std = get_avg(key)
    d = direction(f[key], avg, std)
    if d == 0: return w, zone_mask  # 无信号, 不动

    # 不同特征 → 不同权重 / mask
    if key == 'F1_repeat_density':
        # 重号密度高 → repeat 权重要不要加重?
        w['repeat'] *= (1 + mult * 0.5 * d)
        w['freq'] *= (1 - mult * 0.3 * d)
    elif key == 'F2_neighbor_density':
        w['neighbor'] *= (1 + mult * 0.6 * d)
    elif key == 'F3_consec_count':
        w['neighbor'] *= (1 + mult * 0.4 * d)
    elif key == 'F4_big_ratio':
        zone_mask[60:80] *= (1 + mult * 0.5 * d)
    elif key == 'F5_small_ratio':
        zone_mask[0:20] *= (1 + mult * 0.5 * d)
    elif key == 'F6_odd_ratio':
        # 不直接调权, 用 mask 偏斜
        for n in range(1, 81, 2):
            zone_mask[n-1] *= (1 + mult * 0.3 * d)
    elif key == 'F9_rebound':
        w['freq'] *= (1 + mult * 0.4 * d)
    elif key == 'F10_cold_recover':
        w['miss'] *= (1 + mult * 0.8 * d)
    elif key == 'F11_zone_max':
        zone_idx = int(f['F11_zone_idx'])
        zone_mask[zone_idx*20:(zone_idx+1)*20] *= (1 + mult * 0.5 * d)
    elif key == 'F12_avg_gap':
        w['neighbor'] *= (1 + mult * 0.3 * d)
    elif key == 'F13_freq_concentration':
        w['freq'] *= (1 - mult * 0.4 * d)

    for k in ['freq', 'repeat', 'neighbor', 'miss']:
        w[k] = max(0.05, w[k])
    return w, zone_mask

def single_feature_score(i, key, mult):
    """只用单个特征调整的分数"""
    f = FEATURES[i]
    w, zmask = apply_feature_w(f, key, mult)
    base = base_score(i)
    miss_part = F_MISS_30[i] * 0.1 * w['miss']
    return (base + miss_part) * zmask

def eval_single_feature(indices, key, mult, top_n=20):
    """评估"只用单特征 + 方向"在 indices 上的 TopN 命中率"""
    hits = []
    for k in indices:
        i = TEST_INDICES[k]
        actual = set(data[i].tolist())
        s = single_feature_score(i, key, mult)
        sel = set((np.argsort(-s)[:top_n] + 1).tolist())
        hits.append(len(sel & actual))
    return float(np.mean(hits))

# 对每个特征, 在训练集上学 A vs B 的最优方向
best_dirs = {}
print(f'{"特征":<25} {"A 方向":>10} {"B 方向":>10} {"最优":>8} {"方向 mult":>8}')
print('-' * 70)

for key in FEATURE_KEYS:
    a_score = eval_single_feature(train_indices, key, mult=+1)
    b_score = eval_single_feature(train_indices, key, mult=-1)
    # 选择训练集最优方向
    if a_score > b_score:
        best_dirs[key] = +1
        best_label = 'A 延续'
    else:
        best_dirs[key] = -1
        best_label = 'B 反转'
    print(f'{key:<25} {a_score:>10.3f} {b_score:>10.3f} {"→ " + best_label:>18}')

# =================== V16.E 评分: 用学到的方向组合 ===================
def score_v16e(i):
    """V16.E: 对每个特征应用训练出的最优方向"""
    f = FEATURES[i]
    w = {'freq': 1.0, 'repeat': 1.0, 'neighbor': 1.0, 'miss': 1.0}
    zone_mask = np.ones(80, dtype=float)
    for key in FEATURE_KEYS:
        mult = best_dirs[key]
        w_k, zm_k = apply_feature_w(f, key, mult)
        # 组合权重 (乘法)
        for k in ['freq', 'repeat', 'neighbor', 'miss']:
            w[k] *= w_k[k]
        zone_mask *= zm_k
    for k in ['freq', 'repeat', 'neighbor', 'miss']:
        w[k] = max(0.05, w[k])
    base = base_score(i)
    miss_part = F_MISS_30[i] * 0.1 * w['miss']
    return (base + miss_part) * zone_mask

# =================== 测试 (后 100 期) ===================
print()
print('=' * 80)
print('测试 (后 100 期, 防过拟合)')
print('=' * 80)

def eval_score_at(indices, score_fn, top_n):
    hits = []
    for k in indices:
        i = TEST_INDICES[k]
        actual = set(data[i].tolist())
        s = score_fn(i)
        sel = set((np.argsort(-s)[:top_n] + 1).tolist())
        hits.append(len(sel & actual))
    h = np.array(hits)
    return float(h.mean()), float((h>=8).mean()*100), float((h>=10).mean()*100)

# V16.E
print(f'\n方法                      TopN    avg    命中率    ≥8中    ≥10中')
print('-' * 70)

# V10/C 基线 (复用 v16 结果做对照)
v10_t20 = eval_score_at(test_indices, lambda i: base_score(i), 20)
v10_t30 = eval_score_at(test_indices, lambda i: base_score(i), 30)
print(f'{"V10 基线 (test set)":<24} {20:>5} {v10_t20[0]:>6.3f} {v10_t20[0]/20*100:>7.1f}% {v10_t20[1]:>5.1f}% {v10_t20[2]:>6.1f}%')
print(f'{"V10 基线 (test set)":<24} {30:>5} {v10_t30[0]:>6.3f} {v10_t30[0]/30*100:>7.1f}% {v10_t30[1]:>5.1f}% {v10_t30[2]:>6.1f}%')

# V16.E
e_t20 = eval_score_at(test_indices, score_v16e, 20)
e_t30 = eval_score_at(test_indices, score_v16e, 30)
e_t40 = eval_score_at(test_indices, score_v16e, 40)
print(f'{"V16.E 条件追随":<24} {20:>5} {e_t20[0]:>6.3f} {e_t20[0]/20*100:>7.1f}% {e_t20[1]:>5.1f}% {e_t20[2]:>6.1f}%')
print(f'{"V16.E 条件追随":<24} {30:>5} {e_t30[0]:>6.3f} {e_t30[0]/30*100:>7.1f}% {e_t30[1]:>5.1f}% {e_t30[2]:>6.1f}%')
print(f'{"V16.E 条件追随":<24} {40:>5} {e_t40[0]:>6.3f} {e_t40[0]/40*100:>7.1f}%')

# 全 200 期对比 (训练 + 测试)
print()
print('=' * 80)
print('全 200 期对比')
print('=' * 80)
v10_full = eval_score_at(range(N_TEST), lambda i: base_score(i), 20)
e_full = eval_score_at(range(N_TEST), score_v16e, 20)
v10_full30 = eval_score_at(range(N_TEST), lambda i: base_score(i), 30)
e_full30 = eval_score_at(range(N_TEST), score_v16e, 30)
print(f'{"方法":<24} {"TopN":>5} {"avg":>6} {"命中率":>8} {"≥8中":>6} {"≥10中":>7}')
print('-' * 70)
print(f'{"V10 基线":<24} {20:>5} {v10_full[0]:>6.3f} {v10_full[0]/20*100:>7.1f}% {v10_full[1]:>5.1f}% {v10_full[2]:>6.1f}%')
print(f'{"V16.E 条件追随":<24} {20:>5} {e_full[0]:>6.3f} {e_full[0]/20*100:>7.1f}% {e_full[1]:>5.1f}% {e_full[2]:>6.1f}%')
print(f'{"V10 基线":<24} {30:>5} {v10_full30[0]:>6.3f} {v10_full30[0]/30*100:>7.1f}% {v10_full30[1]:>5.1f}% {v10_full30[2]:>6.1f}%')
print(f'{"V16.E 条件追随":<24} {30:>5} {e_full30[0]:>6.3f} {e_full30[0]/30*100:>7.1f}% {e_full30[1]:>5.1f}% {e_full30[2]:>6.1f}%')

# 头对头 (V16.E vs V10)
wins_e, wins_v10, ties_e = 0, 0, 0
for k in range(N_TEST):
    i = TEST_INDICES[k]
    actual = set(data[i].tolist())
    s_e = score_v16e(i)
    s_v10 = base_score(i)
    h_e = len(set((np.argsort(-s_e)[:20] + 1).tolist()) & actual)
    h_v10 = len(set((np.argsort(-s_v10)[:20] + 1).tolist()) & actual)
    if h_e > h_v10: wins_e += 1
    elif h_e < h_v10: wins_v10 += 1
    else: ties_e += 1
print(f'\nV16.E vs V10 (Top20, 全 200 期逐期):')
print(f'  E 胜: {wins_e} ({wins_e/N_TEST*100:.1f}%)')
print(f'  V10 胜: {wins_v10} ({wins_v10/N_TEST*100:.1f}%)')
print(f'  平: {ties_e} ({ties_e/N_TEST*100:.1f}%)')

# =================== 2026200 期 V16.E 预测 ===================
print()
print('=' * 80)
print('2026200 期 V16.E 预测')
print('=' * 80)

last_period = df.iloc[0]['期号']
target_period = int(last_period) + 1
print(f'上期: {last_period}, 预测目标: {target_period}')

s_e = score_v16e(0)
top20 = sorted((np.argsort(-s_e)[:20] + 1).tolist())
top30 = sorted((np.argsort(-s_e)[:30] + 1).tolist())
print(f'V16.E Top20: {top20}')
print(f'V16.E Top30: {top30}')

# 与 V10 共识
s_v10 = base_score(0)
v10_top20 = set((np.argsort(-s_v10)[:20] + 1).tolist())
consensus = set(top20) & v10_top20
print(f'V16.E ∩ V10 共 {len(consensus)} 个 → {sorted(consensus)}')

# 保存
out = {
    'method': 'v16_conditional_follow',
    'test_periods': 200,
    'philosophy': '对每个特征单独学"延续 vs 反转"方向 (前 100 期训练, 后 100 测试)',
    'best_directions': best_dirs,
    'results_full_200': {
        'V10_20': v10_full, 'V16E_20': e_full,
        'V10_30': v10_full30, 'V16E_30': e_full30,
    },
    'head_to_head_E_vs_V10': {'E_wins': wins_e, 'V10_wins': wins_v10, 'ties': ties_e},
    'prediction_target': str(target_period),
    'predictions': {'top20': top20, 'top30': top30,
                    'consensus_with_V10': sorted(consensus)},
}

with open('/Users/clarkkong/Library/Mobile Documents/com~apple~CloudDocs/PycharmProjects/Lottery/data/backtest/v16_conditional_results.json', 'w', encoding='utf-8') as f:
    json.dump(out, f, indent=2, ensure_ascii=False, default=str)

print()
print('结果保存到 data/backtest/v16_conditional_results.json')