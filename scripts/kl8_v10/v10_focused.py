"""
KL8 V10 - 精简版: 只用 2-3 特征 + 反向 + 形态约束
基于之前发现: freq(20,负) + freq(100,正) 是最优
"""
import sys
from pathlib import Path
sys.path.insert(0, '/Users/clarkkong/.openclaw/workspace/agents/lucky')
from lottery_data import LotteryData
import pandas as pd
import numpy as np
import json
import itertools

ROOT = Path.home() / 'Library/Mobile Documents/com~apple~CloudDocs/PycharmProjects/Lottery'
ld = LotteryData(ROOT)
df, conf = ld.load('快乐8')
red_cols = [f'红球{i}' for i in range(1, 21)]
data = df[red_cols].astype(int).values
n_periods = len(data)

def feat_freq(i, window):
    if i + window > n_periods: window = n_periods - i
    if window <= 0: return np.zeros(80, dtype=float)
    block = data[i+1:i+1+window]
    flat = block.flatten()
    return np.bincount(flat, minlength=81)[1:81].astype(float)

def feat_repeat(i, span):
    counts = np.zeros(80, dtype=float)
    for j in range(1, span+1):
        if i + j >= n_periods: break
        nums = data[i+j]
        counts[nums-1] += 1
    return counts

def feat_neighbor(i, span, distance=2):
    counts = np.zeros(80, dtype=float)
    for j in range(1, span+1):
        if i + j >= n_periods: break
        last = data[i+j]
        for n in last:
            for d in range(-distance, distance+1):
                if d == 0: continue
                if 1 <= n+d <= 80:
                    counts[n+d-1] += 1
    return counts

def feat_miss(i, max_window):
    miss_count = np.zeros(80, dtype=float)
    for j in range(min(max_window, i+1)):
        nums = data[i+1+j]
        in_period = np.zeros(80, dtype=bool)
        in_period[nums-1] = True
        miss_count = np.where(in_period, 0, miss_count + 1)
    return miss_count

TEST_INDICES = list(range(0, 200))
N_TEST = len(TEST_INDICES)

# 精简特征池
FREQ_POOL = [10, 20, 30, 50, 75, 100, 150, 200, 300, 500]
REPEAT_POOL = [1, 2, 3, 5, 7, 10]
NEIGHBOR_POOL = [1, 2, 3, 5, 7, 10]
MISS_POOL = [20, 30, 50, 100]

def get_feat(feat_name, span, i):
    if feat_name == 'freq': return feat_freq(i, span)
    if feat_name == 'repeat': return feat_repeat(i, span)
    if feat_name == 'neighbor': return feat_neighbor(i, span)
    if feat_name == 'miss': return feat_miss(i, span)

# 预计算
print('预计算特征...')
feat_cache = {}
for w in FREQ_POOL:
    feat_cache[('freq', w)] = np.array([feat_freq(i, w) for i in TEST_INDICES])
for span in REPEAT_POOL:
    feat_cache[('repeat', span)] = np.array([feat_repeat(i, span) for i in TEST_INDICES])
for span in NEIGHBOR_POOL:
    feat_cache[('neighbor', span)] = np.array([feat_neighbor(i, span) for i in TEST_INDICES])
for w in MISS_POOL:
    feat_cache[('miss', w)] = np.array([feat_miss(i, w) for i in TEST_INDICES])

def score_combination(weights_dict, top_n):
    scores = np.zeros((N_TEST, 80))
    for (feat, span), w in weights_dict.items():
        scores += feat_cache[(feat, span)] * w
    hits = []
    for k in range(N_TEST):
        top_idx = np.argsort(-scores[k])[:top_n] + 1
        actual = set(data[TEST_INDICES[k]].tolist())
        top_set = set(top_idx.tolist())
        hits.append(len(top_set & actual))
    h = np.array(hits)
    return h.mean(), (h>=8).mean()*100, (h>=10).mean()*100, (h>=11).mean()*100, h

# ============================================================
# 搜索: 1 freq + 1 repeat + 1 neighbor + 1 miss (4特征)
# 限制: 权重 ∈ {-3, -1, 1, 3}
# 总组合: 10 * 6 * 6 * 4 * 4^4 = 92,160 (5 分钟内跑完)
# ============================================================
print('=' * 70)
print('V10 精细搜索: 1 freq + 1 repeat + 1 neighbor + 1 miss')
print('=' * 70)

weights_options = [-3.0, -1.0, 1.0, 3.0]
best_combos = []
count = 0

for w_f in FREQ_POOL:
    for span_r in REPEAT_POOL:
        for span_n in NEIGHBOR_POOL:
            for w_m in MISS_POOL:
                for wf in weights_options:
                    for wr in weights_options:
                        for wn in weights_options:
                            for wm in weights_options:
                                weights = {
                                    ('freq', w_f): wf,
                                    ('repeat', span_r): wr,
                                    ('neighbor', span_n): wn,
                                    ('miss', w_m): wm,
                                }
                                avg, ge8, ge10, ge11, _ = score_combination(weights, 20)
                                count += 1
                                if avg >= 5.3:
                                    best_combos.append((avg, ge8, ge10, ge11, weights))
                                if count % 20000 == 0:
                                    cur_best = max((c[0] for c in best_combos), default=0)
                                    print(f'  ... {count}/{92160} 评估, best avg={cur_best:.4f}')

best_combos.sort(key=lambda x: -x[0])
print(f'\n总计评估 {count} 个')
print(f'avg≥5.3: {len(best_combos)}')
print(f'\n最优 Top20 (Top20):')
for c in best_combos[:20]:
    label = ' + '.join(f'{k[0]}({k[1]},w={v})' for k, v in c[4].items())
    print(f'  avg={c[0]:.4f} ≥8中 {c[1]:.1f}% ≥10中 {c[2]:.1f}% ≥11中 {c[3]:.1f}% | {label}')

# 评估最优组合在不同 TopN
if best_combos:
    best_w = best_combos[0][4]
    print()
    print('=' * 70)
    print('最优 V10 组合在不同 TopN 上的表现')
    print('=' * 70)
    label = ' + '.join(f'{k[0]}({k[1]},w={v})' for k, v in best_w.items())
    print(f'配置: {label}')
    for top_n in [9, 12, 15, 18, 20, 25, 30, 35, 40]:
        avg, ge8, ge10, ge11, _ = score_combination(best_w, top_n)
        print(f'  Top{top_n}: 平均 {avg:.3f}/20 ({avg*5:.2f}%)  ≥8中 {ge8:.1f}%  ≥10中 {ge10:.1f}%  ≥11中 {ge11:.1f}%')

# 保存
out = {
    'method': 'v10_4feat_neg',
    'test_periods': 200,
    'total_evaluated': count,
    'top_combos': [
        {
            'config': {f'{k[0]}_{k[1]}': v for k, v in c[4].items()},
            'avg': float(c[0]),
            'ge8': float(c[1]),
            'ge10': float(c[2]),
            'ge11': float(c[3]),
        }
        for c in best_combos[:30]
    ]
}
with open('/tmp/kl8_v10_top.json', 'w', encoding='utf-8') as f:
    json.dump(out, f, indent=2, ensure_ascii=False)
print()
print('结果保存到 /tmp/kl8_v10_top.json')