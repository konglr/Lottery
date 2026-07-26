"""
KL8 V3-V9 V3: 完整网格 + 反向特征 + 形态过滤
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

# 特征函数
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

FEATURE_POOL = []
for w in [20, 30, 50, 75, 100, 150, 200, 300]:
    FEATURE_POOL.append(('freq', w))
for span in [1, 2, 3, 5, 7]:
    FEATURE_POOL.append(('repeat', span))
for span in [1, 2, 3, 5, 7]:
    FEATURE_POOL.append(('neighbor', span))
for w in [20, 30, 50, 100]:
    FEATURE_POOL.append(('miss', w))

print(f'特征池: {len(FEATURE_POOL)}')

def get_feat(feat_name, span, i):
    if feat_name == 'freq': return feat_freq(i, span)
    if feat_name == 'repeat': return feat_repeat(i, span)
    if feat_name == 'neighbor': return feat_neighbor(i, span)
    if feat_name == 'miss': return feat_miss(i, span)

# 预计算
feat_cache = {}
for feat_name, span in FEATURE_POOL:
    key = (feat_name, span)
    feat_cache[key] = np.array([get_feat(feat_name, span, i) for i in TEST_INDICES])

def score_combination(weights_dict, top_n):
    n_test = len(TEST_INDICES)
    scores = np.zeros((n_test, 80))
    for (feat, span), w in weights_dict.items():
        scores += feat_cache[(feat, span)] * w
    hits = []
    for k in range(n_test):
        top_idx = np.argsort(-scores[k])[:top_n] + 1
        actual = set(data[TEST_INDICES[k]].tolist())
        top_set = set(top_idx.tolist())
        hits.append(len(top_set & actual))
    h = np.array(hits)
    return h.mean(), (h>=8).mean()*100, (h>=10).mean()*100, (h>=11).mean()*100, h

# ============================================================
# 关键测试: 反向特征 (负权重) 是否真的优于正向
# ============================================================
print('=' * 70)
print('测试 1: 单特征 正向 vs 反向权重')
print('=' * 70)
print(f'{"特征":>15} {"W":>4} {"方向":>6} {"Top20 平均":>12} {"≥8中":>6} {"≥10中":>7}')

for feat_name, span in FEATURE_POOL:
    for w_dir in [('正', 1.0), ('负', -1.0)]:
        label, w = w_dir
        weights = {(feat_name, span): w}
        avg, ge8, ge10, ge11, _ = score_combination(weights, 20)
        print(f'{feat_name+"("+str(span)+")":>15} {span:>4} {label:>6} {avg:>12.4f} {ge8:>5.1f}% {ge10:>6.1f}%')

# ============================================================
# 测试 2: 2-特征组合的负+正组合
# ============================================================
print()
print('=' * 70)
print('测试 2: 双特征负+正组合 (Top20)')
print('=' * 70)
print(f'{"配置":>50} {"avg":>6} {"≥8中":>6} {"≥10中":>7}')

best_combos = []
# 穷举所有 (feat1, feat2) 配 (w1, w2)
for (f1, s1) in FEATURE_POOL:
    for (f2, s2) in FEATURE_POOL:
        if (f1, s1) >= (f2, s2): continue  # 去重
        for w1 in [-3.0, -1.0, 1.0, 3.0]:
            for w2 in [-3.0, -1.0, 1.0, 3.0]:
                weights = {(f1, s1): w1, (f2, s2): w2}
                avg, ge8, ge10, ge11, _ = score_combination(weights, 20)
                if avg >= 5.3:  # 只打印 Top20 平均 ≥5.3 的
                    label = f'{f1}({s1},w={w1}) + {f2}({s2},w={w2})'
                    print(f'{label:>50} {avg:>6.3f} {ge8:>5.1f}% {ge10:>6.1f}%')
                    best_combos.append((avg, ge8, ge10, ge11, weights))

best_combos.sort(key=lambda x: -x[0])
print(f'\n总计 {len(best_combos)} 个组合 avg≥5.3')
print(f'最优 Top5 (Top20):')
for c in best_combos[:5]:
    label = ' + '.join(f'{k[0]}({k[1]},w={v})' for k, v in c[4].items())
    print(f'  {label} → avg={c[0]:.4f} ≥8中 {c[1]:.1f}% ≥10中 {c[2]:.1f}%')

# 保存
out = {
    'method': 'grid_search_2feat_with_neg',
    'test_periods': 200,
    'top_combos': [
        {
            'config': {f'{k[0]}_{k[1]}': v for k, v in c[4].items()},
            'avg': float(c[0]),
            'ge8': float(c[1]),
            'ge10': float(c[2]),
            'ge11': float(c[3]),
        }
        for c in best_combos[:20]
    ]
}
with open('/tmp/kl8_v3v9_grid_v3.json', 'w', encoding='utf-8') as f:
    json.dump(out, f, indent=2, ensure_ascii=False)
print()
print('结果保存到 /tmp/kl8_v3v9_grid_v3.json')