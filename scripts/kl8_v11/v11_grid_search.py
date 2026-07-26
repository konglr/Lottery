"""
KL8 V11 完整版: V10 + V3 + 4 分类 + 4 区位 综合
"""
import sys
from pathlib import Path
sys.path.insert(0, '/Users/clarkkong/.openclaw/workspace/agents/lucky')
from lottery_data import LotteryData
import pandas as pd
import numpy as np
import json

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
                if 1 <= n+d <= 80: counts[n+d-1] += 1
    return counts

def classify_freq(freq, ratios):
    rank = np.argsort(-freq)
    n_total = 80
    n_high = int(ratios[0] * n_total)
    n_mid_high = int(ratios[1] * n_total)
    n_mid_low = int(ratios[2] * n_total)
    n_low = n_total - n_high - n_mid_high - n_mid_low
    classes = np.zeros(80, dtype=int)
    classes[rank[:n_high]] = 1
    classes[rank[n_high:n_high+n_mid_high]] = 2
    classes[rank[n_high+n_mid_high:n_high+n_mid_high+n_mid_low]] = 3
    classes[rank[n_high+n_mid_high+n_mid_low:]] = 4
    return classes

TEST_INDICES = list(range(0, 200))
N_TEST = len(TEST_INDICES)

# 预计算特征
print('预计算特征...')
F_FREQ = {}
for w in [10, 20, 30, 50, 75, 100, 150, 200, 300]:
    F_FREQ[w] = np.array([feat_freq(i, w) for i in TEST_INDICES])
F_REPEAT = {}
for span in [1, 2, 3, 5, 7, 10]:
    F_REPEAT[span] = np.array([feat_repeat(i, span) for i in TEST_INDICES])
F_NEIGHBOR = {}
for span in [1, 2, 3, 5, 7, 10]:
    F_NEIGHBOR[span] = np.array([feat_neighbor(i, span) for i in TEST_INDICES])

# 预计算 4 分类
print('预计算 4 分类...')
F_CLASS = {}
for w in [30, 50, 75, 100, 150]:
    for ratios in [
        [0.25, 0.25, 0.25, 0.25],
        [0.20, 0.30, 0.30, 0.20],
        [0.30, 0.20, 0.20, 0.30],
        [0.15, 0.35, 0.35, 0.15],
    ]:
        key = (w, tuple(ratios))
        F_CLASS[key] = np.zeros((N_TEST, 80), dtype=int)
        for k in range(N_TEST):
            F_CLASS[key][k] = classify_freq(F_FREQ[w][k], ratios)

# 4 区位 (固定)
ZONE_CLASS = np.array([((n-1)//20) + 1 for n in range(1, 81)], dtype=float)

def eval_scores(scores, top_n):
    hits = []
    for k in range(N_TEST):
        top_idx = np.argsort(-scores[k])[:top_n] + 1
        actual = set(data[TEST_INDICES[k]].tolist())
        hits.append(len(set(top_idx.tolist()) & actual))
    h = np.array(hits)
    return h.mean(), (h>=8).mean()*100, (h>=10).mean()*100, (h>=11).mean()*100, h

# ============================================================
# 关键测试: V11 综合特征
# ============================================================
print('=' * 80)
print('V11 综合特征组合 (V10 + 4 分类 + 4 区位)')
print('=' * 80)

# 基础 V10 分数
v10_base = F_FREQ[10] * -3.0 + F_REPEAT[3] * -3.0 + F_NEIGHBOR[5] * -3.0

# V3 基线 (正向)
v3_base = F_FREQ[50] * 1.0

# 探索: V10 + V3 混合 + 4 分类 + 4 区位
best = []
for w_v10 in [-1.0, -0.5, 0.0, 0.5, 1.0]:
    for w_v3 in [0.0, 0.5, 1.0]:
        for (w_cls, ratios) in [(50, [0.25]*4), (100, [0.25]*4), (100, [0.30, 0.20, 0.20, 0.30])]:
            for w_class in [-2.0, -1.0, 0.0, 1.0, 2.0]:
                for w_zone in [-2.0, -1.0, 0.0, 1.0, 2.0]:
                    scores = (v10_base * w_v10 +
                              v3_base * w_v3 +
                              F_CLASS[(w_cls, tuple(ratios))].astype(float) * w_class +
                              ZONE_CLASS * w_zone)
                    avg, ge8, ge10, ge11, _ = eval_scores(scores, 20)
                    best.append((avg, ge8, ge10, ge11, w_v10, w_v3, w_cls, ratios, w_class, w_zone))

best.sort(key=lambda x: -x[0])
print(f'总搜索 {len(best)} 个组合')
print(f'\n最优 Top15:')
print(f'{"v10_w":>6} {"v3_w":>5} {"cls_W":>6} {"class_w":>7} {"zone_w":>7} {"avg":>6} {"≥8中":>6} {"≥10中":>7} {"≥11中":>7}')
for c in best[:15]:
    print(f'{c[4]:>6.1f} {c[5]:>5.1f} {c[6]:>6} {c[8]:>7.1f} {c[9]:>7.1f} {c[0]:>6.3f} {c[1]:>5.1f}% {c[2]:>6.1f}% {c[3]:>6.1f}%')

# 保存
out = {
    'method': 'v11_comprehensive',
    'test_periods': 200,
    'best_combos': [
        {
            'config': {
                'v10_weight': c[4], 'v3_weight': c[5],
                'classify_window': c[6], 'classify_ratios': c[7],
                'class_weight': c[8], 'zone_weight': c[9],
            },
            'avg': float(c[0]), 'ge8': float(c[1]), 'ge10': float(c[2]), 'ge11': float(c[3])
        }
        for c in best[:30]
    ]
}
with open('/tmp/kl8_v11_grid.json', 'w', encoding='utf-8') as f:
    json.dump(out, f, indent=2, ensure_ascii=False)
print()
print('结果保存到 /tmp/kl8_v11_grid.json')