"""
KL8 V10 向量化版: 用矩阵运算同时评估所有组合
对每个候选权重组合,一次性算出所有测试期的得分
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

# 预计算
print('预计算特征...')
feat_data = {}  # (feat, span) -> (N_TEST, 80) numpy array
for w in FREQ_POOL:
    feat_data[('freq', w)] = np.array([feat_freq(i, w) for i in TEST_INDICES])
for span in REPEAT_POOL:
    feat_data[('repeat', span)] = np.array([feat_repeat(i, span) for i in TEST_INDICES])
for span in NEIGHBOR_POOL:
    feat_data[('neighbor', span)] = np.array([feat_neighbor(i, span) for i in TEST_INDICES])
for w in MISS_POOL:
    feat_data[('miss', w)] = np.array([feat_miss(i, w) for i in TEST_INDICES])

# 实际开奖矩阵 (N_TEST, 80)
actual_matrix = np.zeros((N_TEST, 80), dtype=bool)
for k, i in enumerate(TEST_INDICES):
    actual_matrix[k, data[i]-1] = True

print(f'特征池: freq={len(FREQ_POOL)}, repeat={len(REPEAT_POOL)}, neighbor={len(NEIGHBOR_POOL)}, miss={len(MISS_POOL)}')

# ============================================================
# 评估函数 (向量化)
# ============================================================
def eval_config_fast(weights, top_n):
    """
    weights: dict {(feat, span): weight}
    返回: avg_hits, ge8_pct, ge10_pct, ge11_pct
    """
    scores = np.zeros((N_TEST, 80))
    for (feat, span), w in weights.items():
        scores += feat_data[(feat, span)] * w
    # Top N 索引
    top_idx = np.argsort(-scores, axis=1)[:, :top_n]  # (N_TEST, top_n)
    # 命中数
    hits = np.array([actual_matrix[k, top_idx[k]].sum() for k in range(N_TEST)])
    return hits.mean(), (hits>=8).mean()*100, (hits>=10).mean()*100, (hits>=11).mean()*100, hits

# ============================================================
# 测试: 验证 V3-V9 模型的表现
# ============================================================
print('=' * 70)
print('V3-V9 等价方案验证 (基于反向特征)')
print('=' * 70)
# V3 简化: freq(50) + repeat(1) + neighbor(1) + miss(50) 全正向
v3 = {('freq', 50): 1.0, ('repeat', 1): 1.0, ('neighbor', 1): 1.0, ('miss', 50): -1.0}
avg, ge8, ge10, ge11, _ = eval_config_fast(v3, 20)
print(f'V3 等价 (频次+重号+邻号+漏号反向): avg={avg:.4f} ≥8中 {ge8:.1f}% ≥10中 {ge10:.1f}%')

# V3 全正向
v3_pos = {('freq', 50): 1.0, ('repeat', 1): 1.0, ('neighbor', 1): 1.0, ('miss', 50): 1.0}
avg, ge8, ge10, ge11, _ = eval_config_fast(v3_pos, 20)
print(f'V3 全正向 (V3-V9 默认方向): avg={avg:.4f} ≥8中 {ge8:.1f}% ≥10中 {ge10:.1f}%')

# ============================================================
# 重点搜索: 2-3 特征负权重组合
# ============================================================
print()
print('=' * 70)
print('2-3 特征组合 (允许负权重)')
print('=' * 70)

# 2 特征: freq + repeat/repeat/neighbor/miss
best_results = []

# 2 特征: freq + 另一个
for w_f in FREQ_POOL:
    for (feat2_name, span2_pool) in [('repeat', REPEAT_POOL), ('neighbor', NEIGHBOR_POOL), ('miss', MISS_POOL)]:
        for span2 in span2_pool:
            for wf in [-3, -1, 1, 3]:
                for w2 in [-3, -1, 1, 3]:
                    weights = {('freq', w_f): wf, (feat2_name, span2): w2}
                    avg, ge8, ge10, ge11, _ = eval_config_fast(weights, 20)
                    if avg >= 5.4:
                        best_results.append((avg, ge8, ge10, ge11, weights))

# 3 特征: freq + repeat + neighbor
for w_f in FREQ_POOL:
    for span_r in REPEAT_POOL:
        for span_n in NEIGHBOR_POOL:
            for wf in [-3, -1, 1, 3]:
                for wr in [-3, -1, 1, 3]:
                    for wn in [-3, -1, 1, 3]:
                        weights = {('freq', w_f): wf, ('repeat', span_r): wr, ('neighbor', span_n): wn}
                        avg, ge8, ge10, ge11, _ = eval_config_fast(weights, 20)
                        if avg >= 5.4:
                            best_results.append((avg, ge8, ge10, ge11, weights))

best_results.sort(key=lambda x: -x[0])
print(f'总计 {len(best_results)} 个 avg≥5.4 组合')
print(f'\n最优 Top20 (Top20):')
for c in best_results[:20]:
    label = ' + '.join(f'{k[0]}({k[1]},w={v})' for k, v in c[4].items())
    print(f'  avg={c[0]:.4f} ≥8中 {c[1]:.1f}% ≥10中 {c[2]:.1f}% ≥11中 {c[3]:.1f}% | {label}')

# 在最优组合的不同 TopN 上评估
if best_results:
    best_w = best_results[0][4]
    print()
    print('=' * 70)
    print('最优 V10 组合在不同 TopN')
    print('=' * 70)
    label = ' + '.join(f'{k[0]}({k[1]},w={v})' for k, v in best_w.items())
    print(f'配置: {label}')
    for top_n in [9, 12, 15, 18, 20, 25, 30]:
        avg, ge8, ge10, ge11, _ = eval_config_fast(best_w, top_n)
        print(f'  Top{top_n}: 平均 {avg:.3f}/20 ({avg*5:.2f}%)  ≥8中 {ge8:.1f}%  ≥10中 {ge10:.1f}%  ≥11中 {ge11:.1f}%')

# 保存
out = {
    'method': 'v10_vectorized_2_3feat',
    'test_periods': 200,
    'total_top_combos': len(best_results),
    'top_combos': [
        {
            'config': {f'{k[0]}_{k[1]}': v for k, v in c[4].items()},
            'avg': float(c[0]),
            'ge8': float(c[1]),
            'ge10': float(c[2]),
            'ge11': float(c[3]),
        }
        for c in best_results[:30]
    ]
}
with open('/tmp/kl8_v10_vectorized.json', 'w', encoding='utf-8') as f:
    json.dump(out, f, indent=2, ensure_ascii=False)
print()
print('结果保存到 /tmp/kl8_v10_vectorized.json')