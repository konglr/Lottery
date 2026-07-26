"""
KL8 V3-V9 贪心前向选择优化
- 特征池: freq, repeat, neighbor, miss, span, recent_freq
- 每个特征多个窗口
- 目标: Top20 平均命中率最大化
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

# ============= 特征函数 =============
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

# 特征池: (feature_name, span, weight_candidate)
FEATURE_POOL = []
# freq 不同窗口
for w in [20, 30, 50, 75, 100, 150, 200, 300]:
    FEATURE_POOL.append(('freq', w, 1.0))
# repeat 不同 span
for span in [1, 2, 3, 5]:
    FEATURE_POOL.append(('repeat', span, 5.0))
# neighbor 不同 span
for span in [1, 2, 3, 5]:
    FEATURE_POOL.append(('neighbor', span, 3.0))
# miss 不同窗口 (权重可正可负)
for w in [30, 50, 100]:
    FEATURE_POOL.append(('miss', w, -1.0))  # 漏号反向
    FEATURE_POOL.append(('miss', w, 1.0))   # 漏号正向

print(f'特征池大小: {len(FEATURE_POOL)}')

def get_feat(feat_name, span, i):
    if feat_name == 'freq': return feat_freq(i, span)
    if feat_name == 'repeat': return feat_repeat(i, span)
    if feat_name == 'neighbor': return feat_neighbor(i, span)
    if feat_name == 'miss': return feat_miss(i, span)
    raise ValueError(feat_name)

# ============= 评分 =============
def score_combination(i, config):
    """config: [(feat, span, weight), ...]"""
    scores = np.zeros(80, dtype=float)
    for feat, span, weight in config:
        s = get_feat(feat, span, i)
        scores += s * weight
    return scores

# ============= 预计算所有特征(加速回测) =============
print('预计算特征矩阵...')
TEST_INDICES = list(range(0, 200))

# feat_cache[feat_name][span] = list of (80,) arrays
feat_cache = {}
for feat_name, span, _ in FEATURE_POOL:
    if feat_name not in feat_cache: feat_cache[feat_name] = {}
    if span not in feat_cache[feat_name]:
        feat_cache[feat_name][span] = []
    for i in TEST_INDICES:
        feat_cache[feat_name][span].append(get_feat(feat_name, span, i))

def score_cached(i_idx, config):
    """i_idx 是 TEST_INDICES 里的位置"""
    scores = np.zeros(80, dtype=float)
    for feat, span, weight in config:
        scores += feat_cache[feat][span][i_idx] * weight
    return scores

# ============= 评估 TopN 命中率 =============
def eval_topn(config, top_n):
    """返回 TopN 的 (avg_hits, ge8_pct, ge10_pct, ge11_pct)"""
    hits = []
    for i_idx in range(len(TEST_INDICES)):
        scores = score_cached(i_idx, config)
        top_idx = np.argsort(-scores)[:top_n] + 1
        actual = set(data[TEST_INDICES[i_idx]].tolist())
        top_set = set(top_idx.tolist())
        hits.append(len(top_set & actual))
    h = np.array(hits)
    return h.mean(), (h>=8).mean()*100, (h>=10).mean()*100, (h>=11).mean()*100

# ============= 贪心前向选择 =============
print('=' * 70)
print('贪心前向选择 - 目标: Top20 最大化 avg_hits')
print('=' * 70)
selected = []  # 当前最优组合
best_avg = 0
history = []

for step in range(8):
    print(f'\nStep {step+1}: 当前组合 {len(selected)} 个特征, best_avg={best_avg:.4f}')
    candidates = []
    for feat_name, span, weight in FEATURE_POOL:
        new_config = selected + [(feat_name, span, weight)]
        avg, ge8, ge10, ge11 = eval_topn(new_config, 20)
        candidates.append((avg, ge8, ge10, ge11, feat_name, span, weight))
    
    # 选最优
    candidates.sort(key=lambda x: -x[0])
    best_cand = candidates[0]
    if best_cand[0] > best_avg:
        best_avg = best_cand[0]
        selected.append((best_cand[4], best_cand[5], best_cand[6]))
        history.append({
            'step': step+1,
            'added': (best_cand[4], best_cand[5], best_cand[6]),
            'avg': best_cand[0],
            'ge8': best_cand[1],
            'ge10': best_cand[2],
            'ge11': best_cand[3],
            'config': list(selected),
        })
        print(f'  添加: {best_cand[4]}({best_cand[5]},w={best_cand[6]}) → avg={best_cand[0]:.4f} ≥8中 {best_cand[1]:.1f}% ≥10中 {best_cand[2]:.1f}%')
    else:
        print(f'  无改进,停止。当前最优: {best_avg:.4f}')
        break

print()
print('=' * 70)
print('最终最优组合 (Top20)')
print('=' * 70)
for feat, span, weight in selected:
    print(f'  {feat}(span={span}, weight={weight})')

# 同时评估其他 TopN
print()
print('=' * 70)
print('最优组合在不同 TopN 上的表现')
print('=' * 70)
for top_n in [9, 12, 15, 18, 20, 25, 30]:
    avg, ge8, ge10, ge11 = eval_topn(selected, top_n)
    print(f'  Top{top_n}: 平均 {avg:.3f}/20 ({avg*5:.2f}%)  ≥8中 {ge8:.1f}%  ≥10中 {ge10:.1f}%  ≥11中 {ge11:.1f}%')

# 与基线对比
print()
print('=' * 70)
print('对比基线: 朴素频次 W=100')
print('=' * 70)
baseline_config = [('freq', 100, 1.0)]
for top_n in [9, 15, 20, 25, 30]:
    avg, ge8, ge10, ge11 = eval_topn(baseline_config, top_n)
    print(f'  Top{top_n}: 平均 {avg:.3f}/20 ({avg*5:.2f}%)  ≥8中 {ge8:.1f}%  ≥10中 {ge10:.1f}%')

# 保存
out = {
    'method': 'greedy_forward_selection',
    'test_periods': 200,
    'feature_pool_size': len(FEATURE_POOL),
    'best_config': selected,
    'best_top20_avg': best_avg,
    'history': history,
}
with open('/tmp/kl8_v3v9_greedy_result.json', 'w', encoding='utf-8') as f:
    json.dump(out, f, indent=2, ensure_ascii=False)
print()
print('结果保存到 /tmp/kl8_v3v9_greedy_result.json')