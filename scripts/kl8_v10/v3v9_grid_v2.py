"""
KL8 V3-V9 改进版贪心 - 加入特征去重和权重微调
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

# 预计算
TEST_INDICES = list(range(0, 200))

FEATURE_POOL = []
# freq 不同窗口
for w in [10, 20, 30, 50, 75, 100, 150, 200, 300]:
    FEATURE_POOL.append(('freq', w))
# repeat 不同 span
for span in [1, 2, 3, 5, 7]:
    FEATURE_POOL.append(('repeat', span))
# neighbor 不同 span
for span in [1, 2, 3, 5, 7]:
    FEATURE_POOL.append(('neighbor', span))
# miss 不同窗口
for w in [20, 30, 50, 100]:
    FEATURE_POOL.append(('miss', w))

print(f'特征池: {len(FEATURE_POOL)}')

def get_feat(feat_name, span, i):
    if feat_name == 'freq': return feat_freq(i, span)
    if feat_name == 'repeat': return feat_repeat(i, span)
    if feat_name == 'neighbor': return feat_neighbor(i, span)
    if feat_name == 'miss': return feat_miss(i, span)

# 预计算 (feat, span) → (n_test, 80) 矩阵
feat_cache = {}
for feat_name, span in FEATURE_POOL:
    key = (feat_name, span)
    feat_cache[key] = []
    for i in TEST_INDICES:
        feat_cache[key].append(get_feat(feat_name, span, i))
    feat_cache[key] = np.array(feat_cache[key])  # (n_test, 80)

# ============= 评分 =============
def score_combination(weights_dict, top_n):
    """
    weights_dict: {(feat, span): weight}
    返回: hits 数组,平均命中率,等
    """
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

# ============= 贪心 (禁止重复特征) =============
print('=' * 70)
print('贪心前向选择 V2 (禁止重复特征)')
print('=' * 70)

selected = {}  # {(feat, span): weight}
best_avg = 0
history = []

for step in range(15):
    candidates = []
    for feat_name, span in FEATURE_POOL:
        if (feat_name, span) in selected: continue
        for w in [0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 12.0, -1.0, -2.0, -3.0]:
            test_w = dict(selected)
            test_w[(feat_name, span)] = w
            avg, ge8, ge10, ge11, _ = score_combination(test_w, 20)
            candidates.append((avg, ge8, ge10, ge11, feat_name, span, w))
    
    candidates.sort(key=lambda x: -x[0])
    best_cand = candidates[0]
    
    if best_cand[0] > best_avg + 0.005:  # 至少提升 0.005
        best_avg = best_cand[0]
        selected[(best_cand[4], best_cand[5])] = best_cand[6]
        history.append({
            'step': step+1,
            'added': (best_cand[4], best_cand[5], best_cand[6]),
            'avg': best_cand[0],
            'ge8': best_cand[1],
            'ge10': best_cand[2],
            'ge11': best_cand[3],
            'config': dict(selected),
        })
        print(f'  Step {step+1}: +{best_cand[4]}({best_cand[5]},w={best_cand[6]}) → avg={best_cand[0]:.4f} ≥8中 {best_cand[1]:.1f}% ≥10中 {best_cand[2]:.1f}%')
    else:
        print(f'  Step {step+1}: 无改进,停止')
        break

# 输出最优组合
print()
print('=' * 70)
print('最优组合')
print('=' * 70)
for (feat, span), w in selected.items():
    print(f'  {feat}(span={span}, weight={w})')

# 在不同 TopN 上评估
print()
print('=' * 70)
print('最优组合在不同 TopN 的表现')
print('=' * 70)
for top_n in [9, 12, 15, 18, 20, 25, 30]:
    avg, ge8, ge10, ge11, _ = score_combination(selected, top_n)
    print(f'  Top{top_n}: 平均 {avg:.3f}/20 ({avg*5:.2f}%)  ≥8中 {ge8:.1f}%  ≥10中 {ge10:.1f}%  ≥11中 {ge11:.1f}%')

# 与基线对比
print()
print('=' * 70)
print('对比基线: 朴素 freq W=100')
print('=' * 70)
baseline = {('freq', 100): 1.0}
for top_n in [9, 15, 20, 25, 30]:
    avg, ge8, ge10, ge11, _ = score_combination(baseline, top_n)
    print(f'  Top{top_n}: 平均 {avg:.3f}/20  ≥8中 {ge8:.1f}%  ≥10中 {ge10:.1f}%')

# 在 2026195 期验证
print()
print('=' * 70)
print('2026195 期实测 (索引 0)')
print('=' * 70)
actual = set(data[0].tolist())
print(f'2026195 开奖: {sorted(actual)}')

# 用最优组合评分
scores_2026195 = np.zeros(80)
for (feat, span), w in selected.items():
    s = feat_cache[(feat, span)][0]  # 索引 0 是 2026195
    scores_2026195 += s * w
top20_opt = sorted(np.argsort(-scores_2026195)[:20].tolist())
top20_opt = [t+1 for t in top20_opt]
hits = set(top20_opt) & actual
print(f'最优 Top20: {top20_opt}')
print(f'命中 {len(hits)}/20 = {len(hits)*5:.1f}%: {sorted(hits)}')

# 用朴素 freq W=100 Top20
scores_naive = feat_cache[('freq', 100)][0]
top20_naive = sorted(np.argsort(-scores_naive)[:20].tolist())
top20_naive = [t+1 for t in top20_naive]
hits = set(top20_naive) & actual
print(f'\n朴素 freq(100) Top20: {top20_naive}')
print(f'命中 {len(hits)}/20 = {len(hits)*5:.1f}%: {sorted(hits)}')

# 保存
out = {
    'method': 'greedy_forward_no_dup',
    'test_periods': 200,
    'feature_pool_size': len(FEATURE_POOL),
    'best_config': {f'{k[0]}_{k[1]}': v for k, v in selected.items()},
    '_history_config_str': [str(h['config']) for h in history],
    'best_top20_avg': best_avg,
    'history': history,
}
with open('/tmp/kl8_v3v9_greedy_v2.json', 'w', encoding='utf-8') as f:
    json.dump(out, f, indent=2, ensure_ascii=False)
print()
print('结果保存到 /tmp/kl8_v3v9_greedy_v2.json')