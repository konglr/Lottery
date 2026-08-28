"""
KL8 V14: 4 个提升策略实验
================================
基线: V10 反向 + V13 v4 (格子相关系数)
方法:
  A. 形态约束后过滤 (sum/odd/even/span/zone/repeat)
  B. 号码级共现矩阵 (P(num_b=next | num_a=current))
  C. 三模型集成投票 (V7 + V13 v4 + 朴素 W=100)
  D. 跨期回归权重 (用最近 200 期, 6 特征 → 下期是否出现的回归)
每个方法都在 Top20/30/40 上评估
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

# =================== 基础特征函数 ===================
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

def classify_freq(freq):
    rank = np.argsort(-freq)
    classes = np.zeros(80, dtype=int)
    classes[rank[:20]] = 1
    classes[rank[20:40]] = 2
    classes[rank[40:60]] = 3
    classes[rank[60:80]] = 4
    return classes

# =================== 预计算 ===================
TEST_INDICES = list(range(0, 200))
N_TEST = len(TEST_INDICES)

print('预计算特征...')
F_FREQ_10 = np.array([feat_freq(i, 10) for i in TEST_INDICES])
F_FREQ_100 = np.array([feat_freq(i, 100) for i in TEST_INDICES])
F_FREQ_50 = np.array([feat_freq(i, 50) for i in TEST_INDICES])
F_FREQ_200 = np.array([feat_freq(i, 200) for i in TEST_INDICES])
F_REPEAT_3 = np.array([feat_repeat(i, 3) for i in TEST_INDICES])
F_NEIGHBOR_5 = np.array([feat_neighbor(i, 5) for i in TEST_INDICES])
F_NEIGHBOR_2_2 = np.array([feat_neighbor(i, 2, distance=2) for i in TEST_INDICES])
F_CLASS = np.array([classify_freq(f) for f in F_FREQ_100])
ZONE = np.array([((n-1)//20) + 1 for n in range(1, 81)], dtype=int)

# V7 正向 7 特征 (简化版, 用上 200 期数据)
def score_v7(i):
    # 频次(200 期) + 跨 3/5/7 + 大小反转 + 反热号
    return (F_FREQ_200[i] * 0.4 +
            F_REPEAT_3[i] * 0.35 +
            np.minimum(15, F_FREQ_50[i] * 2) * 0.11 +
            F_NEIGHBOR_5[i] * 0.20 +
            # 简化的额外项
            np.array([np.sum(F_FREQ_100[i, max(0,n-2):min(80,n+2)]) for n in range(80)]) * 0.1)

# V10 反向 3 特征
def score_v10(i):
    return (F_FREQ_10[i] * -3.0 +
            F_REPEAT_3[i] * -3.0 +
            F_NEIGHBOR_5[i] * -3.0)

# 朴素 W=100 (频次排名)
def score_naive(i):
    return F_FREQ_100[i].copy()

# V13 v4: 学 16 格子相关系数
print('学 V13 v4 相关系数...')
def learn_correlations_v13(S):
    corrs = np.zeros((4, 4))
    for c in [1, 2, 3, 4]:
        for z in [1, 2, 3, 4]:
            pairs = []
            for k in range(N_TEST):
                i_idx = TEST_INDICES[k]
                if i_idx + S >= n_periods: continue
                cls = F_CLASS[k]
                short = 0
                for j in range(1, S+1):
                    if i_idx+j >= n_periods: break
                    nums = data[i_idx+j]
                    for n in nums:
                        if cls[n-1] == c and ZONE[n-1] == z:
                            short += 1
                next_i = i_idx - 1
                if next_i < 0: continue
                actual = data[next_i]
                next_c = sum(1 for n in actual if cls[n-1] == c and ZONE[n-1] == z)
                pairs.append((short, next_c))
            if len(pairs) >= 10:
                short_arr = np.array([p[0] for p in pairs])
                next_arr = np.array([p[1] for p in pairs])
                if short_arr.std() > 0:
                    corrs[c-1, z-1] = np.corrcoef(short_arr, next_arr)[0, 1]
    return corrs

CORRS_S5 = learn_correlations_v13(5)

def score_v13_v4(i, S=5, w_corr=5.0):
    s = score_v10(i).copy()
    if i + S < n_periods:
        grid_count = np.zeros((4, 4), dtype=int)
        for j in range(1, S+1):
            if i+j >= n_periods: break
            nums = data[i+j]
            for n in nums:
                c = F_CLASS[i, n-1]
                z = ZONE[n-1]
                grid_count[c-1, z-1] += 1
        for c in [1, 2, 3, 4]:
            for z in [1, 2, 3, 4]:
                if abs(CORRS_S5[c-1, z-1]) > 0.1:
                    mask = (F_CLASS[i] == c) & (ZONE == z)
                    avg_grid = S * 20 / 16
                    delta = (grid_count[c-1, z-1] - avg_grid) / avg_grid
                    s[mask] += w_corr * CORRS_S5[c-1, z-1] * delta
    return s

# =================== 评估函数 ===================
def evaluate(select_fn, top_n):
    """select_fn(i, top_n) -> set of selected numbers (1..80)
    select_fn 也可以只接受 i (默认按 select_fn 内部的 top_n 处理)
    """
    hits = []
    for k in range(N_TEST):
        i = TEST_INDICES[k]
        actual = set(data[i].tolist())
        # 优先用 (i, top_n) 调用, 失败则只用 (i)
        try:
            sel = select_fn(i, top_n)
        except TypeError:
            sel = select_fn(i)
        if isinstance(sel, np.ndarray):
            sel = set((sel + 1).tolist())
        else:
            sel = set(sel)
        hits.append(len(sel & actual))
    h = np.array(hits)
    return {
        'top_n': top_n,
        'avg': float(h.mean()),
        'hit_rate': float(h.mean() / top_n),
        'ge6': float((h >= 6).mean() * 100),
        'ge7': float((h >= 7).mean() * 100),
        'ge8': float((h >= 8).mean() * 100),
        'ge9': float((h >= 9).mean() * 100),
        'ge10': float((h >= 10).mean() * 100),
        'ge11': float((h >= 11).mean() * 100),
    }

# =================== 基线方法 ===================
def baseline_v10(i, top_n=20):
    return np.argsort(-score_v10(i))[:top_n]

def baseline_v13_v4(i, top_n=20):
    return np.argsort(-score_v13_v4(i))[:top_n]

def baseline_v7(i, top_n=20):
    return np.argsort(-score_v7(i))[:top_n]

def baseline_naive(i, top_n=20):
    return np.argsort(-F_FREQ_100[i])[:top_n]

# =================== 方法 A: 形态约束后过滤 ===================
def morphology_score(selected_set, last_period_nums):
    """计算组合的形态得分 (越接近黄金区间越高)"""
    nums = sorted(selected_set)
    if not nums: return -1e9
    s = sum(nums)
    span = max(nums) - min(nums)
    odd = sum(1 for n in nums if n % 2 == 1)
    even = 20 - odd
    z1 = sum(1 for n in nums if n <= 20)
    z4 = sum(1 for n in nums if n >= 61)
    consec = sum(1 for i in range(len(nums)-1) if nums[i+1] - nums[i] == 1)
    repeat = len(set(nums) & set(last_period_nums))
    score = 0
    # 黄金区间
    if 680 <= s <= 940: score += 5
    if 55 <= span <= 78: score += 3
    if 8 <= odd <= 12: score += 3
    if 4 <= z1 <= 7: score += 1
    if 3 <= z4 <= 7: score += 1
    if 2 <= consec <= 5: score += 4
    if 4 <= repeat <= 8: score += 3
    return score

def method_a_morphology(i, top_n=20, top_n_score=50):
    """从 V13 v4 Top50 中贪心选 20 满足形态约束"""
    s = score_v13_v4(i)
    # 候选 50 个
    candidates = np.argsort(-s)[:top_n_score]
    last_nums = set(data[i+1].tolist()) if i+1 < n_periods else set()
    cand_set = set((candidates + 1).tolist())

    # 贪心: 先固定 V13 v4 排名最高 5 个, 再迭代替换
    selected = list(candidates[:5] + 1)  # top 5
    remaining_pool = [c for c in (candidates + 1) if c not in selected]
    while len(selected) < top_n and remaining_pool:
        best_n = None
        best_score = -1e18
        for n in remaining_pool:
            trial = selected + [n]
            sc = morphology_score(set(trial), last_nums)
            # 加上 V13 v4 分数加成
            sc += s[n-1] * 0.01
            if sc > best_score:
                best_score = sc
                best_n = n
        if best_n is None: break
        selected.append(best_n)
        remaining_pool.remove(best_n)
    return np.array(selected[:top_n]) - 1  # 转回 0-indexed

# =================== 方法 B: 号码级共现矩阵 ===================
print('学号码级共现矩阵 (200 期)...')
COOC_MATRIX = np.zeros((80, 80), dtype=float)
COOC_COUNT = np.zeros((80,), dtype=float)
for i_idx in TEST_INDICES:
    if i_idx + 1 >= n_periods: continue
    last = data[i_idx]
    next_p = data[i_idx - 1] if i_idx - 1 >= 0 else data[i_idx + 1]
    for a in last:
        for b in next_p:
            if a != b:
                COOC_MATRIX[a-1, b-1] += 1
    COOC_COUNT += 0
# 标准化
COOC_MATRIX_NORM = COOC_MATRIX / (COOC_MATRIX.sum(axis=1, keepdims=True) + 1e-9)

def method_b_cooccurrence(i, top_n=20, w_cooc=2.0):
    """V13 v4 + 号码级共现矩阵加权"""
    s = score_v13_v4(i).copy()
    if i + 1 < n_periods:
        last = data[i+1]
        for a in last:
            # num_b 历史上下期出现率
            s += COOC_MATRIX_NORM[a-1] * w_cooc
    return np.argsort(-s)[:top_n]

# =================== 方法 C: 三模型集成投票 ===================
def method_c_vote(i, top_n=20, top_n_score=30):
    """V7 + V13 v4 + 朴素 各取 Top30, 投票"""
    s_v7 = score_v7(i)
    s_v13 = score_v13_v4(i)
    s_naive = F_FREQ_100[i]
    top_v7 = set((np.argsort(-s_v7)[:top_n_score] + 1).tolist())
    top_v13 = set((np.argsort(-s_v13)[:top_n_score] + 1).tolist())
    top_naive = set((np.argsort(-s_naive)[:top_n_score] + 1).tolist())
    # 投票计数
    vote_count = Counter()
    for n in top_v7: vote_count[n] += 1
    for n in top_v13: vote_count[n] += 1
    for n in top_naive: vote_count[n] += 1
    # 3 票 > 2 票 > 1 票
    sorted_by_vote = sorted(vote_count.items(), key=lambda x: -x[1])
    selected = [n for n, v in sorted_by_vote if v == 3]
    if len(selected) < top_n:
        for n, v in sorted_by_vote:
            if v == 2 and n not in selected:
                selected.append(n)
                if len(selected) >= top_n: break
    if len(selected) < top_n:
        for n, v in sorted_by_vote:
            if n not in selected:
                selected.append(n)
                if len(selected) >= top_n: break
    return np.array(selected[:top_n]) - 1

# =================== 方法 D: 跨期回归权重 ===================
print('学跨期回归权重 (6 特征 → 下期是否出现)...')
# 特征: freq(5,10,20,50), repeat(3), miss
# 目标: 该号在下期是否出现
REGRESSION_FEATS = np.zeros((N_TEST, 80, 6), dtype=float)
REGRESSION_TARGET = np.zeros((N_TEST, 80), dtype=float)
for k in range(N_TEST):
    i_idx = TEST_INDICES[k]
    REGRESSION_FEATS[k, :, 0] = feat_freq(i_idx, 5) if i_idx + 5 < n_periods else 0
    REGRESSION_FEATS[k, :, 1] = feat_freq(i_idx, 10)
    REGRESSION_FEATS[k, :, 2] = feat_freq(i_idx, 20)
    REGRESSION_FEATS[k, :, 3] = feat_freq(i_idx, 50)
    REGRESSION_FEATS[k, :, 4] = feat_repeat(i_idx, 3)
    # miss 简化: 100 - freq(50)
    REGRESSION_FEATS[k, :, 5] = 100 - feat_freq(i_idx, 50)
    # 目标: 该号在 i-1 (即下期) 是否出现
    if i_idx - 1 >= 0:
        next_p = data[i_idx - 1]
        for n in next_p:
            REGRESSION_TARGET[k, n-1] = 1

# 收集所有 (feat, target) 样本
all_feats = REGRESSION_FEATS.reshape(-1, 6)
all_target = REGRESSION_TARGET.flatten()
# 简单 OLS
X = np.column_stack([np.ones(len(all_feats)), all_feats])
# 用 pinv 求系数
try:
    BETA = np.linalg.lstsq(X, all_target, rcond=None)[0]
    print(f'OLS 回归系数: {BETA}')
except Exception as e:
    print(f'OLS 失败: {e}')
    BETA = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

def method_d_regression(i, top_n=20):
    """V13 v4 + 回归加权"""
    s = score_v13_v4(i).copy()
    # 加回归分数
    feat = np.array([
        feat_freq(i, 5) if i + 5 < n_periods else np.zeros(80),
        feat_freq(i, 10),
        feat_freq(i, 20),
        feat_freq(i, 50),
        feat_repeat(i, 3),
        100 - feat_freq(i, 50),
    ]).T  # (80, 6)
    feat_with_const = np.column_stack([np.ones(80), feat])
    reg_score = feat_with_const @ BETA
    # reg_score 已经反映了 "下期出现概率",但 V13 v4 是反向的, 所以要反向加
    s -= reg_score * 30  # 负相关 (V13 v4 选冷号, reg_score 高 → 该号越可能出 → 不该选)
    return np.argsort(-s)[:top_n]

# =================== 主实验 ===================
results = {}

# 基线
print()
print('=' * 80)
print('基线 (V10, V13 v4, V7, 朴素)')
print('=' * 80)
for name, fn in [('V10 反向', baseline_v10),
                 ('V13 v4', baseline_v13_v4),
                 ('V7 正向', baseline_v7),
                 ('朴素 W=100', baseline_naive)]:
    for top_n in [20, 30, 40]:
        key = f'{name}_top{top_n}'
        results[key] = evaluate(fn, top_n)
        print(f'  {name:<15} Top{top_n:<3} avg={results[key]["avg"]:.3f} '
              f'命中率={results[key]["hit_rate"]*100:.1f}% ≥8中={results[key]["ge8"]:.1f}% ≥10中={results[key]["ge10"]:.1f}%')

# 方法 A
print()
print('=' * 80)
print('方法 A: 形态约束后过滤')
print('=' * 80)
for top_n in [20, 30, 40]:
    key = f'方法A_形态_top{top_n}'
    results[key] = evaluate(method_a_morphology, top_n)
    print(f'  {key:<30} avg={results[key]["avg"]:.3f} '
          f'命中率={results[key]["hit_rate"]*100:.1f}% ≥8中={results[key]["ge8"]:.1f}% ≥10中={results[key]["ge10"]:.1f}%')

# 方法 B
print()
print('=' * 80)
print('方法 B: 号码级共现矩阵')
print('=' * 80)
for top_n in [20, 30, 40]:
    for w in [1.0, 2.0, 3.0]:
        key = f'方法B_共现_w{w}_top{top_n}'
        results[key] = evaluate(lambda i, tn=top_n, ww=w: method_b_cooccurrence(i, tn, ww), top_n)
        print(f'  {key:<35} avg={results[key]["avg"]:.3f} '
              f'命中率={results[key]["hit_rate"]*100:.1f}% ≥8中={results[key]["ge8"]:.1f}% ≥10中={results[key]["ge10"]:.1f}%')

# 方法 C
print()
print('=' * 80)
print('方法 C: 三模型集成投票')
print('=' * 80)
for top_n in [20, 30, 40]:
    key = f'方法C_投票_top{top_n}'
    results[key] = evaluate(method_c_vote, top_n)
    print(f'  {key:<30} avg={results[key]["avg"]:.3f} '
          f'命中率={results[key]["hit_rate"]*100:.1f}% ≥8中={results[key]["ge8"]:.1f}% ≥10中={results[key]["ge10"]:.1f}%')

# 方法 D
print()
print('=' * 80)
print('方法 D: 跨期回归权重')
print('=' * 80)
for top_n in [20, 30, 40]:
    key = f'方法D_回归_top{top_n}'
    results[key] = evaluate(method_d_regression, top_n)
    print(f'  {key:<30} avg={results[key]["avg"]:.3f} '
          f'命中率={results[key]["hit_rate"]*100:.1f}% ≥8中={results[key]["ge8"]:.1f}% ≥10中={results[key]["ge10"]:.1f}%')

# =================== 综合对比表 ===================
print()
print('=' * 80)
print('综合对比 (Top20, 200 期)')
print('=' * 80)
print(f'{"方法":<35} {"avg":>6} {"命中率":>8} {"≥8中":>6} {"≥10中":>7} {"改进":>8}')
v13_v4_avg = results['V13 v4_top20']['avg']
v13_v4_hr = results['V13 v4_top20']['hit_rate']
for key in ['V10 反向_top20', 'V13 v4_top20', 'V7 正向_top20', '朴素 W=100_top20',
            '方法A_形态_top20',
            '方法B_共现_w1.0_top20', '方法B_共现_w2.0_top20', '方法B_共现_w3.0_top20',
            '方法C_投票_top20',
            '方法D_回归_top20']:
    r = results[key]
    delta_avg = r['avg'] - v13_v4_avg
    print(f'{key:<35} {r["avg"]:>6.3f} {r["hit_rate"]*100:>7.1f}% '
          f'{r["ge8"]:>5.1f}% {r["ge10"]:>6.1f}% {delta_avg:>+7.3f}')

print()
print('=' * 80)
print('综合对比 (Top30, 200 期)')
print('=' * 80)
print(f'{"方法":<35} {"avg":>6} {"命中率":>8} {"≥8中":>6} {"≥10中":>7}')
for key in ['V10 反向_top30', 'V13 v4_top30', 'V7 正向_top30', '朴素 W=100_top30',
            '方法A_形态_top30',
            '方法B_共现_w1.0_top30', '方法B_共现_w2.0_top30', '方法B_共现_w3.0_top30',
            '方法C_投票_top30',
            '方法D_回归_top30']:
    r = results[key]
    print(f'{key:<35} {r["avg"]:>6.3f} {r["hit_rate"]*100:>7.1f}% '
          f'{r["ge8"]:>5.1f}% {r["ge10"]:>6.1f}%')

print()
print('=' * 80)
print('综合对比 (Top40, 200 期)')
print('=' * 80)
print(f'{"方法":<35} {"avg":>6} {"命中率":>8} {"≥8中":>6} {"≥10中":>7}')
for key in ['V10 反向_top40', 'V13 v4_top40', 'V7 正向_top40', '朴素 W=100_top40',
            '方法A_形态_top40',
            '方法B_共现_w1.0_top40', '方法B_共现_w2.0_top40', '方法B_共现_w3.0_top40',
            '方法C_投票_top40',
            '方法D_回归_top40']:
    r = results[key]
    print(f'{key:<35} {r["avg"]:>6.3f} {r["hit_rate"]*100:>7.1f}% '
          f'{r["ge8"]:>5.1f}% {r["ge10"]:>6.1f}%')

# 保存结果
out = {
    'method': 'v14_4_methods',
    'test_periods': 200,
    'methods_summary': {
        'A_形态约束后过滤': '从 V13 v4 Top50 贪心选 20 满足 sum 680-940, span 55-78, odd 8-12, consec 2-5, repeat 4-8',
        'B_号码级共现矩阵': 'V13 v4 + 学习 P(num_b=next|num_a=current) 80x80 矩阵加权 (w=1,2,3)',
        'C_三模型集成投票': 'V7 + V13 v4 + 朴素 W=100 各取 Top30, 投票决定最终 20',
        'D_跨期回归权重': '用最近 200 期 6 特征 (freq(5,10,20,50), repeat(3), miss) → 下期是否出现 OLS 回归',
    },
    'results': results,
}
with open('/Users/clarkkong/Library/Mobile Documents/com~apple~CloudDocs/PycharmProjects/Lottery/data/backtest/v14_4methods_results.json', 'w', encoding='utf-8') as f:
    json.dump(out, f, indent=2, ensure_ascii=False)
print()
print('结果保存到 data/backtest/v14_4methods_results.json')
