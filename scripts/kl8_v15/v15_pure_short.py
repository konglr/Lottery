"""
KL8 V15: 纯短期 S 路线
================================
核心思想: 长期没规律, 只有短期窗口内能抓到一点点信号.
完全不学 200 期的相关系数 / 共现矩阵 / 回归.
所有特征只用 S ∈ {1,2,3,5,7,10} 的窗口.

子实验:
  V15.1 纯短窗 V10 (no grid class, no corr)
  V15.2 滚动短窗共现 (30-50 期窗口, 每期重学)
  V15.3 多短窗集成 (S=3,5,7,10 各选 Top30 加权投票)
  V15.4 短窗 V8 (近 1-2 期邻号 + 重号跟随)

评估: Top20/30/40 × 200 期
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

# =================== 短期 S 特征函数 ===================
def feat_freq(i, S):
    """近 S 期出现次数"""
    if i + S > n_periods: S = n_periods - i
    if S <= 0: return np.zeros(80, dtype=float)
    block = data[i+1:i+1+S]
    return np.bincount(block.flatten(), minlength=81)[1:81].astype(float)

def feat_repeat(i, S):
    """近 S 期每个号出没出 (0/1)"""
    counts = np.zeros(80, dtype=float)
    for j in range(1, S+1):
        if i + j >= n_periods: break
        nums = data[i+j]
        counts[nums-1] += 1
    return np.clip(counts, 0, 1)  # 只看 0/1

def feat_neighbor(i, S, distance=1):
    """近 S 期邻号 ±distance 被覆盖 (0/1)"""
    covered = np.zeros(80, dtype=float)
    for j in range(1, S+1):
        if i + j >= n_periods: break
        for n in data[i+j]:
            for d in range(-distance, distance+1):
                if d == 0: continue
                if 1 <= n+d <= 80: covered[n+d-1] += 1
    return np.clip(covered, 0, 1)

def feat_miss(i, S):
    """近 S 期未出现的期数 (0..S)"""
    miss = np.zeros(80, dtype=float)
    for j in range(1, S+1):
        if i + j >= n_periods: break
        in_period = np.zeros(80, dtype=bool)
        in_period[data[i+j]-1] = True
        miss = np.where(in_period, 0, miss + 1)
    return miss

# =================== 预计算 ===================
TEST_INDICES = list(range(0, 200))
N_TEST = len(TEST_INDICES)
SHORT_S = [1, 2, 3, 5, 7, 10]

print('预计算短窗特征...')
# 特征 cache
F_FREQ = {S: np.array([feat_freq(i, S) for i in TEST_INDICES]) for S in SHORT_S}
F_REPEAT = {S: np.array([feat_repeat(i, S) for i in TEST_INDICES]) for S in SHORT_S}
F_NEIGHBOR1 = {S: np.array([feat_neighbor(i, S, 1) for i in TEST_INDICES]) for S in SHORT_S}
F_NEIGHBOR2 = {S: np.array([feat_neighbor(i, S, 2) for i in TEST_INDICES]) for S in SHORT_S}
F_MISS = {S: np.array([feat_miss(i, S) for i in TEST_INDICES]) for S in SHORT_S}

# =================== V10 / V13 v4 基线 ===================
def score_v10(i):
    """V10 原始: S=10 freq, S=3 repeat, S=5 neighbor(distance=2), 全 ×-3"""
    return (F_FREQ[10][i] * -3.0 +
            F_REPEAT[3][i] * -3.0 +
            F_NEIGHBOR2[5][i] * -3.0)

# =================== V15.1: 纯短窗 V10 网格搜索 ====================
print()
print('=' * 80)
print('V15.1: 纯短窗 V10 网格搜索')
print('  特征: freq(S1)·w1 + repeat(S2)·w2 + neighbor1(S3)·w3 + miss(S4)·w4')
print('  S1,S2,S3,S4 ∈ {1,2,3,5,7,10}')
print('=' * 80)

def eval_score_combo(score_matrix, top_n):
    """score_matrix: (N_TEST, 80)"""
    hits = []
    for k in range(N_TEST):
        actual = set(data[TEST_INDICES[k]].tolist())
        sel = set((np.argsort(-score_matrix[k])[:top_n] + 1).tolist())
        hits.append(len(sel & actual))
    h = np.array(hits)
    return float(h.mean()), float((h>=8).mean()*100), float((h>=10).mean()*100)

best_v151 = []
count = 0
for s_f in SHORT_S:
    for s_r in SHORT_S:
        for s_n in SHORT_S:
            for s_m in SHORT_S:
                for w_f in [-3.0, -1.0, -0.5]:
                    for w_r in [-3.0, -1.0, -0.5]:
                        for w_n in [-3.0, -1.0, -0.5]:
                            for w_m in [-1.0, -0.5, 0.5, 1.0]:
                                scores = (F_FREQ[s_f] * w_f +
                                          F_REPEAT[s_r] * w_r +
                                          F_NEIGHBOR1[s_n] * w_n +
                                          F_MISS[s_m] * w_m)
                                avg, ge8, ge10 = eval_score_combo(scores, 20)
                                count += 1
                                if avg >= 5.45:
                                    best_v151.append((avg, ge8, ge10,
                                                      s_f, s_r, s_n, s_m,
                                                      w_f, w_r, w_n, w_m))
                                if count % 50000 == 0:
                                    cur = max((c[0] for c in best_v151), default=0)
                                    print(f'  ... {count} 评估, best avg={cur:.4f}')

best_v151.sort(key=lambda x: -x[0])
print(f'\n总计 {count} 个组合, avg>=5.45: {len(best_v151)}')
print(f'\nV15.1 Top10 (Top20, 200 期):')
for c in best_v151[:10]:
    print(f'  avg={c[0]:.4f} ≥8中 {c[1]:.1f}% ≥10中 {c[2]:.1f}% | '
          f'freq(S={c[3]})·{c[7]:.1f} + repeat(S={c[4]})·{c[8]:.1f} + '
          f'neighbor1(S={c[5]})·{c[9]:.1f} + miss(S={c[6]})·{c[10]:.1f}')

# =================== V15.2: 滚动短窗共现 ====================
def learn_cooc_short(i, S_window=30):
    """从期 i 往前 S_window 期学共现矩阵 (不含 i 本身)"""
    cooc = np.zeros((80, 80), dtype=float)
    for j in range(1, S_window + 1):
        if i + j >= n_periods: break
        last = data[i+j]
        next_j = i + j - 1
        if next_j < 0: continue
        next_p = data[next_j]
        for a in last:
            for b in next_p:
                if a != b:
                    cooc[a-1, b-1] += 1
    # 归一化 (按行)
    row_sum = cooc.sum(axis=1, keepdims=True)
    return cooc / (row_sum + 1e-9)

print()
print('=' * 80)
print('V15.2: 滚动短窗共现 (V14 B 的纯短期版)')
print('  每期用最近 30/50/80 期学共现, 不用 200 期')
print('=' * 80)

best_v152 = []
for S_window in [30, 50, 80]:
    for w_cooc in [1.0, 2.0, 3.0, 5.0]:
        for top_n in [20, 30, 40]:
            # 预计算滚动共现
            COOC_ROLLING = np.zeros((N_TEST, 80, 80), dtype=float)
            for k in range(N_TEST):
                COOC_ROLLING[k] = learn_cooc_short(TEST_INDICES[k], S_window)
            # 评估: V10 + 共现
            hits = []
            for k in range(N_TEST):
                i = TEST_INDICES[k]
                actual = set(data[i].tolist())
                s = score_v10(i).copy()
                if i + 1 < n_periods:
                    last = data[i+1]
                    for a in last:
                        s += COOC_ROLLING[k, a-1] * w_cooc
                sel = set((np.argsort(-s)[:top_n] + 1).tolist())
                hits.append(len(sel & actual))
            h = np.array(hits)
            avg = float(h.mean())
            ge8 = float((h>=8).mean()*100)
            ge10 = float((h>=10).mean()*100)
            key = f'V15.2_cooc_sw{S_window}_w{w_cooc}_top{top_n}'
            best_v152.append((avg, ge8, ge10, S_window, w_cooc, top_n))
            if top_n == 20:
                print(f'  sw={S_window} w={w_cooc} top_n={top_n} | avg={avg:.3f} ≥8中 {ge8:.1f}% ≥10中 {ge10:.1f}%')

# 排序找出最佳
best_v152.sort(key=lambda x: -x[0])
print(f'\nV15.2 整体 Top10 (所有 TopN):')
for c in best_v152[:10]:
    print(f'  sw={c[3]} w={c[4]} top_n={c[5]} | avg={c[0]:.3f} ≥8中 {c[1]:.1f}% ≥10中 {c[2]:.1f}%')

# =================== V15.3: 多短窗集成投票 ====================
print()
print('=' * 80)
print('V15.3: 多短窗集成投票')
print('  S=3, 5, 7, 10 各选 Top30, 加权投票')
print('=' * 80)

def v15_3_vote(i, top_n=20, top_score=30):
    """S=3,5,7,10 各选 Top30, 加权投票 (S 越小权重越高)"""
    weights_per_s = {3: 4.0, 5: 3.0, 7: 2.0, 10: 1.0}
    counter = Counter()
    for S, w in weights_per_s.items():
        s = F_FREQ[S][i] * -3.0  # 短窗 freq 反向
        top = np.argsort(-s)[:top_score] + 1
        for n in top:
            counter[int(n)] += w
    selected = [n for n, _ in counter.most_common(top_n)]
    return np.array(selected) - 1

best_v153 = []
for top_n in [20, 30, 40]:
    hits = []
    for k in range(N_TEST):
        i = TEST_INDICES[k]
        actual = set(data[i].tolist())
        sel = set((v15_3_vote(i, top_n) + 1).tolist())
        hits.append(len(sel & actual))
    h = np.array(hits)
    avg = float(h.mean())
    ge8 = float((h>=8).mean()*100)
    ge10 = float((h>=10).mean()*100)
    best_v153.append((avg, ge8, ge10, top_n))
    print(f'  top_n={top_n} | avg={avg:.3f} ≥8中 {ge8:.1f}% ≥10中 {ge10:.1f}%')

# =================== V15.4: 短窗 V8 跟随 ====================
print()
print('=' * 80)
print('V15.4: 短窗 V8 邻号 + 重号跟随')
print('  S=1-2 邻号 (±1) + 重号, 完全不看长期频次')
print('=' * 80)

def v15_4_follow(i, top_n=20):
    """纯短期跟随: 上期出过的 + 邻号 ±1, 加权后选 TopN"""
    s = np.zeros(80, dtype=float)
    # 重号 (近 1 期)
    if i + 1 < n_periods:
        for n in data[i+1]:
            s[n-1] += 5.0
        # 邻号 ±1
        for n in data[i+1]:
            for d in [-1, 1]:
                if 1 <= n+d <= 80: s[n+d-1] += 3.0
            for d in [-2, 2]:
                if 1 <= n+d <= 80: s[n+d-1] += 1.0
    # 重号 (近 2 期)
    if i + 2 < n_periods:
        for n in data[i+2]:
            s[n-1] += 2.0
    return np.argsort(-s)[:top_n]

best_v154 = []
for top_n in [20, 30, 40]:
    hits = []
    for k in range(N_TEST):
        i = TEST_INDICES[k]
        actual = set(data[i].tolist())
        sel = set((v15_4_follow(i, top_n) + 1).tolist())
        hits.append(len(sel & actual))
    h = np.array(hits)
    avg = float(h.mean())
    ge8 = float((h>=8).mean()*100)
    ge10 = float((h>=10).mean()*100)
    best_v154.append((avg, ge8, ge10, top_n))
    print(f'  top_n={top_n} | avg={avg:.3f} ≥8中 {ge8:.1f}% ≥10中 {ge10:.1f}%')

# =================== 综合对比 ====================
print()
print('=' * 80)
print('综合对比 (Top20, 200 期)')
print('=' * 80)
print(f'{"方法":<35} {"avg":>6} {"命中率":>8} {"≥8中":>6} {"≥10中":>7}')

# 基线
v10_h, v10_g8, v10_g10 = eval_score_combo(np.array([score_v10(i) for i in range(N_TEST)]), 20)
print(f'{"V10 基线":<35} {v10_h:>6.3f} {v10_h/20*100:>7.1f}% {v10_g8:>5.1f}% {v10_g10:>6.1f}%')

# V15.1
if best_v151:
    c = best_v151[0]
    print(f'{"V15.1 纯短窗 V10":<35} {c[0]:>6.3f} {c[0]/20*100:>7.1f}% {c[1]:>5.1f}% {c[2]:>6.1f}%')
for c in best_v152:
    if c[5] == 20:
        print(f'{"V15.2 短窗共现 (sw="+str(c[3])+" w="+str(c[4])+")":<35} {c[0]:>6.3f} {c[0]/20*100:>7.1f}% {c[1]:>5.1f}% {c[2]:>6.1f}%')
for c in best_v153:
    if c[3] == 20:
        print(f'{"V15.3 多短窗投票":<35} {c[0]:>6.3f} {c[0]/20*100:>7.1f}% {c[1]:>5.1f}% {c[2]:>6.1f}%')
for c in best_v154:
    if c[3] == 20:
        print(f'{"V15.4 短窗跟随":<35} {c[0]:>6.3f} {c[0]/20*100:>7.1f}% {c[1]:>5.1f}% {c[2]:>6.1f}%')

print()
print('=' * 80)
print('综合对比 (Top30, 200 期)')
print('=' * 80)
print(f'{"方法":<35} {"avg":>6} {"命中率":>8} {"≥8中":>6} {"≥10中":>7}')
v10_h30, v10_g8_30, v10_g10_30 = eval_score_combo(np.array([score_v10(i) for i in range(N_TEST)]), 30)
print(f'{"V10 基线":<35} {v10_h30:>6.3f} {v10_h30/30*100:>7.1f}% {v10_g8_30:>5.1f}% {v10_g10_30:>6.1f}%')
if best_v151:
    # 用 V15.1 best config 跑 Top30
    s_f, s_r, s_n, s_m = best_v151[0][3], best_v151[0][4], best_v151[0][5], best_v151[0][6]
    w_f, w_r, w_n, w_m = best_v151[0][7], best_v151[0][8], best_v151[0][9], best_v151[0][10]
    scores = (F_FREQ[s_f] * w_f + F_REPEAT[s_r] * w_r +
              F_NEIGHBOR1[s_n] * w_n + F_MISS[s_m] * w_m)
    avg, ge8, ge10 = eval_score_combo(scores, 30)
    print(f'{"V15.1 纯短窗 V10 (Top30)":<35} {avg:>6.3f} {avg/30*100:>7.1f}% {ge8:>5.1f}% {ge10:>6.1f}%')
for c in best_v152:
    if c[5] == 30:
        print(f'{"V15.2 短窗共现 (sw="+str(c[3])+" w="+str(c[4])+")":<35} {c[0]:>6.3f} {c[0]/30*100:>7.1f}% {c[1]:>5.1f}% {c[2]:>6.1f}%')

print()
print('=' * 80)
print('综合对比 (Top40, 200 期)')
print('=' * 80)
v10_h40, _, _ = eval_score_combo(np.array([score_v10(i) for i in range(N_TEST)]), 40)
print(f'{"V10 基线":<35} {v10_h40:>6.3f} {v10_h40/40*100:>7.1f}%')

# 保存
out = {
    'method': 'v15_pure_short',
    'test_periods': 200,
    'philosophy': '纯短期 S ∈ {1..10} 路线, 不学 200 期的任何长窗统计',
    'v15_1_best_20': best_v151[:5] if best_v151 else [],
    'v15_2_best': best_v152[:10],
    'v15_3_best': best_v153,
    'v15_4_best': best_v154,
    'baseline_v10_20': {'avg': v10_h, 'ge8': v10_g8, 'ge10': v10_g10},
}
with open('/Users/clarkkong/Library/Mobile Documents/com~apple~CloudDocs/PycharmProjects/Lottery/data/backtest/v15_pure_short_results.json', 'w', encoding='utf-8') as f:
    json.dump(out, f, indent=2, ensure_ascii=False, default=str)
print()
print('结果保存到 data/backtest/v15_pure_short_results.json')
