"""
KL8 V11 最终版: V10 自适应 (根据上期重号数切换策略)
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

def score_v10(i):
    return (feat_freq(i, 10) * -3.0 +
            feat_repeat(i, 3) * -3.0 +
            feat_neighbor(i, 5) * -3.0)

def score_v3_simple(i):
    return feat_freq(i, 50)  # 朴素频次

TEST_INDICES = list(range(0, 200))
N_TEST = len(TEST_INDICES)

# 计算每期的"上期重号数"和"上期连号对数"
period_meta = []
for k in range(N_TEST):
    i = TEST_INDICES[k]
    if i + 1 >= n_periods:
        period_meta.append({'n_repeat': 0, 'n_consec': 0, 'n_zone1': 0})
        continue
    prev = set(data[i+1].tolist())
    curr = set(data[i].tolist())
    n_repeat = len(prev & curr)
    # 连号对
    sorted_curr = sorted(curr)
    n_consec = 0
    for j in range(len(sorted_curr)-1):
        if sorted_curr[j+1] - sorted_curr[j] == 1:
            n_consec += 1
    # 1 区号 (1-20)
    n_zone1 = sum(1 for n in curr if n <= 20)
    period_meta.append({
        'n_repeat': n_repeat,
        'n_consec': n_consec,
        'n_zone1': n_zone1,
    })

# ============================================================
# V11 自适应: 根据上期形态选 V10 或 V3
# ============================================================
print('=' * 70)
print('V11 自适应: 根据上期重号数切换 V10 / V3')
print('=' * 70)

# 探索不同的重号阈值
best_adaptive = []
for th_repeat in [4, 5, 6, 7, 8]:  # 重号 ≥ 阈值时用 V3,否则用 V10
    hits = []
    for k in range(N_TEST):
        meta = period_meta[k]
        i = TEST_INDICES[k]
        actual = set(data[i].tolist())
        # 选择策略
        if meta['n_repeat'] >= th_repeat:
            scores = score_v3_simple(i)  # 用 V3 (追热号)
        else:
            scores = score_v10(i)  # 用 V10 (反热号)
        top = set((np.argsort(-scores)[:20] + 1).tolist())
        hits.append(len(top & actual))
    h = np.array(hits)
    avg = h.mean()
    ge8 = (h >= 8).mean() * 100
    ge10 = (h >= 10).mean() * 100
    best_adaptive.append((avg, ge8, ge10, th_repeat))
    print(f'  重号 ≥ {th_repeat} 用 V3, 否则 V10: avg={avg:.3f} ({avg/20*100:.1f}%) ≥8中 {ge8:.1f}% ≥10中 {ge10:.1f}%')

# 同样, 探索连号数阈值
print()
print('=' * 70)
print('V11 自适应: 根据上期连号数切换')
print('=' * 70)
for th_consec in [3, 4, 5, 6, 7]:
    hits = []
    for k in range(N_TEST):
        meta = period_meta[k]
        i = TEST_INDICES[k]
        actual = set(data[i].tolist())
        if meta['n_consec'] >= th_consec:
            scores = score_v3_simple(i)
        else:
            scores = score_v10(i)
        top = set((np.argsort(-scores)[:20] + 1).tolist())
        hits.append(len(top & actual))
    h = np.array(hits)
    avg = h.mean()
    ge8 = (h >= 8).mean() * 100
    ge10 = (h >= 10).mean() * 100
    print(f'  连号 ≥ {th_consec} 用 V3, 否则 V10: avg={avg:.3f} ({avg/20*100:.1f}%) ≥8中 {ge8:.1f}% ≥10中 {ge10:.1f}%')

# ============================================================
# 综合自适应: 重号 + 连号 联合判断
# ============================================================
print()
print('=' * 70)
print('V11 终极自适应: 多条件联合判断')
print('=' * 70)

best_combo = []
for th_r in [5, 6, 7, 8]:
    for th_c in [4, 5, 6, 7]:
        hits = []
        for k in range(N_TEST):
            meta = period_meta[k]
            i = TEST_INDICES[k]
            actual = set(data[i].tolist())
            # 任一条件满足就用 V3
            if meta['n_repeat'] >= th_r or meta['n_consec'] >= th_c:
                scores = score_v3_simple(i)
            else:
                scores = score_v10(i)
            top = set((np.argsort(-scores)[:20] + 1).tolist())
            hits.append(len(top & actual))
        h = np.array(hits)
        best_combo.append((h.mean(), (h>=8).mean()*100, (h>=10).mean()*100, th_r, th_c))

best_combo.sort(key=lambda x: -x[0])
print(f'\n最优 Top10:')
print(f'{"th_repeat":>10} {"th_consec":>10} {"avg":>6} {"≥8中":>6} {"≥10中":>7}')
for c in best_combo[:10]:
    print(f'{c[3]:>10} {c[4]:>10} {c[0]:>6.3f} {c[1]:>5.1f}% {c[2]:>6.1f}%')

# ============================================================
# 跟 V10 / V3 对比
# ============================================================
print()
print('=' * 70)
print('V10 / V3 / V11 自适应 Top20 对比')
print('=' * 70)

v10_h = []
v3_h = []
v11_h = []  # 用最优自适应
best_th_r, best_th_c = best_combo[0][3], best_combo[0][4]

for k in range(N_TEST):
    i = TEST_INDICES[k]
    actual = set(data[i].tolist())
    
    # V10
    s_v10 = score_v10(i)
    top = set((np.argsort(-s_v10)[:20] + 1).tolist())
    v10_h.append(len(top & actual))
    
    # V3
    s_v3 = score_v3_simple(i)
    top = set((np.argsort(-s_v3)[:20] + 1).tolist())
    v3_h.append(len(top & actual))
    
    # V11 自适应
    meta = period_meta[k]
    if meta['n_repeat'] >= best_th_r or meta['n_consec'] >= best_th_c:
        scores = score_v3_simple(i)
    else:
        scores = score_v10(i)
    top = set((np.argsort(-scores)[:20] + 1).tolist())
    v11_h.append(len(top & actual))

v10_h = np.array(v10_h)
v3_h = np.array(v3_h)
v11_h = np.array(v11_h)

print(f'{"策略":<25} {"avg":>6} {"≥8中":>6} {"≥10中":>7}')
print(f'{"V3 (朴素频次)":<25} {v3_h.mean():>6.3f} {(v3_h>=8).mean()*100:>5.1f}% {(v3_h>=10).mean()*100:>6.1f}%')
print(f'{"V10 (反向特征)":<25} {v10_h.mean():>6.3f} {(v10_h>=8).mean()*100:>5.1f}% {(v10_h>=10).mean()*100:>6.1f}%')
print(f'{"V11 (自适应)":<25} {v11_h.mean():>6.3f} {(v11_h>=8).mean()*100:>5.1f}% {(v11_h>=10).mean()*100:>6.1f}%')

# 胜率对比
v11_wins = sum(1 for v in zip(v11_h, v10_h) if v[0] > v[1])
v11_loses = sum(1 for v in zip(v11_h, v10_h) if v[0] < v[1])
v11_ties = N_TEST - v11_wins - v11_loses
print(f'\nV11 vs V10 胜率: V11 胜 {v11_wins}, 平局 {v11_ties}, V10 胜 {v11_loses}')

# 保存
out = {
    'method': 'v11_adaptive',
    'test_periods': 200,
    'best_thresholds': {'th_repeat': int(best_th_r), 'th_consec': int(best_th_c)},
    'comparison': {
        'v3': {'avg': float(v3_h.mean()), 'ge8': float((v3_h>=8).mean()*100), 'ge10': float((v3_h>=10).mean()*100)},
        'v10': {'avg': float(v10_h.mean()), 'ge8': float((v10_h>=8).mean()*100), 'ge10': float((v10_h>=10).mean()*100)},
        'v11_adaptive': {'avg': float(v11_h.mean()), 'ge8': float((v11_h>=8).mean()*100), 'ge10': float((v11_h>=10).mean()*100)},
    },
    'v11_vs_v10': {'wins': v11_wins, 'ties': v11_ties, 'loses': v11_loses},
}
with open('/tmp/kl8_v11_adaptive.json', 'w', encoding='utf-8') as f:
    json.dump(out, f, indent=2, ensure_ascii=False)
print()
print('结果保存到 /tmp/kl8_v11_adaptive.json')