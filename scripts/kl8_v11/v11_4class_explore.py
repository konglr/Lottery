"""
KL8 V11: 4 分类 + V10 优化
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
    """按频次排名分 4 类: 1=高频, 2=中高, 3=中低, 4=低频"""
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

def classify_zone(num):
    """4 区位置分类: Z1=1-20, Z2=21-40, Z3=41-60, Z4=61-80"""
    return ((num - 1) // 20) + 1  # 1-4

TEST_INDICES = list(range(0, 200))
N_TEST = len(TEST_INDICES)

# 预计算特征
print('预计算特征...')
FEATURES = {}
for w in [10, 20, 30, 50, 75, 100, 150, 200]:
    FEATURES[('freq', w)] = np.array([feat_freq(i, w) for i in TEST_INDICES])
for span in [1, 2, 3, 5, 7, 10]:
    FEATURES[('repeat', span)] = np.array([feat_repeat(i, span) for i in TEST_INDICES])
for span in [1, 2, 3, 5, 7, 10]:
    FEATURES[('neighbor', span)] = np.array([feat_neighbor(i, span) for i in TEST_INDICES])

# 预计算 4 分类 (多个窗口 + 比例)
CLASSIFY_CACHE = {}
for w in [50, 100, 150]:
    for ratios in [
        [0.25, 0.25, 0.25, 0.25],
        [0.20, 0.30, 0.30, 0.20],
        [0.15, 0.35, 0.35, 0.15],
        [0.30, 0.20, 0.20, 0.30],
    ]:
        key = (w, tuple(ratios))
        cls = np.zeros((N_TEST, 80), dtype=int)
        for k in range(N_TEST):
            cls[k] = classify_freq(FEATURES[('freq', w)][k], ratios)
        CLASSIFY_CACHE[key] = cls

# ============================================================
# 评估函数
# ============================================================
def eval_scores(scores, top_n):
    """scores: (N_TEST, 80) array"""
    hits = []
    for k in range(N_TEST):
        top_idx = np.argsort(-scores[k])[:top_n] + 1
        actual = set(data[TEST_INDICES[k]].tolist())
        top_set = set(top_idx.tolist())
        hits.append(len(top_set & actual))
    h = np.array(hits)
    return h.mean(), (h>=8).mean()*100, (h>=10).mean()*100, h

# ============================================================
# 1. 4 分类作为"软信号"(加权叠加)
# ============================================================
print('=' * 80)
print('1. 4 分类作为软信号叠加 V10')
print('=' * 80)
print(f'{"配置":<55} {"avg":>6} {"≥8中":>6} {"≥10中":>7}')

# V10 基线
def v10_base_scores():
    s = (FEATURES[('freq', 10)] * -3.0 +
         FEATURES[('repeat', 3)] * -3.0 +
         FEATURES[('neighbor', 5)] * -3.0)
    return s

v10_s = v10_base_scores()
avg, ge8, ge10, _ = eval_scores(v10_s, 20)
print(f'{"V10 基线":<55} {avg:>6.3f} {ge8:>5.1f}% {ge10:>6.1f}%')

# V10 + 4 分类(类 4 加权)
for w_cls in [50, 100, 150]:
    for ratios in [
        [0.25, 0.25, 0.25, 0.25],
        [0.20, 0.30, 0.30, 0.20],
        [0.15, 0.35, 0.35, 0.15],
        [0.30, 0.20, 0.20, 0.30],
    ]:
        cls_key = (w_cls, tuple(ratios))
        for w_class in [1.0, 2.0, 3.0, 5.0, 8.0]:
            # 反向: 类越大 (越冷) 分数越高
            class_signal = CLASSIFY_CACHE[cls_key].astype(float) * w_class
            scores = v10_s + class_signal
            avg, ge8, ge10, _ = eval_scores(scores, 20)
            if avg >= 5.45:
                label = f'V10 + 4分类(W={w_cls}, ratios={ratios}, w={w_class})'
                print(f'{label:<55} {avg:>6.3f} {ge8:>5.1f}% {ge10:>6.1f}%')

# ============================================================
# 2. 多窗口 4 分类组合
# ============================================================
print()
print('=' * 80)
print('2. 多窗口 4 分类组合 (V10 + 多个分类窗口)')
print('=' * 80)

# V10 + 短窗口分类 + 长窗口分类
best_combos = []
for w_short in [30, 50, 75]:
    for w_long in [100, 150, 200]:
        for w_short_weight in [1.0, 2.0, 3.0]:
            for w_long_weight in [1.0, 2.0, 3.0]:
                # 计算短/长窗口的 4 分类
                cls_short = np.zeros((N_TEST, 80), dtype=int)
                cls_long = np.zeros((N_TEST, 80), dtype=int)
                for k in range(N_TEST):
                    cls_short[k] = classify_freq(FEATURES[('freq', w_short)][k], [0.25]*4)
                    cls_long[k] = classify_freq(FEATURES[('freq', w_long)][k], [0.25]*4)
                # 组合信号
                class_signal = (cls_short.astype(float) * w_short_weight +
                                cls_long.astype(float) * w_long_weight)
                scores = v10_s + class_signal
                avg, ge8, ge10, _ = eval_scores(scores, 20)
                best_combos.append((avg, ge8, ge10, w_short, w_long, w_short_weight, w_long_weight))

best_combos.sort(key=lambda x: -x[0])
print(f'\n最优 Top10 (V10 + 多窗口 4 分类):')
print(f'{"w_short":>8} {"w_long":>7} {"w_short_w":>9} {"w_long_w":>8} {"avg":>6} {"≥8中":>6} {"≥10中":>7}')
for c in best_combos[:10]:
    print(f'{c[3]:>8} {c[4]:>7} {c[5]:>9.1f} {c[6]:>8.1f} {c[0]:>6.3f} {c[1]:>5.1f}% {c[2]:>6.1f}%')

# ============================================================
# 3. V11 最终方案: V10 + 4 分类 + 4 区位
# ============================================================
print()
print('=' * 80)
print('3. V11 完整方案: V10 + 4 分类 + 4 区位信号')
print('=' * 80)

# 4 区位置分类 (固定, 不需要计算)
ZONE_CLASS = np.array([classify_zone(n) for n in range(1, 81)])  # (80,) 值 1-4

# V11 = V10 + 4 分类 + 4 区位
def v11_score(class_window=100, class_weight=2.0, zone_weight=0.0):
    """class_window: 4 分类用频次窗口
       class_weight: 4 分类权重
       zone_weight: 4 区位权重 (0 = 不用)
    """
    s = v10_s.copy()
    # 4 分类反向
    cls = np.zeros((N_TEST, 80), dtype=int)
    for k in range(N_TEST):
        cls[k] = classify_freq(FEATURES[('freq', class_window)][k], [0.25]*4)
    s += cls.astype(float) * class_weight
    # 4 区位 (固定)
    if zone_weight != 0:
        s += ZONE_CLASS * zone_weight
    return s

# 搜索最优 class_weight + zone_weight
best_v11 = []
for cw in [0.0, 1.0, 2.0, 3.0, 5.0]:
    for zw in [-2.0, -1.0, 0.0, 1.0, 2.0, 3.0]:
        scores = v11_score(class_weight=cw, zone_weight=zw)
        avg, ge8, ge10, _ = eval_scores(scores, 20)
        best_v11.append((avg, ge8, ge10, cw, zw))

best_v11.sort(key=lambda x: -x[0])
print(f'\nV11 (V10 + 4分类 + 4区位) 最优 Top10:')
print(f'{"class_w":>8} {"zone_w":>8} {"avg":>6} {"≥8中":>6} {"≥10中":>7}')
for c in best_v11[:10]:
    print(f'{c[3]:>8.1f} {c[4]:>8.1f} {c[0]:>6.3f} {c[1]:>5.1f}% {c[2]:>6.1f}%')

# 取最优, 在不同 TopN 上评估
if best_v11:
    best_cw, best_zw = best_v11[0][3], best_v11[0][4]
    print()
    print(f'=' * 80)
    print(f'V11 最优 (class_w={best_cw}, zone_w={best_zw}) 在不同 TopN')
    print(f'=' * 80)
    scores = v11_score(class_weight=best_cw, zone_weight=best_zw)
    for top_n in [9, 12, 15, 18, 20, 25, 30]:
        avg, ge8, ge10, _ = eval_scores(scores, top_n)
        print(f'  Top{top_n}: 平均 {avg:.3f}/20 ({avg*5:.2f}%)  ≥8中 {ge8:.1f}%  ≥10中 {ge10:.1f}%')

    # 跟 V10 对比
    print()
    print('V10 基线对比 (同样 TopN):')
    for top_n in [9, 12, 15, 18, 20, 25, 30]:
        avg, ge8, ge10, _ = eval_scores(v10_s, top_n)
        print(f'  Top{top_n}: 平均 {avg:.3f}/20 ({avg*5:.2f}%)  ≥8中 {ge8:.1f}%  ≥10中 {ge10:.1f}%')

# 保存
out = {
    'method': 'v11_4class_4zone',
    'test_periods': 200,
    'best_v10_plus_4class': best_combos[:10] if best_combos else [],
    'best_v11': best_v11[:10] if best_v11 else [],
}
with open('/tmp/kl8_v11_results.json', 'w', encoding='utf-8') as f:
    json.dump(out, f, indent=2, ensure_ascii=False)
print()
print('结果保存到 /tmp/kl8_v11_results.json')