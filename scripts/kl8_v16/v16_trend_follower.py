"""
KL8 V16: 特征热点追随 (Trend Follower)
========================================
核心思想: 每期先看"上期有什么特征热点",再动态决定这期的 (F,S,W) 配置。

13 个特征热点 (每期动态计算, vs 历史均值比较):
  F1  重号密度    F2 邻号密度    F3 连号密度
  F4  大数主导    F5 小数主导    F6 奇偶偏斜
  F7  和值水平    F8 跨度水平    F9 跨期回弹
  F10 冷号回补    F11 4 区位偏斜  F12 极端间距
  F13 短期命中反转 (近期 Top20 命中率)

4 个子模型:
  V16.A 形态镜像  - 追趋势 (重号多就加重号权)
  V16.B 形态反转  - 押回归 (重号多就减重号权)
  V16.C 形态中性  - 基础权重不调整 (对照)
  V16.D 混合 A+B  - A 和 B 各取 Top20 共识 Top30

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

# =================== 基础特征函数 ===================
def feat_freq(i, window):
    """近 window 期出现次数"""
    if i + window > n_periods: window = n_periods - i
    if window <= 0: return np.zeros(80, dtype=float)
    block = data[i+1:i+1+window]
    return np.bincount(block.flatten(), minlength=81)[1:81].astype(float)

def feat_repeat(i, span):
    """近 span 期每个号出没出 (0/1)"""
    counts = np.zeros(80, dtype=float)
    for j in range(1, span+1):
        if i + j >= n_periods: break
        nums = data[i+j]
        counts[nums-1] += 1
    return np.clip(counts, 0, 1)

def feat_neighbor(i, span, distance=1):
    """近 span 期邻号 ±distance"""
    covered = np.zeros(80, dtype=float)
    for j in range(1, span+1):
        if i + j >= n_periods: break
        for n in data[i+j]:
            for d in range(-distance, distance+1):
                if d == 0: continue
                if 1 <= n+d <= 80: covered[n+d-1] += 1
    return covered

def feat_miss(i, window):
    """近 window 期漏号 (0..window)"""
    miss = np.zeros(80, dtype=float)
    for j in range(1, window+1):
        if i + j >= n_periods: break
        in_period = np.zeros(80, dtype=bool)
        in_period[data[i+j]-1] = True
        miss = np.where(in_period, 0, miss + 1)
    return miss

# =================== 预计算 (历史统计) ===================
TEST_INDICES = list(range(0, 200))
N_TEST = len(TEST_INDICES)

# 预计算所有特征
print('预计算特征...')
F_FREQ_10 = np.array([feat_freq(i, 10) for i in TEST_INDICES])
F_FREQ_5 = np.array([feat_freq(i, 5) for i in TEST_INDICES])
F_REPEAT_3 = np.array([feat_repeat(i, 3) for i in TEST_INDICES])
F_REPEAT_1 = np.array([feat_repeat(i, 1) for i in TEST_INDICES])
F_NEIGHBOR_5 = np.array([feat_neighbor(i, 5, 1) for i in TEST_INDICES])
F_NEIGHBOR_3 = np.array([feat_neighbor(i, 3, 1) for i in TEST_INDICES])
F_MISS_30 = np.array([feat_miss(i, 30) for i in TEST_INDICES])

# =================== 13 个特征热点 (每期计算) ===================
print('计算 13 个特征热点...')

def detect_features(i):
    """检测第 i 期(预测位置)的特征热点"""
    # data[i] 是上期 (预测时已知)
    # data[i-1] data[i-2] data[i-3] 是更早
    last = data[i]  # 上期
    f = {}

    # F1 重号密度: last 与前 3 期 (data[i+1..i+3]) 的号码交集
    if i + 3 < n_periods:
        prev_3 = set(data[i+1].tolist()) | set(data[i+2].tolist()) | set(data[i+3].tolist())
        f['F1_repeat_density'] = len(set(last.tolist()) & prev_3) / 20.0
    else:
        f['F1_repeat_density'] = 0.25

    # F2 邻号密度: 上期出号 ±1±2, 在前 3 期出现比例
    if i + 3 < n_periods:
        nbrs = set()
        for n in last:
            for d in [-2, -1, 1, 2]:
                if 1 <= n+d <= 80: nbrs.add(n+d)
        prev_3 = set(data[i+1].tolist()) | set(data[i+2].tolist()) | set(data[i+3].tolist())
        f['F2_neighbor_density'] = len(nbrs & prev_3) / 80.0
    else:
        f['F2_neighbor_density'] = 0.2

    # F3 连号密度: 上期有多少组 [n, n+1]
    f['F3_consec_count'] = sum(1 for n in last if n+1 in set(last.tolist())) / 20.0

    # F4 大数主导: 上期 >40 的比例
    f['F4_big_ratio'] = sum(1 for n in last if n > 40) / 20.0

    # F5 小数主导: 上期 ≤20 的比例
    f['F5_small_ratio'] = sum(1 for n in last if n <= 20) / 20.0

    # F6 奇偶偏斜: 上期奇数比例
    f['F6_odd_ratio'] = sum(1 for n in last if n % 2 == 1) / 20.0

    # F7 和值水平
    f['F7_sum_norm'] = (last.sum() - 800) / 80.0  # 归一到 ±1 区间

    # F8 跨度水平
    f['F8_span_norm'] = (last.max() - last.min() - 55) / 15.0

    # F9 跨期回弹: 5 期前出号中上期重号比例
    if i + 5 < n_periods:
        prev_5 = set(data[i+5].tolist())
        f['F9_rebound'] = len(set(last.tolist()) & prev_5) / 20.0
    else:
        f['F9_rebound'] = 0.25

    # F10 冷号回补: 上期出现近 30 期漏号 ≥20 期的号码
    if i + 30 < n_periods:
        cold_count = 0
        for n in last:
            miss_n = sum(1 for j in range(i+1, i+31) if n not in data[j])
            if miss_n >= 20: cold_count += 1
        f['F10_cold_recover'] = cold_count / 20.0
    else:
        f['F10_cold_recover'] = 0.05

    # F11 4 区位偏斜: Z1(1-20) Z2(21-40) Z3(41-60) Z4(61-80)
    z = [0, 0, 0, 0]
    for n in last:
        if n <= 20: z[0] += 1
        elif n <= 40: z[1] += 1
        elif n <= 60: z[2] += 1
        else: z[3] += 1
    f['F11_zone_max'] = max(z) / 20.0
    f['F11_zone_min'] = min(z) / 20.0
    f['F11_zone_idx'] = z.index(max(z))  # 偏斜区 (0-3)

    # F12 极端间距: 平均相邻差
    sorted_last = sorted(last.tolist())
    gaps = [sorted_last[j+1] - sorted_last[j] for j in range(19)]
    f['F12_avg_gap'] = np.mean(gaps) / 4.0  # 归一

    # F13 短期命中反转: 近 3 期 Top20 命中率 (需要 ground truth, 在训练中只能用历史)
    # 简化: 用 freq 集中度代替
    if i + 30 < n_periods:
        freq30 = feat_freq(i, 30)
        f['F13_freq_concentration'] = freq30.max() / freq30.mean()
    else:
        f['F13_freq_concentration'] = 1.0

    return f

# 缓存特征
print('  缓存每期特征...')
FEATURES = [detect_features(i) for i in TEST_INDICES]

# 计算历史均值 (用于阈值比较)
def get_avg(key):
    vals = [f[key] for f in FEATURES]
    return np.mean(vals), np.std(vals)

print(f'\n特征统计 (200 期均值 ± std):')
for k in ['F1_repeat_density', 'F2_neighbor_density', 'F3_consec_count',
         'F4_big_ratio', 'F5_small_ratio', 'F6_odd_ratio',
         'F7_sum_norm', 'F8_span_norm', 'F9_rebound',
         'F10_cold_recover', 'F11_zone_max', 'F11_zone_min',
         'F12_avg_gap', 'F13_freq_concentration']:
    avg, std = get_avg(k)
    print(f'  {k:<30} avg={avg:.3f}  std={std:.3f}')

# =================== 动态权重调整 ===================
def adjust_weights(i, mode='A'):
    """
    根据第 i 期上期特征, 动态调整权重
    mode='A' 镜像 (追趋势), mode='B' 反转 (押回归), mode='C' 中性
    返回 dict: {'freq': w, 'repeat': w, 'neighbor': w, 'miss': w, 'zone_mask': (z1, z2, z3, z4)}
    """
    f = FEATURES[i]
    # 基础权重 (V13 v4 起点)
    w = {'freq': 0.4, 'repeat': 0.3, 'neighbor': 0.2, 'miss': 0.1}
    zone_mask = np.ones(80, dtype=float)

    if mode == 'C':
        return w, zone_mask  # 中性, 不调整

    # 计算各特征偏离均值的方向 (-1 低, 0 正常, 1 高)
    def direction(val, avg, std, thresh=0.5):
        if val > avg + thresh * std: return 1
        if val < avg - thresh * std: return -1
        return 0

    sigs = {
        'F1': direction(f['F1_repeat_density'], *get_avg('F1_repeat_density')),
        'F2': direction(f['F2_neighbor_density'], *get_avg('F2_neighbor_density')),
        'F3': direction(f['F3_consec_count'], *get_avg('F3_consec_count')),
        'F4': direction(f['F4_big_ratio'], *get_avg('F4_big_ratio')),
        'F5': direction(f['F5_small_ratio'], *get_avg('F5_small_ratio')),
        'F10': direction(f['F10_cold_recover'], *get_avg('F10_cold_recover')),
        'F11_max': direction(f['F11_zone_max'], *get_avg('F11_zone_max')),
        'F11_min': direction(f['F11_zone_min'], *get_avg('F11_zone_min')),
        'F13': direction(f['F13_freq_concentration'], *get_avg('F13_freq_concentration')),
    }

    # A 模式: 镜像 (信号越强 → 对应权重越大)
    # B 模式: 反转 (信号越强 → 对应权重越小)
    if mode == 'A':
        mult = 1
    else:  # mode == 'B'
        mult = -1

    # F1 重号 → repeat 权重
    if sigs['F1'] != 0:
        w['repeat'] *= (1 + mult * 0.5 * sigs['F1'])
        w['freq'] *= (1 - mult * 0.3 * sigs['F1'])  # 反向调整

    # F2 邻号 → neighbor 权重
    if sigs['F2'] != 0:
        w['neighbor'] *= (1 + mult * 0.6 * sigs['F2'])

    # F3 连号 → neighbor 权重
    if sigs['F3'] != 0:
        w['neighbor'] *= (1 + mult * 0.4 * sigs['F3'])

    # F4 大数 → zone 4 mask
    if sigs['F4'] != 0:
        zone_mask[60:80] *= (1 + mult * 0.5 * sigs['F4'])

    # F5 小数 → zone 1 mask
    if sigs['F5'] != 0:
        zone_mask[0:20] *= (1 + mult * 0.5 * sigs['F5'])

    # F10 冷号回补 → miss 权重
    if sigs['F10'] != 0:
        w['miss'] *= (1 + mult * 0.8 * sigs['F10'])

    # F11 区位偏斜 → 对应区 mask
    if sigs['F11_max'] != 0 and sigs['F11_min'] != 0:
        # 主导区加权
        zone_idx = int(f['F11_zone_idx'])
        zone_mask[zone_idx*20:(zone_idx+1)*20] *= (1 + mult * 0.5 * sigs['F11_max'])

    # F13 集中度 → freq 权重
    if sigs['F13'] != 0:
        w['freq'] *= (1 - mult * 0.4 * sigs['F13'])

    # 确保权重为正
    for k in ['freq', 'repeat', 'neighbor', 'miss']:
        w[k] = max(0.05, w[k])

    return w, zone_mask

# =================== 评分函数 ===================
def score_v10_base(i):
    """V10 基线: 反向 freq(10) + repeat(3) + neighbor(5)"""
    return (F_FREQ_10[i] * -3.0 +
            F_REPEAT_3[i] * -3.0 +
            F_NEIGHBOR_5[i] * -3.0)

def score_v16(i, mode):
    """V16 动态权重: 基于 V10 基线, 加 zone_mask 和权重调整"""
    w, zmask = adjust_weights(i, mode)
    base = score_v10_base(i)  # V10 反向分数
    # 加上 miss 反向 (用 V10 思路: 漏号越多越加)
    miss_part = F_MISS_30[i] * w['miss'] * 1.0
    # 应用 zone mask
    return (base * zmask + miss_part)

# =================== 4 个子模型评估 ===================
def eval_mode(mode, top_n):
    hits = []
    for k in range(N_TEST):
        i = TEST_INDICES[k]
        actual = set(data[i].tolist())
        s = score_v16(i, mode)
        sel = set((np.argsort(-s)[:top_n] + 1).tolist())
        hits.append(len(sel & actual))
    h = np.array(hits)
    return float(h.mean()), float((h>=8).mean()*100), float((h>=10).mean()*100)

print()
print('=' * 80)
print('V16 评估 (200 期, 4 子模型 × Top20/30/40)')
print('=' * 80)

results = {}
for mode in ['A', 'B', 'C', 'D']:
    results[mode] = {}

# A 镜像 / B 反转 / C 中性
for mode in ['A', 'B', 'C']:
    for top_n in [20, 30, 40]:
        avg, ge8, ge10 = eval_mode(mode, top_n)
        results[mode][top_n] = {'avg': avg, 'ge8': ge8, 'ge10': ge10}

# D 混合 A+B
print('  V16.D (混合 A+B 共识 Top30)...')
hits_d = []
for k in range(N_TEST):
    i = TEST_INDICES[k]
    actual = set(data[i].tolist())
    s_a = score_v16(i, 'A')
    s_b = score_v16(i, 'B')
    # 各取 Top20, 共识 Top30
    top_a = set((np.argsort(-s_a)[:20] + 1).tolist())
    top_b = set((np.argsort(-s_b)[:20] + 1).tolist())
    consensus = top_a | top_b  # 并集
    # 取 Top30 (按 A 的分数排序)
    sel = sorted(consensus, key=lambda n: -s_a[n-1])[:30]
    hits_d.append(len(set(sel) & actual))
h_d = np.array(hits_d)
results['D'] = {30: {'avg': float(h_d.mean()), 'ge8': float((h_d>=8).mean()*100), 'ge10': float((h_d>=10).mean()*100)}}

# 输出对比
print()
print(f'{"方法":<20} {"TopN":>5} {"avg":>6} {"命中率":>8} {"≥8中":>6} {"≥10中":>7}')
print('-' * 70)
# V10 基线
v10_h20, v10_g8_20, v10_g10_20 = eval_mode('C', 20)  # 用 C 模式当 V10
v10_h30, v10_g8_30, v10_g10_30 = eval_mode('C', 30)
v10_h40, _, _ = eval_mode('C', 40)
print(f'{"V10/V16-C 基线":<20} {20:>5} {v10_h20:>6.3f} {v10_h20/20*100:>7.1f}% {v10_g8_20:>5.1f}% {v10_g10_20:>6.1f}%')
print(f'{"V10/V16-C 基线":<20} {30:>5} {v10_h30:>6.3f} {v10_h30/30*100:>7.1f}% {v10_g8_30:>5.1f}% {v10_g10_30:>6.1f}%')
print(f'{"V10/V16-C 基线":<20} {40:>5} {v10_h40:>6.3f} {v10_h40/40*100:>7.1f}%')
print()
for mode_name, mode_label in [('A', 'V16.A 镜像(追趋势)'),
                               ('B', 'V16.B 反转(押回归)'),
                               ('D', 'V16.D 混合A+B')]:
    for top_n in [20, 30, 40]:
        if top_n in results[mode_name]:
            r = results[mode_name][top_n]
            print(f'{mode_label:<20} {top_n:>5} {r["avg"]:>6.3f} {r["avg"]/top_n*100:>7.1f}% {r["ge8"]:>5.1f}% {r["ge10"]:>6.1f}%')

# =================== 关键诊断: 动态调整真的有效吗? ===================
print()
print('=' * 80)
print('关键诊断: 动态调整 vs 固定权重 (V16.A - V16.C)')
print('=' * 80)

# 200 期逐期比较
wins_a, wins_c = 0, 0
ties = 0
for k in range(N_TEST):
    i = TEST_INDICES[k]
    actual = set(data[i].tolist())
    s_a = score_v16(i, 'A')
    s_c = score_v16(i, 'C')
    h_a = len(set((np.argsort(-s_a)[:20] + 1).tolist()) & actual)
    h_c = len(set((np.argsort(-s_c)[:20] + 1).tolist()) & actual)
    if h_a > h_c: wins_a += 1
    elif h_a < h_c: wins_c += 1
    else: ties += 1

print(f'V16.A 镜像 vs V16.C 中性 (Top20, 200 期逐期对比):')
print(f'  A 胜: {wins_a} 期 ({wins_a/N_TEST*100:.1f}%)')
print(f'  C 胜: {wins_c} 期 ({wins_c/N_TEST*100:.1f}%)')
print(f'  平: {ties} 期 ({ties/N_TEST*100:.1f}%)')

wins_b, _ = 0, 0
wins_b, wins_c2 = 0, 0
ties2 = 0
for k in range(N_TEST):
    i = TEST_INDICES[k]
    actual = set(data[i].tolist())
    s_b = score_v16(i, 'B')
    s_c = score_v16(i, 'C')
    h_b = len(set((np.argsort(-s_b)[:20] + 1).tolist()) & actual)
    h_c = len(set((np.argsort(-s_c)[:20] + 1).tolist()) & actual)
    if h_b > h_c: wins_b += 1
    elif h_b < h_c: wins_c2 += 1
    else: ties2 += 1

print(f'\nV16.B 反转 vs V16.C 中性 (Top20, 200 期逐期对比):')
print(f'  B 胜: {wins_b} 期 ({wins_b/N_TEST*100:.1f}%)')
print(f'  C 胜: {wins_c2} 期 ({wins_c2/N_TEST*100:.1f}%)')
print(f'  平: {ties2} 期 ({ties2/N_TEST*100:.1f}%)')

# =================== 2026197 期预测 ===================
print()
print('=' * 80)
print('2026197 期 V16 预测')
print('=' * 80)

# 最新一期是 data[0] (时间倒序), 所以预测 data[0] 时用 data[1..n] 的历史
# 实际上 LotteryData 是按时间倒序: data[0]=最新已开奖, data[1]=上期...
# 我们要预测 2026197 (下一期), 所以用 data[0] = 2026196 (上期) 的特征
# 但我们的 predict 需要 i 指向 "上期" 的位置 (因为 model 用 i+1..i+N 的历史)
# 实际: predict(2026197) -> 用 data[0]=2026196 作为 last -> adjust_weights(0)
#       score 用 feat_freq(0, 10) 等 (即用 data[1..11] 作为历史)

# 重新计算: 预测下一期 (target=2026197)
# last = data[0] = 2026196
# 历史: data[1..n] = 2026195 及更早
# 但我们的 TEST_INDICES 是 [0, 200), 所以 score_v16(0) 就是预测 2026197

# 预测下一期 = 2026197
last_period_num = df.iloc[0]['期号']  # 2026196
target_period_num = int(last_period_num) + 1  # 2026197
print(f'上期: {last_period_num}, 预测目标: {target_period_num}')

predictions = {}
for mode_name, mode_label in [('A', 'V16.A 镜像'),
                               ('B', 'V16.B 反转'),
                               ('C', 'V16.C 中性'),
                               ('D', 'V16.D 混合')]:
    s = score_v16(0, mode_name)
    top20 = sorted((np.argsort(-s)[:20] + 1).tolist())
    top30 = sorted((np.argsort(-s)[:30] + 1).tolist())
    predictions[mode_name] = {'top20': top20, 'top30': top30}
    print(f'\n{mode_label}:')
    print(f'  Top20: {top20}')
    print(f'  Top30: {top30}')

# 共识 Top20 (3 个模式都选)
all_3 = (set(predictions['A']['top20']) &
         set(predictions['B']['top20']) &
         set(predictions['C']['top20']))
print(f'\n三模式共识 (Top20 ∩): {len(all_3)} 个 → {sorted(all_3)}')

# =================== 保存 ===================
out = {
    'method': 'v16_trend_follower',
    'test_periods': 200,
    'philosophy': '每期根据上期特征热点动态调整权重 (F1-F13)',
    'features': {
        'F1_repeat_density': '重号密度',
        'F2_neighbor_density': '邻号密度',
        'F3_consec_count': '连号密度',
        'F4_big_ratio': '大数主导',
        'F5_small_ratio': '小数主导',
        'F6_odd_ratio': '奇偶偏斜',
        'F7_sum_norm': '和值水平',
        'F8_span_norm': '跨度水平',
        'F9_rebound': '跨期回弹',
        'F10_cold_recover': '冷号回补',
        'F11_zone_max/min': '4 区位偏斜',
        'F12_avg_gap': '极端间距',
        'F13_freq_concentration': '短期集中度',
    },
    'results': results,
    'head_to_head_A_vs_C': {'A_wins': wins_a, 'C_wins': wins_c, 'ties': ties},
    'head_to_head_B_vs_C': {'B_wins': wins_b, 'C_wins': wins_c2, 'ties': ties2},
    'prediction_target': str(target_period_num),
    'predictions': predictions,
    'consensus_top20': sorted(all_3),
}

with open('/Users/clarkkong/Library/Mobile Documents/com~apple~CloudDocs/PycharmProjects/Lottery/data/backtest/v16_trend_follower_results.json', 'w', encoding='utf-8') as f:
    json.dump(out, f, indent=2, ensure_ascii=False, default=str)

print()
print('结果保存到 data/backtest/v16_trend_follower_results.json')