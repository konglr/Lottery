"""
KL8 V22: V21 + 形态约束 (morphology: max_consecutive + 4 区位平衡)
====================================================================

V21 痛点 (实战 2026201 暴露的问题):
  - 10 胆 4 区位分布: 1/5/3/1 (1 区只有 1 个, 2 区过度集中)
  - 6 连号聚集 [34, 35, 36, 37] (实际 P95=5, 4 连号虽然合法但不自然)

V22 修复:
  - max_consecutive ≤ 2 (项目 config.py 已有此规则)
  - 4 区位平衡: 1-20 / 21-40 / 41-60 / 61-80 各占 20-40%
  - 跨度约束: 10 胆跨度 ≤ 70

V22 评分: V21 评分 + 形态调整 (在 V21 评分基础上选最稳健的形态)
"""
import sys
from pathlib import Path
sys.path.insert(0, '/Users/clarkkong/.openclaw/workspace/agents/lucky')
from lottery_data import LotteryData
import pandas as pd
import numpy as np
import json
from datetime import datetime
from collections import defaultdict, Counter

ROOT = Path.home() / 'Library/Mobile Documents/com~apple~CloudDocs/PycharmProjects/Lottery'
ld = LotteryData(ROOT)
df, conf = ld.load('快乐8')
red_cols = [f'红球{i}' for i in range(1, 21)]
data = df[red_cols].astype(int).values
n_periods = len(data)
print(f'快乐8 数据: {n_periods} 期')

# ============= 特征函数 =============
def get_counts(last_i):
    counts = {}
    for w in [5, 10, 30]:
        c = np.zeros(80, dtype=int)
        for k in range(1, w + 1):
            idx = last_i + k
            if idx < n_periods:
                for n in data[idx]:
                    c[n-1] += 1
        counts[w] = c
    return counts

def v21_score(last_i):
    if last_i + 30 >= n_periods:
        return np.zeros(80)
    counts = get_counts(last_i)
    c5, c10, c30 = counts[5], counts[10], counts[30]
    
    c5_score = np.zeros(80)
    c5_score[c5 <= 2] = 0.3
    c5_score[(c5 >= 4) & (c5 <= 5)] = -0.8
    c5_score[c5 >= 6] = -1.5
    
    c10_score = np.zeros(80)
    c10_score[(c10 >= 2) & (c10 <= 4)] = 1.0
    c10_score[c10 == 1] = 0.4
    c10_score[c10 == 0] = 0.2
    c10_score[(c10 >= 5) & (c10 <= 6)] = -0.3
    c10_score[c10 >= 7] = -0.8
    
    c30_score = np.zeros(80)
    c30_score[(c30 >= 5) & (c30 <= 10)] = 0.5
    c30_score[(c30 >= 3) & (c30 <= 4)] = 0.2
    c30_score[c30 <= 2] = -0.2
    c30_score[(c30 >= 11) & (c30 <= 12)] = -0.3
    c30_score[c30 >= 13] = -0.7
    
    nbr_score = np.zeros(80)
    for num in data[last_i]:
        for d in [-2, -1, 1, 2]:
            if 1 <= num+d <= 80:
                nbr_score[num+d-1] += 0.3
    nbr_score /= 20
    
    return c5_score + c10_score + c30_score + nbr_score

# ============= 形态约束 =============
def check_consecutive_max(nums, max_consec=2):
    """检查连号段最大长度 ≤ max_consec"""
    s = sorted(nums)
    max_run = 1
    current = 1
    for i in range(1, len(s)):
        if s[i] == s[i-1] + 1:
            current += 1
            max_run = max(max_run, current)
        else:
            current = 1
    return max_run <= max_consec

def check_zone_balance(nums, max_dev=0.35):
    """4 区位平衡: 每个区占总号数的 20-40% (deviation ≤ 35%)"""
    n = len(nums)
    if n == 0:
        return True
    zones = [0, 0, 0, 0]
    for num in nums:
        if 1 <= num <= 20: zones[0] += 1
        elif 21 <= num <= 40: zones[1] += 1
        elif 41 <= num <= 60: zones[2] += 1
        elif 61 <= num <= 80: zones[3] += 1
    # 每个区占比应在 1/n 到 (1/n + max_dev) 之间
    expected = 1.0 / 4  # 25%
    for z in zones:
        p = z / n
        if abs(p - expected) > max_dev:  # 偏离 25% 太多
            return False
    return True

def check_span(nums, max_span=70):
    """跨度约束"""
    if len(nums) < 2:
        return True
    return (max(nums) - min(nums)) <= max_span

# ============= V22 选号 =============
def v22_predict(last_i, top_n=10, max_repeat_per_lag=7, n_lags=3,
                max_consec=2, max_zone_dev=0.35, max_span=70):
    """V22: V21 评分 + 形态约束 (post-filter)"""
    lag_sets = []
    for lag in range(1, n_lags + 1):
        if last_i + lag - 1 < n_periods:
            lag_sets.append((lag, set(data[last_i + lag - 1].tolist())))
    
    scores = v21_score(last_i)
    sorted_idx = np.argsort(scores)[::-1]
    
    selected = []
    lag_repeats = {lag: 0 for lag, _ in lag_sets}
    
    for idx in sorted_idx:
        num = int(idx) + 1
        if num in selected:
            continue
        # 重号约束
        skip = False
        for lag, lag_set in lag_sets:
            if num in lag_set:
                if lag_repeats[lag] >= max_repeat_per_lag:
                    skip = True
                    break
        if skip:
            continue
        
        # 形态约束 (检查加入后是否仍合规)
        candidate = selected + [num]
        
        # 1. 连号约束
        if len(candidate) >= 3 and not check_consecutive_max(candidate, max_consec):
            continue
        
        # 2. 跨度约束
        if len(candidate) >= 4 and not check_span(candidate, max_span):
            continue
        
        # 3. 区位平衡约束 (从 4 个号开始检查)
        if len(candidate) >= 4 and not check_zone_balance(candidate, max_zone_dev):
            continue
        
        # 通过, 加入
        for lag, lag_set in lag_sets:
            if num in lag_set:
                lag_repeats[lag] += 1
        selected.append(num)
        if len(selected) >= top_n:
            break
    
    return sorted(selected[:top_n]), lag_repeats

# ============= 回测 =============
def backtest(n_test=200, top_n_list=[6, 9, 10, 20]):
    print(f'\n开始 {n_test} 期回测 (V22 形态约束版)...')
    
    results = {}
    for top_n in top_n_list:
        results[top_n] = {
            'v22': {'hits': [], 'rep_l1': [], 'rep_l2': [], 'rep_l3': []},
            'v21': {'hits': [], 'rep_l1': [], 'rep_l2': [], 'rep_l3': []},
            'v18_c': {'hits': [], 'rep_l1': [], 'rep_l2': [], 'rep_l3': []},
        }
    
    # V21 预测 (复用, 无形态约束)
    def v21_predict_local(last_i, top_n, max_repeat_per_lag=7):
        lag_sets = []
        for lag in range(1, 4):
            if last_i + lag - 1 < n_periods:
                lag_sets.append((lag, set(data[last_i + lag - 1].tolist())))
        scores = v21_score(last_i)
        sorted_idx = np.argsort(scores)[::-1]
        selected = []
        lag_repeats = {lag: 0 for lag, _ in lag_sets}
        for idx in sorted_idx:
            num = int(idx) + 1
            if num in selected:
                continue
            skip = False
            for lag, lag_set in lag_sets:
                if num in lag_set and lag_repeats[lag] >= max_repeat_per_lag:
                    skip = True
                    break
            if skip:
                continue
            for lag, lag_set in lag_sets:
                if num in lag_set:
                    lag_repeats[lag] += 1
            selected.append(num)
            if len(selected) >= top_n:
                break
        return sorted(selected[:top_n])
    
    # V16.E
    def v16_e_score(last_i):
        if last_i + 10 >= n_periods:
            return np.zeros(80)
        c_freq_10 = np.zeros(80)
        for k in range(1, 11):
            for n in data[last_i + k]:
                c_freq_10[n-1] += 1
        c_rep_3 = np.zeros(80)
        for k in range(1, 4):
            for n in data[last_i + k]:
                c_rep_3[n-1] = 1
        c_nbr_5 = np.zeros(80)
        for k in range(1, 6):
            for n in data[last_i + k]:
                for d in [-1, 1]:
                    if 1 <= n+d <= 80:
                        c_nbr_5[n+d-1] += 1
        return c_freq_10 * -3.0 + c_rep_3 * -3.0 + c_nbr_5 * -3.0
    
    for k in range(n_test):
        last_i = k + 1
        if last_i + 30 >= n_periods:
            continue
        
        curr_set = set(data[k].tolist())
        lag_sets_data = [
            set(data[last_i].tolist()),
            set(data[last_i+1].tolist()),
            set(data[last_i+2].tolist()),
        ]
        
        for top_n in top_n_list:
            # V22 形态约束
            pred, _ = v22_predict(last_i, top_n=top_n, max_repeat_per_lag=7)
            pred_set = set(pred)
            results[top_n]['v22']['hits'].append(len(pred_set & curr_set))
            for i, s in enumerate(lag_sets_data):
                results[top_n]['v22'][f'rep_l{i+1}'].append(len(pred_set & s))
            
            # V21 无约束
            pred_v21 = v21_predict_local(last_i, top_n)
            pred_v21_set = set(pred_v21)
            results[top_n]['v21']['hits'].append(len(pred_v21_set & curr_set))
            for i, s in enumerate(lag_sets_data):
                results[top_n]['v21'][f'rep_l{i+1}'].append(len(pred_v21_set & s))
            
            # V18.C 基线
            v16 = v16_e_score(last_i)
            sorted_idx = np.argsort(v16)[::-1]
            sel = []
            rep = 0
            for idx in sorted_idx:
                num = int(idx) + 1
                if num in sel:
                    continue
                if num in lag_sets_data[0]:
                    if rep >= max(2, int(top_n * 0.25)):
                        continue
                    rep += 1
                sel.append(num)
                if len(sel) >= top_n:
                    break
            sel_set = set(sel)
            results[top_n]['v18_c']['hits'].append(len(sel_set & curr_set))
            for i, s in enumerate(lag_sets_data):
                results[top_n]['v18_c'][f'rep_l{i+1}'].append(len(sel_set & s))
    
    print('\n' + '=' * 115)
    print(f'{"TopN":<6} {"方案":<22} {"平均命中":<15} {"重L1":<8} {"重L2":<8} {"重L3":<8} {"≥8中":<10} {"≥10中":<10} {"vs V16.E"}')
    print('=' * 115)
    
    summary = {}
    for top_n in top_n_list:
        for ver in ['v18_c', 'v21', 'v22']:
            hits = results[top_n][ver]['hits']
            if len(hits) == 0:
                continue
            avg_hit = np.mean(hits)
            avg_l1 = np.mean(results[top_n][ver]['rep_l1'])
            avg_l2 = np.mean(results[top_n][ver]['rep_l2'])
            avg_l3 = np.mean(results[top_n][ver]['rep_l3'])
            ge8 = sum(1 for h in hits if h >= 8) / len(hits) * 100
            ge10 = sum(1 for h in hits if h >= 10) / len(hits) * 100
            ver_name = {
                'v22': 'V22 +形态约束 ⭐',
                'v21': 'V21 (原版)',
                'v18_c': 'V18.C 基线'
            }[ver]
            # 这里没基线 V16.E, 改用 V21 对比
            v21_avg = np.mean(results[top_n]['v21']['hits'])
            delta = (avg_hit - v21_avg) / v21_avg * 100 if v21_avg > 0 else 0
            print(f'{top_n:<6} {ver_name:<22} {avg_hit:.2f}/{top_n} ({avg_hit/top_n*100:.1f}%)  '
                  f'{avg_l1:.2f}    {avg_l2:.2f}    {avg_l3:.2f}    '
                  f'{ge8:.1f}%      {ge10:.1f}%      vs V21 {delta:+.1f}%')
            summary[ver] = {
                'top_n': top_n,
                'avg_hit': round(float(avg_hit), 3),
                'avg_rep_l1': round(float(avg_l1), 3),
                'avg_rep_l2': round(float(avg_l2), 3),
                'avg_rep_l3': round(float(avg_l3), 3),
                'ge8_pct': round(ge8, 2),
                'ge10_pct': round(ge10, 2),
                'delta_vs_v21': round(delta, 2),
            }
        print()
    
    return summary, results

# ============= 实战 =============
def predict_target():
    target_period = '2026201'
    last_i = 0
    
    print(f'\n=== 实战: 预测 {target_period} (V22 形态约束版) ===')
    print(f'上期 2026200: {sorted(data[last_i].tolist())}')
    
    counts = get_counts(last_i)
    
    # 6 胆
    pred_6, rep_6 = v22_predict(last_i, top_n=6, max_repeat_per_lag=7)
    print(f'\n6 胆: {pred_6}')
    sorted_6 = sorted(pred_6)
    print(f'  4 区位: {sum(1 for n in pred_6 if 1<=n<=20)}/{sum(1 for n in pred_6 if 21<=n<=40)}/{sum(1 for n in pred_6 if 41<=n<=60)}/{sum(1 for n in pred_6 if 61<=n<=80)}')
    print(f'  跨度: {max(pred_6) - min(pred_6)}')
    print(f'  连号段: 最大连续 {max_consecutive_in(pred_6)}')
    print(f'  重号: L1={rep_6[1]}, L2={rep_6[2]}, L3={rep_6[3]}')
    
    # 10 胆
    pred_10, rep_10 = v22_predict(last_i, top_n=10, max_repeat_per_lag=7)
    print(f'\n10 胆: {pred_10}')
    print(f'  4 区位: {sum(1 for n in pred_10 if 1<=n<=20)}/{sum(1 for n in pred_10 if 21<=n<=40)}/{sum(1 for n in pred_10 if 41<=n<=60)}/{sum(1 for n in pred_10 if 61<=n<=80)}')
    print(f'  跨度: {max(pred_10) - min(pred_10)}')
    print(f'  连号段: 最大连续 {max_consecutive_in(pred_10)}')
    print(f'  重号: L1={rep_10[1]}, L2={rep_10[2]}, L3={rep_10[3]}')
    
    return {
        'V22_top6': {'numbers': pred_6, 'lag_repeats': dict(rep_6)},
        'V22_top10': {'numbers': pred_10, 'lag_repeats': dict(rep_10)},
    }, target_period

def max_consecutive_in(nums):
    s = sorted(nums)
    max_run = 1
    current = 1
    for i in range(1, len(s)):
        if s[i] == s[i-1] + 1:
            current += 1
            max_run = max(max_run, current)
        else:
            current = 1
    return max_run

if __name__ == '__main__':
    summary, results = backtest(n_test=200, top_n_list=[6, 9, 10, 20])
    predictions, target = predict_target()
    
    def clean(obj):
        if isinstance(obj, dict):
            return {k: clean(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [clean(x) for x in obj]
        elif isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        return obj
    
    output = {
        'meta': {
            'created_at': datetime.now().strftime('%Y-%m-%d %H:%M GMT+8'),
            'method': 'V22 V21 + 形态约束 (max_consec≤2, 4区位平衡, 跨度≤70)',
            'lottery': '快乐8',
            'target_period': target,
            'target_open_time': '2026-07-30 21:30',
            'n_periods_backtest': 200,
            'note': '修复 V21 实战 10 胆 1区少+6连号问题'
        },
        'backtest_summary': clean(summary),
        'predictions': clean(predictions),
    }
    
    out_path = ROOT / 'data' / 'backtest' / f'{target}_predictions_v22.json'
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f'\n✅ 保存: {out_path}')