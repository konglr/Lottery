"""
KL8 V20: 多窗口频次加权 (Multi-Window Frequency Scoring) - v2 修复版
=====================================================================

修正 v1 的索引错误:
  data[i] = df.iloc[i] (按日期降序, i 越大越早)
  last_i = "上期索引"
  - 上期前 1 期 = data[last_i + 1]  ← 不是 data[last_i - 1]!
  - 上期前 k 期 = data[last_i + k]

V20 评分 (基于 200 期实证):
  + c5 ∈ [0,2]  → +0.3 (允许短期冷/温)
  + c10 ∈ [2,4] → +1.0 ⭐ (中等活跃, 最强加分)
  + c30 ∈ [5,10] → +0.5 (中期活跃)
  - c5 ≥ 4      → -0.8
  - c10 ≥ 5     → -0.3
  - c30 ≥ 13    → -0.7
  + 邻号 (上期 ±1/±2 命中数) → +0.3
"""
import sys
from pathlib import Path
sys.path.insert(0, '/Users/clarkkong/.openclaw/workspace/agents/lucky')
from lottery_data import LotteryData
import pandas as pd
import numpy as np
import json
from datetime import datetime
from collections import defaultdict

ROOT = Path.home() / 'Library/Mobile Documents/com~apple~CloudDocs/PycharmProjects/Lottery'
ld = LotteryData(ROOT)
df, conf = ld.load('快乐8')
red_cols = [f'红球{i}' for i in range(1, 21)]
data = df[red_cols].astype(int).values
n_periods = len(data)
print(f'快乐8 数据: {n_periods} 期')

# ============= 多窗口频次特征 =============
def get_counts(last_i, max_window=30):
    """返回近 1/2/3/5/10/20/30 期的频次向量
    last_i = 上期索引, 上期前 k 期 = data[last_i + k]
    """
    counts = {}
    for w in [1, 2, 3, 5, 10, 20, 30]:
        c = np.zeros(80, dtype=int)
        for k in range(1, w + 1):
            idx = last_i + k
            if idx < n_periods:
                for n in data[idx]:
                    c[n-1] += 1
        counts[w] = c
    return counts

# ============= V20 评分函数 =============
def v20_score(last_i):
    if last_i + 30 >= n_periods:
        return np.zeros(80)
    
    counts = get_counts(last_i)
    c5 = counts[5]
    c10 = counts[10]
    c30 = counts[30]
    
    scores = np.zeros(80)
    
    # c5 评分: 0-2 允许(冷/温), 4+ 反向
    c5_score = np.zeros(80)
    c5_score[c5 <= 2] = 0.3
    c5_score[(c5 >= 4) & (c5 <= 5)] = -0.8
    c5_score[c5 >= 6] = -1.5
    
    # c10 评分: 2-4 强烈加分
    c10_score = np.zeros(80)
    c10_score[(c10 >= 2) & (c10 <= 4)] = 1.0
    c10_score[c10 == 1] = 0.4
    c10_score[c10 == 0] = 0.2
    c10_score[(c10 >= 5) & (c10 <= 6)] = -0.3
    c10_score[c10 >= 7] = -0.8
    
    # c30 评分: 5-10 强烈加分
    c30_score = np.zeros(80)
    c30_score[(c30 >= 5) & (c30 <= 10)] = 0.5
    c30_score[(c30 >= 3) & (c30 <= 4)] = 0.2
    c30_score[c30 <= 2] = -0.2
    c30_score[(c30 >= 11) & (c30 <= 12)] = -0.3
    c30_score[c30 >= 13] = -0.7
    
    # 邻号评分: 上期 (data[last_i]) 邻号 ±1/±2
    last_nums = data[last_i].tolist()
    nbr_score = np.zeros(80)
    for num in last_nums:
        for d in [-2, -1, 1, 2]:
            if 1 <= num+d <= 80:
                nbr_score[num+d-1] += 0.3
    nbr_score = nbr_score / 20
    
    scores = c5_score + c10_score + c30_score + nbr_score
    return scores

# ============= V20 选号 (V19 重号约束) =============
def v20_predict(last_i, top_n=10, max_repeat_per_lag=7, n_lags=3):
    lag_sets = []
    for lag in range(1, n_lags + 1):
        if last_i + lag - 1 < n_periods:
            lag_sets.append((lag, set(data[last_i + lag - 1].tolist())))
    
    scores = v20_score(last_i)
    sorted_idx = np.argsort(scores)[::-1]
    
    selected = []
    lag_repeats = {lag: 0 for lag, _ in lag_sets}
    
    for idx in sorted_idx:
        num = int(idx) + 1
        if num in selected:
            continue
        skip = False
        for lag, lag_set in lag_sets:
            if num in lag_set:
                if lag_repeats[lag] >= max_repeat_per_lag:
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
    
    return sorted(selected[:top_n]), lag_repeats

# ============= V16.E / V18.C 基线 =============
def v16_e_score(last_i):
    """正确语义: 上期前 k 期 = data[last_i + k]"""
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

# ============= 回测 =============
def backtest(n_test=200, top_n_list=[6, 9, 20, 30]):
    print(f'\n开始 {n_test} 期回测 (V20 v2)...')
    
    results = {}
    for top_n in top_n_list:
        results[top_n] = {
            'v20': {'hits': [], 'rep_l1': [], 'rep_l2': [], 'rep_l3': []},
            'v18_c': {'hits': [], 'rep_l1': [], 'rep_l2': [], 'rep_l3': []},
            'v16_e': {'hits': [], 'rep_l1': [], 'rep_l2': [], 'rep_l3': []},
        }
    
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
            # V20
            pred, _ = v20_predict(last_i, top_n=top_n, max_repeat_per_lag=7, n_lags=3)
            pred_set = set(pred)
            results[top_n]['v20']['hits'].append(len(pred_set & curr_set))
            for i, s in enumerate(lag_sets_data):
                results[top_n]['v20'][f'rep_l{i+1}'].append(len(pred_set & s))
            
            # V18.C 基线 (V16.E + 上期重号 ≤25%)
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
            
            # V16.E 基线 (无约束)
            sel = [int(idx) + 1 for idx in sorted_idx[:top_n]]
            sel_set = set(sel)
            results[top_n]['v16_e']['hits'].append(len(sel_set & curr_set))
            for i, s in enumerate(lag_sets_data):
                results[top_n]['v16_e'][f'rep_l{i+1}'].append(len(sel_set & s))
    
    print('\n' + '=' * 110)
    print(f'{"TopN":<6} {"方案":<22} {"平均命中":<15} {"重L1":<8} {"重L2":<8} {"重L3":<8} {"≥8中":<10} {"≥10中":<10} {"vs V16.E"}')
    print('=' * 110)
    
    summary = {}
    for top_n in top_n_list:
        for ver in ['v16_e', 'v18_c', 'v20']:
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
                'v20': 'V20 多窗口频次 ⭐',
                'v18_c': 'V18.C 基线',
                'v16_e': 'V16.E (无约束)'
            }[ver]
            baseline_avg = np.mean(results[top_n]['v16_e']['hits'])
            delta = (avg_hit - baseline_avg) / baseline_avg * 100 if baseline_avg > 0 else 0
            print(f'{top_n:<6} {ver_name:<22} {avg_hit:.2f}/{top_n} ({avg_hit/top_n*100:.1f}%)  '
                  f'{avg_l1:.2f}    {avg_l2:.2f}    {avg_l3:.2f}    '
                  f'{ge8:.1f}%      {ge10:.1f}%      {delta:+.1f}%')
            summary[ver] = {
                'top_n': top_n,
                'avg_hit': round(float(avg_hit), 3),
                'avg_rep_l1': round(float(avg_l1), 3),
                'avg_rep_l2': round(float(avg_l2), 3),
                'avg_rep_l3': round(float(avg_l3), 3),
                'ge8_pct': round(ge8, 2),
                'ge10_pct': round(ge10, 2),
                'delta_vs_v16e': round(delta, 2),
            }
        print()
    
    return summary, results

# ============= 实战 =============
def predict_target():
    target_period = '2026201'
    last_i = 0  # data[0] = 2026200 是"上期", 预测 2026201
    
    # 但 last_i=0 时 last_i+30=30 < n_periods=2021 OK
    print(f'\n=== 实战: 预测 {target_period} ===')
    print(f'上期 2026200 号码: {sorted(data[last_i].tolist())}')
    
    # 取前 30 候选
    pred_30, _ = v20_predict(last_i, top_n=30, max_repeat_per_lag=7)
    counts = get_counts(last_i)
    
    print(f'\n=== Top 30 候选的多窗口特征 ===')
    print(f'{"号":<5} {"c5":<5} {"c10":<5} {"c30":<5} {"信号"}')
    print('-' * 60)
    for n in pred_30[:15]:
        c5 = int(counts[5][n-1])
        c10 = int(counts[10][n-1])
        c30 = int(counts[30][n-1])
        signals = []
        if c5 in [0, 1, 2]: signals.append('c5佳')
        elif c5 >= 4: signals.append('c5热')
        if 2 <= c10 <= 4: signals.append('c10加分⭐')
        elif c10 >= 5: signals.append('c10转热')
        elif c10 == 0: signals.append('c10=0')
        if 5 <= c30 <= 10: signals.append('c30加分')
        elif c30 >= 13: signals.append('c30过热')
        # 邻号
        is_nbr = any(abs(n - m) in [1, 2] for m in data[last_i])
        if is_nbr: signals.append('邻号✓')
        sig_str = ' '.join(signals) if signals else '中性'
        print(f'{n:<5} {c5:<5} {c10:<5} {c30:<5} {sig_str}')
    
    # 主推方案
    print(f'\n--- 主推 TopN ---')
    predictions = {}
    for top_n in [10, 6, 4]:
        pred, lag_rep = v20_predict(last_i, top_n=top_n, max_repeat_per_lag=7)
        print(f'  Top{top_n}: {pred}')
        print(f'    lag=1 重 {lag_rep[1]}, lag=2 重 {lag_rep[2]}, lag=3 重 {lag_rep[3]}')
        predictions[f'V20_top{top_n}'] = {
            'numbers': pred,
            'lag_repeats': {f'lag_{k}': v for k, v in lag_rep.items()},
            'method': 'V20 多窗口频次 + V19 ≤7 逐期重号约束'
        }
    
    return predictions, target_period

if __name__ == '__main__':
    summary, results = backtest(n_test=200, top_n_list=[6, 9, 20, 30])
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
            'method': 'V20 多窗口频次加权 (修复版)',
            'lottery': '快乐8',
            'target_period': target,
            'target_open_time': '2026-07-30 21:30',
            'n_periods_backtest': 200,
            'note': '修复 last_i 索引 (用 data[last_i+k] 而非 last_i-k)'
        },
        'backtest_summary': clean(summary),
        'predictions': clean(predictions),
    }
    
    out_path = ROOT / 'data' / 'backtest' / f'{target}_predictions_v20.json'
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f'\n✅ 保存: {out_path}')