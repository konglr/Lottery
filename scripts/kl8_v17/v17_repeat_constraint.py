"""
KL8 V17: 重号约束版 (Repeat-Ratio Constrained Predictor)
============================================================

数据约定:
  data[i] = df.iloc[i] 的 20 个号码
  data[0] = 最新一期 (2026200)
  data[1] = 上期 (2026199)

预测语义:
  last_i: 上期在 data 中的索引
  预测目标 = data[last_i - 1]
  
诊断依据:
  - 200 期实测平均重号 4.80/20 (24%)
  - 9 胆方案: 重 1/9 (11%), 命中 3/9 (33%) ← 唯一突破的方案
  - 结论: 重号越少 → 命中率越高

V17 子方案:
  - V17.A 硬约束: 重号 ≤ 2/9 (超严格, 用于 9 胆)
  - V17.B 中约束: 重号 ≤ 4/20 (适中, 用于 20 码)
  - V17.C 软约束: 重号 ≤ 7/30 (宽松, 用于 30 码)
"""
import sys
from pathlib import Path
sys.path.insert(0, '/Users/clarkkong/.openclaw/workspace/agents/lucky')
from lottery_data import LotteryData
import pandas as pd
import numpy as np
import json
from datetime import datetime

ROOT = Path.home() / 'Library/Mobile Documents/com~apple~CloudDocs/PycharmProjects/Lottery'
ld = LotteryData(ROOT)
df, conf = ld.load('快乐8')
red_cols = [f'红球{i}' for i in range(1, 21)]
data = df[red_cols].astype(int).values
n_periods = len(data)
print(f'快乐8 数据: {n_periods} 期, 范围: {df.iloc[-1]["期号"]} ~ {df.iloc[0]["期号"]}')

# ============= 特征函数 =============
def feat_freq(last_i, window):
    if last_i + 1 + window > n_periods: window = n_periods - last_i - 1
    if window <= 0: return np.zeros(80, dtype=float)
    block = data[last_i+1:last_i+1+window]
    return np.bincount(block.flatten(), minlength=81)[1:81].astype(float)

def feat_repeat(last_i, span):
    counts = np.zeros(80, dtype=float)
    for j in range(1, span+1):
        if last_i + j >= n_periods: break
        nums = data[last_i+j]
        counts[nums-1] += 1
    return np.clip(counts, 0, 1)

def feat_neighbor(last_i, span, distance=1):
    covered = np.zeros(80, dtype=float)
    for j in range(1, span+1):
        if last_i + j >= n_periods: break
        for n in data[last_i+j]:
            for d in range(-distance, distance+1):
                if d == 0: continue
                if 1 <= n+d <= 80: covered[n+d-1] += 1
    return covered

# ============= 评分函数 =============
def v16_e_score(last_i):
    """V16.E / V10 反向基线"""
    return (feat_freq(last_i, 10) * -3.0 +
            feat_repeat(last_i, 3) * -3.0 +
            feat_neighbor(last_i, 5) * -3.0)

# ============= V17 核心: 重号约束选号 =============
def v17_predict(last_i, top_n=9, max_repeat=None):
    """
    直接从 80 号池中按 V16.E 分数排序, 重号超限时跳过.
    """
    last_set = set(data[last_i].tolist())
    scores = v16_e_score(last_i)
    sorted_idx = np.argsort(scores)[::-1]
    
    selected = []
    repeats = 0
    for idx in sorted_idx:
        num = int(idx) + 1
        if num in selected:
            continue
        if num in last_set:
            if max_repeat is not None and repeats >= max_repeat:
                continue
            repeats += 1
        selected.append(num)
        if len(selected) >= top_n:
            break
    
    return sorted(selected[:top_n])

# ============= 回测 =============
def backtest(n_test=200, top_n_list=[9, 20, 30]):
    print(f'\n开始 {n_test} 期回测...')
    
    results = {}
    for top_n in top_n_list:
        results[top_n] = {
            'v17_a': {'hits': [], 'repeats': []},
            'v17_b': {'hits': [], 'repeats': []},
            'v17_c': {'hits': [], 'repeats': []},
            'v16_e_baseline': {'hits': [], 'repeats': []},
        }
    
    max_rep_map = {9: 2, 20: 4, 30: 7}
    ver_map = {9: 'v17_a', 20: 'v17_b', 30: 'v17_c'}
    
    for k in range(n_test):
        last_i = k + 1
        if last_i >= n_periods:
            break
        last = data[last_i]
        curr = data[k]
        last_set = set(last.tolist())
        curr_set = set(curr.tolist())
        
        for top_n in top_n_list:
            pred = v17_predict(last_i, top_n=top_n, max_repeat=max_rep_map[top_n])
            pred_set = set(pred)
            results[top_n][ver_map[top_n]]['hits'].append(len(pred_set & curr_set))
            results[top_n][ver_map[top_n]]['repeats'].append(len(pred_set & last_set))
            
            pred_baseline = v17_predict(last_i, top_n=top_n, max_repeat=None)
            baseline_set = set(pred_baseline)
            results[top_n]['v16_e_baseline']['hits'].append(len(baseline_set & curr_set))
            results[top_n]['v16_e_baseline']['repeats'].append(len(baseline_set & last_set))
    
    print('\n' + '=' * 105)
    print(f'{"TopN":<6} {"方案":<25} {"平均命中":<15} {"平均重号":<12} {"≥8中":<10} {"≥10中":<10} {"对比基线"}')
    print('=' * 105)
    
    summary = {}
    for top_n in top_n_list:
        for ver in results[top_n].keys():
            hits = results[top_n][ver]['hits']
            reps = results[top_n][ver]['repeats']
            if len(hits) == 0:
                continue
            avg_hit = np.mean(hits)
            avg_rep = np.mean(reps)
            ge8 = sum(1 for h in hits if h >= 8) / len(hits) * 100
            ge10 = sum(1 for h in hits if h >= 10) / len(hits) * 100
            ver_name = {
                'v17_a': 'V17.A 重号≤2',
                'v17_b': 'V17.B 重号≤4',
                'v17_c': 'V17.C 重号≤7',
                'v16_e_baseline': 'V16.E 基线(无约束)'
            }[ver]
            baseline_hits = results[top_n]['v16_e_baseline']['hits']
            baseline_avg = np.mean(baseline_hits) if baseline_hits else 0
            delta = (avg_hit - baseline_avg) / baseline_avg * 100 if baseline_avg > 0 else 0
            print(f'{top_n:<6} {ver_name:<25} {avg_hit:.2f}/{top_n} ({avg_hit/top_n*100:.1f}%)  '
                  f'{avg_rep:.2f}/{top_n:<6} {ge8:.1f}%      {ge10:.1f}%      {delta:+.1f}%')
            summary[ver] = {
                'top_n': top_n,
                'avg_hit': round(float(avg_hit), 3),
                'avg_rep': round(float(avg_rep), 3),
                'ge8_pct': round(ge8, 2),
                'ge10_pct': round(ge10, 2),
                'delta_vs_baseline': round(delta, 2),
                'n_periods': len(hits)
            }
        print()
    
    return summary, results

# ============= 实战预测 =============
def predict_target(target_period='2026200'):
    print(f'\n=== 实战: 预测 {target_period} (上期 2026199) ===')
    
    last_i = 1
    last = data[last_i]
    last_set = set(last.tolist())
    print(f'上期 2026199 号码: {sorted(last)}')
    
    plans = {
        '9 胆 V17.A (重号≤2)': (9, 2),
        '20 码 V17.B (重号≤4)': (20, 4),
        '30 码 V17.C (重号≤7)': (30, 7),
        '9 胆 V16.E 基线 (无约束)': (9, None),
        '20 码 V16.E 基线 (无约束)': (20, None),
    }
    
    predictions = {}
    for plan_name, (top_n, max_rep) in plans.items():
        pred = v17_predict(last_i, top_n=top_n, max_repeat=max_rep)
        repeats = sorted(set(pred) & last_set)
        print(f'  {plan_name}: {pred}')
        print(f'    重号: {repeats} ({len(repeats)} 个)')
        predictions[plan_name] = {
            'numbers': pred,
            'repeats': repeats,
            'n_repeat': len(repeats)
        }
    
    return predictions

# ============= 主程序 =============
if __name__ == '__main__':
    summary, results = backtest(n_test=200, top_n_list=[9, 20, 30])
    predictions = predict_target('2026200')
    
    print('\n=== 验证: 2026200 实际开奖对照 ===')
    curr = sorted([int(x) for x in data[0]])
    print(f'实际开奖 2026200: {curr}')
    for plan_name, info in predictions.items():
        hits = sorted(set(info['numbers']) & set(curr))
        print(f'  {plan_name}: 命中 {len(hits)}/20 = {hits}')
    
    output = {
        'meta': {
            'created_at': datetime.now().strftime('%Y-%m-%d %H:%M GMT+8'),
            'method': 'V17 重号约束 + V16.E 基线',
            'lottery': '快乐8',
            'target_period': '2026200',
            'n_periods_backtest': 200,
            'note': 'V17 在 V16.E 基础上按 TopN 限定最大重号数'
        },
        'backtest_summary': summary,
        '2026200_predictions': predictions,
        '2026200_actual': curr
    }
    
    out_path = ROOT / 'data' / 'backtest' / '2026200_predictions_v17.json'
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f'\n✅ 保存: {out_path}')
