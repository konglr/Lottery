"""
KL8 V18 (扩展): 上 2 期重号约束 (Two-Period Repeat Constraint)
==================================================================
V18 基础上, max_repeat 同时排除上期 + 上上期

新方案:
  - V18.E: V18.C 加权共识 + max_repeat(含上 2 期)≤ X
  - V18.F: V18.C 加权共识 + 严格 2 期任一重号 ≤ X
  - V18.G: V18.F + 自适应阈值 (基于上 2 期重叠密度)
"""
import sys
from pathlib import Path
sys.path.insert(0, '/Users/clarkkong/.openclaw/workspace/agents/lucky')
from lottery_data import LotteryData
import pandas as pd
import numpy as np
import json
from datetime import datetime
from collections import Counter

ROOT = Path.home() / 'Library/Mobile Documents/com~apple~CloudDocs/PycharmProjects/Lottery'
ld = LotteryData(ROOT)
df, conf = ld.load('快乐8')
red_cols = [f'红球{i}' for i in range(1, 21)]
data = df[red_cols].astype(int).values
n_periods = len(data)
print(f'快乐8 数据: {n_periods} 期, 范围: {df.iloc[-1]["期号"]} ~ {df.iloc[0]["期号"]}')

# ============= 基础特征函数 =============
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

# ============= 三模型评分 =============
def v16_e_score(last_i):
    return (feat_freq(last_i, 10) * -3.0 +
            feat_repeat(last_i, 3) * -3.0 +
            feat_neighbor(last_i, 5) * -3.0)

def v13_v4_score(last_i):
    base = v16_e_score(last_i)
    if last_i + 5 >= n_periods:
        return base
    short_count = np.zeros(80, dtype=float)
    for j in range(1, 6):
        if last_i + j >= n_periods: break
        for n in data[last_i+j]:
            short_count[n-1] += 1
    high_freq_mask = short_count >= 1
    adjustment = np.where(high_freq_mask, -1.5, 1.0)
    return base + adjustment

def v10_score(last_i):
    return v16_e_score(last_i)

def consensus_score(last_i):
    """V18.C 加权共识"""
    return v16_e_score(last_i) + v13_v4_score(last_i) + v10_score(last_i)

# ============= 上 2 期重号约束 =============
def v18_predict(last_i, top_n=10, max_repeat_window=2, max_repeat=None):
    """
    Args:
        last_i: 上期索引 (data[last_i] = 上期, data[last_i+1] = 上上期)
        top_n: 输出号码数
        max_repeat_window: 重号检查窗口 (2 = 上期+上上期)
        max_repeat: 最大允许在窗口内的重号数
    """
    # 窗口内的期号集合 (last_i 到 last_i + window - 1)
    window_set = set()
    for j in range(max_repeat_window):
        if last_i + j < n_periods:
            window_set.update(data[last_i+j].tolist())
    
    scores = consensus_score(last_i)
    sorted_idx = np.argsort(scores)[::-1]
    
    if max_repeat is None:
        max_repeat = max(1, int(top_n * 0.20))
    
    selected = []
    repeats = 0
    for idx in sorted_idx:
        num = int(idx) + 1
        if num in selected:
            continue
        if num in window_set:
            if repeats >= max_repeat:
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
            'v18_e': {'hits': [], 'repeats_l1': [], 'repeats_window': []},  # 2 期任一 ≤ 2
            'v18_f': {'hits': [], 'repeats_l1': [], 'repeats_window': []},  # 2 期任一 ≤ 3
            'v18_g': {'hits': [], 'repeats_l1': [], 'repeats_window': []},  # 自适应
            'v18_c_baseline': {'hits': [], 'repeats_l1': [], 'repeats_window': []},  # 仅上期约束
        }
    
    for k in range(n_test):
        last_i = k + 1
        if last_i + 1 >= n_periods:  # 需要至少 2 期历史
            break
        
        last = data[last_i]
        curr = data[k]
        last_set = set(last.tolist())
        # 上 2 期任一
        window_set = last_set | set(data[last_i+1].tolist())
        curr_set = set(curr.tolist())
        
        for top_n in top_n_list:
            # V18.E: 2 期任一重号 ≤ 2
            max_rep = max(1, int(top_n * 0.20))
            pred = v18_predict(last_i, top_n=top_n, max_repeat_window=2, max_repeat=max_rep)
            pred_set = set(pred)
            results[top_n]['v18_e']['hits'].append(len(pred_set & curr_set))
            results[top_n]['v18_e']['repeats_l1'].append(len(pred_set & last_set))
            results[top_n]['v18_e']['repeats_window'].append(len(pred_set & window_set))
            
            # V18.F: 2 期任一重号 ≤ 3 (更宽松)
            max_rep = max(1, int(top_n * 0.25))
            pred_f = v18_predict(last_i, top_n=top_n, max_repeat_window=2, max_repeat=max_rep)
            pred_f_set = set(pred_f)
            results[top_n]['v18_f']['hits'].append(len(pred_f_set & curr_set))
            results[top_n]['v18_f']['repeats_l1'].append(len(pred_f_set & last_set))
            results[top_n]['v18_f']['repeats_window'].append(len(pred_f_set & window_set))
            
            # V18.G: 自适应 (基于上 2 期重叠密度)
            prev_density = len(window_set) / 40  # 2 期合并去重后的密度
            # 高密度 (窗口有 > 30 个不重复号 → 上 2 期差异大) → 期望本期也跳出
            # 低密度 (窗口有 < 25 个不重复号 → 上 2 期重叠多) → 期望延续
            if prev_density > 0.75:
                max_rep_g = max(1, int(top_n * 0.15))
            elif prev_density < 0.60:
                max_rep_g = max(2, int(top_n * 0.30))
            else:
                max_rep_g = max(1, int(top_n * 0.22))
            pred_g = v18_predict(last_i, top_n=top_n, max_repeat_window=2, max_repeat=max_rep_g)
            pred_g_set = set(pred_g)
            results[top_n]['v18_g']['hits'].append(len(pred_g_set & curr_set))
            results[top_n]['v18_g']['repeats_l1'].append(len(pred_g_set & last_set))
            results[top_n]['v18_g']['repeats_window'].append(len(pred_g_set & window_set))
            
            # V18.C 基线 (仅上期约束)
            pred_baseline = v18_predict(last_i, top_n=top_n, max_repeat_window=1, 
                                        max_repeat=max(2, int(top_n * 0.25)))
            baseline_set = set(pred_baseline)
            results[top_n]['v18_c_baseline']['hits'].append(len(baseline_set & curr_set))
            results[top_n]['v18_c_baseline']['repeats_l1'].append(len(baseline_set & last_set))
            results[top_n]['v18_c_baseline']['repeats_window'].append(len(baseline_set & window_set))
    
    print('\n' + '=' * 115)
    print(f'{"TopN":<6} {"方案":<25} {"平均命中":<15} {"重上期":<10} {"重2期":<10} {"≥8中":<10} {"≥10中":<10} {"对比基线"}')
    print('=' * 115)
    
    summary = {}
    for top_n in top_n_list:
        for ver in ['v18_c_baseline', 'v18_e', 'v18_f', 'v18_g']:
            hits = results[top_n][ver]['hits']
            rep_l1 = results[top_n][ver]['repeats_l1']
            rep_w = results[top_n][ver]['repeats_window']
            if len(hits) == 0:
                continue
            avg_hit = np.mean(hits)
            avg_rep_l1 = np.mean(rep_l1)
            avg_rep_w = np.mean(rep_w)
            ge8 = sum(1 for h in hits if h >= 8) / len(hits) * 100
            ge10 = sum(1 for h in hits if h >= 10) / len(hits) * 100
            ver_name = {
                'v18_e': 'V18.E 2期重号≤2 (硬)',
                'v18_f': 'V18.F 2期重号≤3 (软)',
                'v18_g': 'V18.G 自适应阈值',
                'v18_c_baseline': 'V18.C 基线(仅上期)'
            }[ver]
            baseline_hits = results[top_n]['v18_c_baseline']['hits']
            baseline_avg = np.mean(baseline_hits) if baseline_hits else 0
            delta = (avg_hit - baseline_avg) / baseline_avg * 100 if baseline_avg > 0 else 0
            print(f'{top_n:<6} {ver_name:<25} {avg_hit:.2f}/{top_n} ({avg_hit/top_n*100:.1f}%)  '
                  f'{avg_rep_l1:.2f}/{top_n:<6} {avg_rep_w:.2f}/{top_n:<6} {ge8:.1f}%      {ge10:.1f}%      {delta:+.1f}%')
            summary[ver] = {
                'top_n': top_n,
                'avg_hit': round(float(avg_hit), 3),
                'avg_rep_l1': round(float(avg_rep_l1), 3),
                'avg_rep_window2': round(float(avg_rep_w), 3),
                'ge8_pct': round(ge8, 2),
                'ge10_pct': round(ge10, 2),
                'delta_vs_baseline': round(delta, 2),
            }
        print()
    
    return summary, results

# ============= 实战预测 =============
def predict_target():
    target_period = '2026201'
    last_i = 0  # data[0] = 2026200 (上期)
    
    # 上 2 期
    last_1 = data[last_i]
    last_2 = data[last_i + 1] if last_i + 1 < n_periods else None
    window_set = set(last_1.tolist())
    if last_2 is not None:
        window_set |= set(last_2.tolist())
    
    print(f'\n=== 实战: 预测 {target_period} ===')
    print(f'上期 2026200 号码: {sorted(last_1)}')
    print(f'上上期 2026199 号码: {sorted(last_2) if last_2 is not None else "无"}')
    print(f'上 2 期合并去重后共 {len(window_set)} 个不重复号 (80 中占 {len(window_set)/80*100:.1f}%)')
    
    # 历史 200 期平均: 上 2 期合并去重占比
    densities = []
    for k in range(min(200, n_periods - 2)):
        s = set(data[k].tolist()) | set(data[k+1].tolist())
        densities.append(len(s) / 80 * 100)
    avg_density = np.mean(densities)
    print(f'历史 200 期平均上 2 期合并去重占比: {avg_density:.1f}%')
    
    prev_density = len(window_set) / 80
    if prev_density > 0.75:
        density_label = '高密度(差异大)'
    elif prev_density < 0.60:
        density_label = '低密度(重叠多)'
    else:
        density_label = '正常'
    print(f'→ 本期密度 {prev_density*100:.1f}% → {density_label}')
    
    # 4 个 V18 方案 (3 个新增 + 1 个基线)
    all_predictions = {}
    
    # 计算各 top_n 的自适应阈值
    for top_n in [10, 6, 4]:
        print(f'\n--- Top{top_n} ---')
        
        # V18.E (硬): 2 期任一 ≤ top_n * 20%
        max_rep_e = max(1, int(top_n * 0.20))
        pred_e = v18_predict(last_i, top_n=top_n, max_repeat_window=2, max_repeat=max_rep_e)
        rep_l1 = sorted(set(pred_e) & set(last_1))
        rep_w = sorted(set(pred_e) & window_set)
        print(f'  V18.E (2期重号≤{max_rep_e}): {pred_e}')
        print(f'    重上期 {len(rep_l1)} = {rep_l1}, 重2期 {len(rep_w)} = {rep_w}')
        all_predictions[f'V18.E_top{top_n}'] = {
            'numbers': pred_e, 'repeats_l1': rep_l1, 'repeats_window2': rep_w,
            'n_repeat_l1': len(rep_l1), 'n_repeat_window2': len(rep_w),
            'method': f'V18.E 加权共识 + 2期重号≤{max_rep_e}'
        }
        
        # V18.F (软): 2 期任一 ≤ top_n * 25%
        max_rep_f = max(1, int(top_n * 0.25))
        pred_f = v18_predict(last_i, top_n=top_n, max_repeat_window=2, max_repeat=max_rep_f)
        rep_l1 = sorted(set(pred_f) & set(last_1))
        rep_w = sorted(set(pred_f) & window_set)
        print(f'  V18.F (2期重号≤{max_rep_f}): {pred_f}')
        print(f'    重上期 {len(rep_l1)} = {rep_l1}, 重2期 {len(rep_w)} = {rep_w}')
        all_predictions[f'V18.F_top{top_n}'] = {
            'numbers': pred_f, 'repeats_l1': rep_l1, 'repeats_window2': rep_w,
            'n_repeat_l1': len(rep_l1), 'n_repeat_window2': len(rep_w),
            'method': f'V18.F 加权共识 + 2期重号≤{max_rep_f}'
        }
        
        # V18.G (自适应)
        if prev_density > 0.75:
            max_rep_g = max(1, int(top_n * 0.15))
            g_label = '高密度→严格'
        elif prev_density < 0.60:
            max_rep_g = max(2, int(top_n * 0.30))
            g_label = '低密度→宽松'
        else:
            max_rep_g = max(1, int(top_n * 0.22))
            g_label = '正常→适中'
        pred_g = v18_predict(last_i, top_n=top_n, max_repeat_window=2, max_repeat=max_rep_g)
        rep_l1 = sorted(set(pred_g) & set(last_1))
        rep_w = sorted(set(pred_g) & window_set)
        print(f'  V18.G (自适应, {g_label}, ≤{max_rep_g}): {pred_g}')
        print(f'    重上期 {len(rep_l1)} = {rep_l1}, 重2期 {len(rep_w)} = {rep_w}')
        all_predictions[f'V18.G_top{top_n}'] = {
            'numbers': pred_g, 'repeats_l1': rep_l1, 'repeats_window2': rep_w,
            'n_repeat_l1': len(rep_l1), 'n_repeat_window2': len(rep_w),
            'method': f'V18.G 自适应 + 2期重号≤{max_rep_g} ({g_label})'
        }
    
    return all_predictions, target_period, prev_density, density_label

# ============= 主程序 =============
if __name__ == '__main__':
    summary, results = backtest(n_test=200, top_n_list=[9, 20, 30])
    predictions, target, density_pct, density_label = predict_target()
    
    # JSON 序列化
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
            'method': 'V18.E/F/G 上 2 期重号约束',
            'lottery': '快乐8',
            'target_period': target,
            'target_open_time': '2026-07-30 21:30',
            'n_periods_backtest': 200,
            'note': 'V18.E 硬约束(2期≤20%) / V18.F 软约束(2期≤25%) / V18.G 自适应'
        },
        'backtest_summary': clean(summary),
        'window_analysis': {
            'last_1_period': '2026200',
            'last_2_period': '2026199',
            'window_size_unique': int(len(set(data[0].tolist()) | set(data[1].tolist()))),
            'window_density_pct': round(density_pct * 100, 1),
            'density_label': density_label,
            'historical_avg_density_pct': round(float(np.mean([len(set(data[k].tolist()) | set(data[k+1].tolist())) / 80 * 100 for k in range(min(200, n_periods-2))])), 1)
        },
        'predictions': clean(predictions),
    }
    
    out_path = ROOT / 'data' / 'backtest' / f'{target}_predictions_v18_window2.json'
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f'\n✅ 保存: {out_path}')