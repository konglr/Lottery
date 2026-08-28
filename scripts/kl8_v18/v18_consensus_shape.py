"""
KL8 V18: 三模型共识 + 形态约束 + 自适应重号 (Consensus + Morphology + Adaptive Repeat)
==================================================================================

V17 痛点: 重号约束效果有限 (Δ < 1%)
V18 思路: 把"为什么选这号"和"选几个"两个问题分开解决
  - 为什么: 三模型投票 (V16.E ∩ V13 v4 ∩ V10)
  - 选几个: 形态约束 (和值/跨度/奇偶/三区)
  - 自适应: 根据上期重号密度动态调整 max_repeat

V18 子方案:
  - V18.A 三模型投票 + 形态约束 (入门)
  - V18.B V18.A + V17 自适应重号
  - V18.C V18.B + 加权共识 (分数而非交集)
  - V18.D V18.C + 形态门控 (不达标重选)

实战目标: 2026201 期 10/6/4 胆预测
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

# ============= 三个模型评分 =============
def v16_e_score(last_i):
    """V16.E: V10 反向 + 微弱特征调整"""
    return (feat_freq(last_i, 10) * -3.0 +
            feat_repeat(last_i, 3) * -3.0 +
            feat_neighbor(last_i, 5) * -3.0)

def v10_score(last_i):
    """V10 反向基线"""
    return v16_e_score(last_i)

def v13_v4_score(last_i):
    """V13 v4: 短期 S=5 格子相关系数加权 (简化版)
    真实 V13 v4 需要 200 期训练学相关系数, 这里用简化的区域加权
    """
    # 基线 (V10 反向)
    base = v10_score(last_i)
    
    # 短期 S=5 格子数加权
    if last_i + 5 >= n_periods:
        return base
    
    # 计算近 5 期每个号出现次数
    short_count = np.zeros(80, dtype=float)
    for j in range(1, 6):
        if last_i + j >= n_periods: break
        for n in data[last_i+j]:
            short_count[n-1] += 1
    
    # 4 分类: 按频次分高低 (基于近 5 期)
    high_freq_mask = short_count >= 1  # 至少出现 1 次
    
    # V13 v4 核心: 学到的"高短期频次 → 反向"
    # 简化: 对高频号加权 -1.5, 低频号加权 +1.0
    adjustment = np.where(high_freq_mask, -1.5, 1.0)
    
    return base + adjustment

# ============= 共识评分 =============
def consensus_score(last_i, weights=(1.0, 1.0, 1.0)):
    """三模型加权共识评分"""
    s16 = v16_e_score(last_i)
    s13 = v13_v4_score(last_i)
    s10 = v10_score(last_i)
    return weights[0] * s16 + weights[1] * s13 + weights[2] * s10

# ============= 形态约束 =============
def compute_morphology(nums):
    """计算号码列表的形态指标"""
    nums = sorted(nums)
    return {
        'sum': sum(nums),
        'span': max(nums) - min(nums),
        'odd': sum(1 for n in nums if n % 2 == 1),
        'even': sum(1 for n in nums if n % 2 == 0),
        'zones': [
            sum(1 for n in nums if 1 <= n <= 20),
            sum(1 for n in nums if 21 <= n <= 40),
            sum(1 for n in nums if 41 <= n <= 60),
            sum(1 for n in nums if 61 <= n <= 80),
        ],
    }

def morphology_ok(m, target_n):
    """形态是否符合项目 config 标准"""
    if target_n <= 4:
        # 4 胆: 只检查奇偶平衡
        return 1 <= m['odd'] <= target_n - 1
    elif target_n <= 6:
        # 6 胆: 奇偶 2-4, 跨度 < 60
        return 2 <= m['odd'] <= 4 and m['span'] <= 60
    elif target_n <= 10:
        # 10 胆: 奇偶 4-6, 跨度 < 60
        return 4 <= m['odd'] <= 6 and m['span'] <= 65
    else:
        # 更大: 按项目 config
        return True

# ============= V18 核心: 多策略选号 =============
def v18_predict(last_i, top_n=10, strategy='C', max_repeat=None):
    """
    Args:
        last_i: 上期索引
        top_n: 输出号码数
        strategy: 'A' / 'B' / 'C' / 'D'
        max_repeat: 最大允许重号数 (None=自适应)
    """
    last_set = set(data[last_i].tolist())
    
    # 自适应重号阈值
    if max_repeat is None:
        # 基于上期与上上期的重号密度
        if last_i + 1 < n_periods:
            prev_set = set(data[last_i+1].tolist())
            prev_density = len(last_set & prev_set) / 20
            # 上期是"密集型" (重号 > 6), 本期期望回归 → 重号少
            # 上期是"稀疏型" (重号 < 4), 本期期望延续
            if prev_density > 0.3:
                max_repeat = max(1, int(top_n * 0.20))
            elif prev_density < 0.2:
                max_repeat = max(2, int(top_n * 0.30))
            else:
                max_repeat = max(2, int(top_n * 0.25))
        else:
            max_repeat = max(2, int(top_n * 0.25))
    
    # 计算各模型 TopN
    if strategy == 'A':
        # A: 三模型投票交集
        s16 = v16_e_score(last_i)
        s13 = v13_v4_score(last_i)
        s10 = v10_score(last_i)
        top16 = set(np.argsort(s16)[::-1][:20] + 1)
        top13 = set(np.argsort(s13)[::-1][:20] + 1)
        top10 = set(np.argsort(s10)[::-1][:20] + 1)
        vote_pool = top16 & top13 & top10  # 交集
        if len(vote_pool) < top_n * 2:
            vote_pool = top16 & top13  # 退化为两模型交集
        # 用 V16.E 分数排序
        scores = {n: s16[n-1] for n in vote_pool}
        sorted_pool = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
        candidates = sorted_pool
    elif strategy == 'B':
        # B: V18.A + 自适应重号
        return v18_predict(last_i, top_n=top_n, strategy='A', max_repeat=max_repeat)
    elif strategy == 'C':
        # C: 三模型加权共识分数
        scores = consensus_score(last_i, weights=(1.0, 1.0, 1.0))
        sorted_idx = np.argsort(scores)[::-1]
        candidates = [int(i) + 1 for i in sorted_idx]
    elif strategy == 'D':
        # D: V18.C + 形态门控
        scores = consensus_score(last_i, weights=(1.0, 1.0, 1.0))
        sorted_idx = np.argsort(scores)[::-1]
        candidates = [int(i) + 1 for i in sorted_idx]
    
    # 应用重号约束 + 形态约束
    selected = []
    repeats = 0
    for num in candidates:
        if num in selected:
            continue
        if num in last_set:
            if repeats >= max_repeat:
                continue
            repeats += 1
        
        # 形态门控 (仅 D)
        if strategy == 'D':
            trial = selected + [num]
            if len(trial) >= 3:  # 至少 3 个才开始检查
                m = compute_morphology(trial)
                if not morphology_ok(m, len(trial)):
                    continue
        
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
            'v18_a': {'hits': [], 'repeats': []},
            'v18_b': {'hits': [], 'repeats': []},
            'v18_c': {'hits': [], 'repeats': []},
            'v18_d': {'hits': [], 'repeats': []},
            'v17_c': {'hits': [], 'repeats': []},  # 对照
            'v16_e': {'hits': [], 'repeats': []},   # 基线
        }
    
    strategies = ['A', 'B', 'C', 'D']
    ver_map = {'A': 'v18_a', 'B': 'v18_b', 'C': 'v18_c', 'D': 'v18_d'}
    
    for k in range(n_test):
        last_i = k + 1
        if last_i >= n_periods:
            break
        last = data[last_i]
        curr = data[k]
        last_set = set(last.tolist())
        curr_set = set(curr.tolist())
        
        for top_n in top_n_list:
            for s in strategies:
                pred = v18_predict(last_i, top_n=top_n, strategy=s)
                pred_set = set(pred)
                results[top_n][ver_map[s]]['hits'].append(len(pred_set & curr_set))
                results[top_n][ver_map[s]]['repeats'].append(len(pred_set & last_set))
            
            # 对照: V17.C (top_n 重号≤n*0.23)
            pred_v17 = v18_predict(last_i, top_n=top_n, strategy='C', 
                                   max_repeat=int(top_n * 0.23))
            pred_v17_set = set(pred_v17)
            results[top_n]['v17_c']['hits'].append(len(pred_v17_set & curr_set))
            results[top_n]['v17_c']['repeats'].append(len(pred_v17_set & last_set))
            
            # 基线: V16.E 无约束
            pred_baseline = v18_predict(last_i, top_n=top_n, strategy='C', max_repeat=top_n)
            baseline_set = set(pred_baseline)
            results[top_n]['v16_e']['hits'].append(len(baseline_set & curr_set))
            results[top_n]['v16_e']['repeats'].append(len(baseline_set & last_set))
    
    print('\n' + '=' * 105)
    print(f'{"TopN":<6} {"方案":<22} {"平均命中":<15} {"平均重号":<12} {"≥8中":<10} {"≥10中":<10} {"对比基线"}')
    print('=' * 105)
    
    summary = {}
    for top_n in top_n_list:
        for ver in ['v16_e', 'v18_a', 'v18_b', 'v18_c', 'v18_d', 'v17_c']:
            hits = results[top_n][ver]['hits']
            reps = results[top_n][ver]['repeats']
            if len(hits) == 0:
                continue
            avg_hit = np.mean(hits)
            avg_rep = np.mean(reps)
            ge8 = sum(1 for h in hits if h >= 8) / len(hits) * 100
            ge10 = sum(1 for h in hits if h >= 10) / len(hits) * 100
            ver_name = {
                'v18_a': 'V18.A 三模型投票',
                'v18_b': 'V18.B +自适应重号',
                'v18_c': 'V18.C 加权共识',
                'v18_d': 'V18.D +形态门控',
                'v17_c': 'V17.C 对照',
                'v16_e': 'V16.E 基线'
            }[ver]
            baseline_hits = results[top_n]['v16_e']['hits']
            baseline_avg = np.mean(baseline_hits) if baseline_hits else 0
            delta = (avg_hit - baseline_avg) / baseline_avg * 100 if baseline_avg > 0 else 0
            print(f'{top_n:<6} {ver_name:<22} {avg_hit:.2f}/{top_n} ({avg_hit/top_n*100:.1f}%)  '
                  f'{avg_rep:.2f}/{top_n:<6} {ge8:.1f}%      {ge10:.1f}%      {delta:+.1f}%')
            summary[ver] = {
                'top_n': top_n,
                'avg_hit': round(float(avg_hit), 3),
                'avg_rep': round(float(avg_rep), 3),
                'ge8_pct': round(ge8, 2),
                'ge10_pct': round(ge10, 2),
                'delta_vs_baseline': round(delta, 2),
            }
        print()
    
    return summary, results

# ============= 实战预测 2026201 =============
def predict_target():
    """预测 2026201 (目标期)
    - 2026201 7/30 21:30 开, 现在 14:42, 未开
    - 数据最新是 2026200 (7/29 已开)
    - 上期 = 2026200, 目标 = 2026201
    - 所以 last_i = 0 (data[0] = 2026200)
    """
    target_period = '2026201'
    last_i = 0  # 上期 = 2026200
    
    print(f'\n=== 实战: 预测 {target_period} (idx=target) ===')
    print(f'上期 = df.iloc[{last_i}]["期号"] = {df.iloc[last_i]["期号"]}')
    
    last = data[last_i]
    last_set = set(last.tolist())
    print(f'上期号码: {sorted(last)}')
    
    # 计算上期与上上期的重号密度
    if last_i + 1 < n_periods:
        prev_set = set(data[last_i+1].tolist())
        prev_density = len(last_set & prev_set) / 20
        print(f'上上期-上期 重号密度: {prev_density:.1%} ({"密集" if prev_density > 0.3 else "稀疏" if prev_density < 0.2 else "正常"})')
    
    # 4 个 V18 方案
    all_predictions = {}
    for top_n in [10, 6, 4]:
        print(f'\n--- Top{top_n} ---')
        for strat in ['A', 'B', 'C', 'D']:
            pred = v18_predict(last_i, top_n=top_n, strategy=strat)
            repeats = sorted(set(pred) & last_set)
            m = compute_morphology(pred)
            print(f'  V18.{strat}: {pred}')
            print(f'    重号 {len(repeats)}/{top_n}, 和值={m["sum"]}, 跨度={m["span"]}, '
                  f'奇={m["odd"]}, 三区={m["zones"]}')
            all_predictions[f'V18.{strat}_top{top_n}'] = {
                'numbers': pred,
                'repeats': repeats,
                'morphology': m
            }
    
    return all_predictions, target_period

# ============= 主程序 =============
if __name__ == '__main__':
    # 1. 回测
    summary, results = backtest(n_test=200, top_n_list=[9, 20, 30])
    
    # 2. 实战
    predictions, target = predict_target()
    
    # 3. 验证 (如果有 2026201 实际开奖)
    # 2026201 未开, 跳过验证
    
    # 4. 转换 np.int64 → int (json 序列化)
    def clean(obj):
        if isinstance(obj, dict):
            return {k: clean(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [clean(x) for x in obj]
        elif isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        else:
            return obj
    
    output = {
        'meta': {
            'created_at': datetime.now().strftime('%Y-%m-%d %H:%M GMT+8'),
            'method': 'V18 三模型共识 + 形态约束 + 自适应重号',
            'lottery': '快乐8',
            'target_period': target,
            'target_open_time': '2026-07-30 21:30',
            'n_periods_backtest': 200,
            'note': 'V18.A 投票 / V18.B 自适应重号 / V18.C 加权共识 / V18.D 形态门控'
        },
        'backtest_summary': clean(summary),
        'predictions': clean(predictions),
    }
    
    out_path = ROOT / 'data' / 'backtest' / f'{target}_predictions_v18.json'
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f'\n✅ 保存: {out_path}')
