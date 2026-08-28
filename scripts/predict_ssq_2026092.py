"""
SSQ 2026092 期 FSW.SSQ1.F 预测脚本
===================================

配置 (2026-08-04 16:21 调优确认):
- 红球: span_3 (W=0.5) + span_5 (W=0.3) + span_10 (W=0.3)
- 蓝球: span_3 (W=0.5) + span_5 (W=0.3) + repeat (W=0.3)
- TopN: 6 (核心胆) + 12 (复式) + 18 (大复式)
"""
import json
import numpy as np
from pathlib import Path

ROOT = Path.home() / 'Library/Mobile Documents/com~apple~CloudDocs/PycharmProjects/Lottery'
import sys
sys.path.insert(0, str(ROOT))
from lottery_data import LotteryData

ld = LotteryData(ROOT)
df, conf = ld.load('双色球')

# 数据列:红球1-6, 蓝球 (frontWinningNum / backWinningNum)
red_cols = [f'红球{i}' for i in range(1, 7)]
df['red'] = df[red_cols].apply(lambda row: [int(x) for x in row], axis=1)
df['blue'] = df['backWinningNum'].astype(int)

# 数据降序排 (最新在前)
data_red = df['red'].tolist()
data_blue = df['blue'].tolist()
periods = df['期号'].tolist()
n_periods = len(data_red)

LAST_PERIOD = '2026091'
TARGET_PERIOD = '2026092'
last_i = periods.index(LAST_PERIOD)

print(f'数据: {n_periods} 期')
print(f'上期 {LAST_PERIOD}: 红球={sorted(data_red[last_i])}, 蓝球={data_blue[last_i]}')
print(f'预测目标: {TARGET_PERIOD}')
print(f'下期开奖时间: 2026-08-12 (周二) 21:30')
print()

# ============= FSW 评分函数 =============
def feat_span(num, last_i, span):
    """近 span 期中,num 出现的次数 (1-33)"""
    cnt = 0
    for k in range(1, span + 1):
        if last_i + k < n_periods:
            if num in data_red[last_i + k]:
                cnt += 1
    return cnt


def feat_repeat(num, last_i):
    """上期是否重号"""
    if last_i < n_periods:
        return 1 if num in data_red[last_i] else 0
    return 0


def feat_neighbor(num, last_i, span=1):
    """近 span 期中,num-1/num+1 出现次数"""
    cnt = 0
    for k in range(1, span + 1):
        if last_i + k < n_periods:
            neighbors = set()
            for n in data_red[last_i + k]:
                if abs(n - num) == 1:
                    neighbors.add(n)
            cnt += len(neighbors)
    return cnt


# FSW.SSQ1.F 红球评分
def score_red_ssq1f(num, last_i):
    s = (feat_span(num, last_i, 3) * 0.5 +
         feat_span(num, last_i, 5) * 0.3 +
         feat_span(num, last_i, 10) * 0.3)
    return s


# FSW.SSQ1.F 蓝球评分 (1-16 池)
def score_blue_ssq1f(num, last_i):
    s = (feat_span(num, last_i, 3) * 0.5 +
         feat_span(num, last_i, 5) * 0.3 +
         feat_repeat(num, last_i) * 0.3)
    return s


# ============= 主流程 =============
red_scores = {n: score_red_ssq1f(n, last_i) for n in range(1, 34)}
blue_scores = {n: score_blue_ssq1f(n, last_i) for n in range(1, 17)}

# 排序
red_ranked = sorted(red_scores.items(), key=lambda x: -x[1])
blue_ranked = sorted(blue_scores.items(), key=lambda x: -x[1])

print('=' * 70)
print(f'【红球评分排序】 (FSW.SSQ1.F: span_3 + span_5 + span_10)')
print('=' * 70)
print(f'{"排名":<5} {"号码":<6} {"评分":<8} {"最近3期":<8} {"最近5期":<8} {"最近10期":<8}')
print('-' * 70)
for rank, (num, score) in enumerate(red_ranked[:18], 1):
    s3 = feat_span(num, last_i, 3)
    s5 = feat_span(num, last_i, 5)
    s10 = feat_span(num, last_i, 10)
    print(f'{rank:<5} {num:<6} {score:<8.2f} {s3:<8} {s5:<8} {s10:<8}')

print()
print('=' * 70)
print(f'【蓝球评分排序】 (FSW.SSQ1.F: span_3 + span_5 + repeat)')
print('=' * 70)
print(f'{"排名":<5} {"号码":<6} {"评分":<8} {"最近3期":<8} {"最近5期":<8} {"是否上期重":<10}')
print('-' * 70)
for rank, (num, score) in enumerate(blue_ranked[:8], 1):
    s3 = feat_span(num, last_i, 3)
    s5 = feat_span(num, last_i, 5)
    rep = feat_repeat(num, last_i)
    print(f'{rank:<5} {num:<6} {score:<8.2f} {s3:<8} {s5:<8} {rep:<10}')

# ============= 选号 =============
top6 = [num for num, _ in red_ranked[:6]]
top9 = [num for num, _ in red_ranked[:9]]
top12 = [num for num, _ in red_ranked[:12]]
top18 = [num for num, _ in red_ranked[:18]]
blue_top3 = [num for num, _ in blue_ranked[:3]]

# 形态检查
def morphology(nums):
    nums = sorted(nums)
    return {
        '和值': sum(nums),
        '跨度': nums[-1] - nums[0],
        '奇数个数': sum(1 for n in nums if n % 2 == 1),
        '3区配额': f'{sum(1 for n in nums if 1<=n<=11)}/{sum(1 for n in nums if 12<=n<=22)}/{sum(1 for n in nums if 23<=n<=33)}',
        '连号对': sum(1 for i in range(len(nums) - 1) if nums[i+1] - nums[i] == 1),
    }

print()
print('=' * 70)
print(f'【红球胆码】 (FSW.SSQ1.F)')
print('=' * 70)
print(f'\nTop 6 (核心胆): {top6}')
print(f'  形态: {morphology(top6)}')
print(f'\nTop 9 (复式):   {top9}')
print(f'  形态: {morphology(top9)}')
print(f'\nTop 12 (大复式): {top12}')
print(f'  形态: {morphology(top12)}')
print(f'\nTop 18 (全包): {top18}')
print(f'  形态: {morphology(top18)}')

print()
print('=' * 70)
print(f'【蓝球胆码】 (FSW.SSQ1.F)')
print('=' * 70)
print(f'\nTop 3: {blue_top3}')
print(f'  Top 5: {[num for num, _ in blue_ranked[:5]]}')

# ============= 上期 2026091 实际命中检查 =============
print()
print('=' * 70)
print(f'【参考】 上期 2026091 实际开奖')
print('=' * 70)
print(f'  红球: {sorted(data_red[last_i])}')
print(f'  蓝球: {data_blue[last_i]}')

actual_red = set(data_red[last_i])
actual_blue = data_blue[last_i]

for label, top in [('Top 6', top6), ('Top 9', top9), ('Top 12', top12), ('Top 18', top18)]:
    hits = sorted(set(top) & actual_red)
    print(f'\n  {label} 命中 {len(hits)}/6: {hits}')

if actual_blue in blue_top3:
    print(f'\n  蓝球 Top 3 命中: {actual_blue} ⭐')
else:
    print(f'\n  蓝球 Top 3: {blue_top3}, 实际: {actual_blue}, 命中: 否')

# ============= 写入 JSON =============
out = {
    'meta': {
        'created_at': '2026-08-10 18:31 GMT+8',
        'lottery': '双色球',
        'lottery_code': 'ssq',
        'target_period': TARGET_PERIOD,
        'prior_period': LAST_PERIOD,
        'prior_red': sorted(data_red[last_i]),
        'prior_blue': int(data_blue[last_i]),
        'open_time_predicted': '2026-08-12 (周二) 21:30',
        'method': 'SSQ-FSW.SSQ1.F: span_3 (W=0.5) + span_5 (W=0.3) + span_10 (W=0.3)',
        'parameters': {
            'red_pool': '1-33, 选 6 个',
            'blue_pool': '1-16, 选 1 个',
            'features_red': 'span_3 (W=0.5) + span_5 (W=0.3) + span_10 (W=0.3)',
            'features_blue': 'span_3 (W=0.5) + span_5 (W=0.3) + repeat (W=0.3)',
            'top_n_red': '6 (核心胆) + 9 + 12 (复式) + 18 (大复式)',
            'top_n_blue': 3,
        },
        'backtest_500_periods_top12': {
            'FSW.SSQ1.F span_3+5+10 (推荐)': {
                'avg': '2.238/6',
                'ge3': '40.6%',
                'ge4': '12.0%',
                'ge5': '1.4%',
            },
        },
        'morphology_check_top6': morphology(top6),
        'author': 'Lucky / MiniMax-M3',
    },
    'predictions': {
        'SSQ-FSW.SSQ1.F': {
            'method': 'span_3 + span_5 + span_10 多窗口 (调优配置)',
            'red_top_6_dan': top6,
            'red_top_9': top9,
            'red_top_12_drag': top12,
            'red_top_18_full': top18,
            'blue_top_3': blue_top3,
            'morphology_top_6': morphology(top6),
            'morphology_top_12': morphology(top12),
            'all_red_scores': {str(n): float(s) for n, s in red_ranked},
            'all_blue_scores': {str(n): float(s) for n, s in blue_ranked},
        },
    },
}

out_path = ROOT / 'data' / 'backtest' / f'{TARGET_PERIOD}_predictions_ssq_fsw.json'
with open(out_path, 'w', encoding='utf-8') as f:
    json.dump(out, f, ensure_ascii=False, indent=2)
print(f'\n✓ 已写入 {out_path.name}')