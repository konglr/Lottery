"""
SSQ 2026094 期 V14 修复版预测脚本
===================================

2026-08-14 15:21 数据方向 bug 修复后版本
- Bug: 之前的回测代码 actual = data_red[i+1] (上一期)
       评分 red_matrix[i+1:i+21] 用 lag1..20
       → 加权 r = data_red[i+1] = actual → 命中率虚高
- 修复: actual = data_red[i] (本期,因为 data_red 降序)

真实回测数据 (489 期):
- FSW Top 6 = 1.17/6 (≥3中 7.4%)
- Naive_20 + 3.0*R = 1.09/6 (≥3中 5.5%) — 跟随机基线一致
- FSW Top 18 复式 = 3.39/18 (≥3中 80.4%)
"""

import sys
import json
import numpy as np
from pathlib import Path

ROOT = Path.home() / 'Library/Mobile Documents/com~apple~CloudDocs/PycharmProjects/Lottery'
sys.path.insert(0, str(ROOT))
from lottery_data import LotteryData

ld = LotteryData(ROOT)
df, conf = ld.load('双色球')

red_cols = [f'红球{i}' for i in range(1, 7)]
df['red'] = df[red_cols].apply(lambda row: sorted([int(x) for x in row]), axis=1)
df['blue'] = df['backWinningNum'].astype(int)

data_red = df['red'].tolist()
data_blue = df['blue'].tolist()
periods = df['期号'].tolist()
n_periods = len(data_red)

print(f'数据: {n_periods} 期 (降序, 最新在前)')

# 当前最新期 = data_red[0] = 2026093
LAST_PERIOD = '2026093'
TARGET_PERIOD = '2026094'
print(f'上期 {LAST_PERIOD}: 红球={data_red[0]}, 蓝球={data_blue[0]}')
print(f'预测目标: {TARGET_PERIOD}')
print(f'下期开奖时间: 2026-08-16 (周日) 21:30')
print()

# 预计算号码 0/1 矩阵
red_matrix = np.zeros((n_periods, 34), dtype=np.int8)
for i, reds in enumerate(data_red):
    for x in reds:
        red_matrix[i, x] = 1

blue_matrix = np.zeros((n_periods, 17), dtype=np.int8)
for i, b in enumerate(data_blue):
    blue_matrix[i, b] = 1


# ============= 评分函数 (修复版) =============
# 关键: 用 lag1..20 (data_red[1..20]) 作为历史,预测下一期 (data_red[0] 之后)
# i 表示 lag 的起点, lag 范围是 [i+1, i+N]
# actual = data_red[i] (i 比 i+1 更近期, 降序)

def score_fsw():
    """FSW.SSQ1.F"""
    s3 = red_matrix[1:4].sum(axis=0)
    s5 = red_matrix[1:6].sum(axis=0)
    s10 = red_matrix[1:11].sum(axis=0)
    return 0.5*s3 + 0.3*s5 + 0.3*s10

def score_naive(win):
    end = min(1 + win, n_periods)
    return red_matrix[1:end].sum(axis=0)

def score_repeat():
    return red_matrix[1].astype(float)


# ============= 主流形态约束 =============
def is_mainstream_strict(combo):
    odd = sum(1 for n in combo if n % 2 == 1)
    if not (2 <= odd <= 4):
        return False
    z1 = sum(1 for n in combo if 1 <= n <= 11)
    z2 = sum(1 for n in combo if 12 <= n <= 22)
    z3 = sum(1 for n in combo if 23 <= n <= 33)
    if max(z1, z2, z3) - min(z1, z2, z3) > 2:
        return False
    s = sum(combo)
    if not (70 <= s <= 130):
        return False
    if max(combo) - min(combo) > 30:
        return False
    return True


from itertools import combinations

def best_combo_strict(pool, scores, n_top, cand_top=12):
    cand = pool[:cand_top]
    best = None
    best_score = -1
    for combo in combinations(cand, n_top):
        if is_mainstream_strict(list(combo)):
            s = sum(scores[n] for n in combo)
            if s > best_score:
                best_score = s
                best = list(combo)
    if best is None:
        return sorted(cand[:n_top])
    return sorted(best)


# ============= 计算 2026094 期预测 =============
fsw = score_fsw()
n20 = score_naive(20)
n30 = score_naive(30)
r = score_repeat()
last_red = data_red[0]

fsw_ranked = np.argsort(-fsw).tolist()
n20_ranked = np.argsort(-n20).tolist()
n30_ranked = np.argsort(-n30).tolist()

# --- 方案 1: FSW + 硬约束 Top 6 ---
print('方案 1: FSW + 硬约束 Top 6')
print('-' * 50)
fsw_strict_top6 = best_combo_strict(fsw_ranked, fsw, 6)
fsw_top6 = sorted(fsw_ranked[:6])

print(f'FSW Top 12 (候选): {sorted(fsw_ranked[:12])}')
print(f'FSW + 严格 Top 6: {fsw_strict_top6}')
print(f'  奇偶: {sum(1 for n in fsw_strict_top6 if n%2==1)}:{6-sum(1 for n in fsw_strict_top6 if n%2==1)}')
print(f'  3 区: z1={sum(1 for n in fsw_strict_top6 if 1<=n<=11)} z2={sum(1 for n in fsw_strict_top6 if 12<=n<=22)} z3={sum(1 for n in fsw_strict_top6 if 23<=n<=33)}')
print(f'  和值: {sum(fsw_strict_top6)}, 跨度: {max(fsw_strict_top6) - min(fsw_strict_top6)}')
print(f'  与上期重号: {sorted(set(fsw_strict_top6) & set(last_red))}')
print(f'[对照] FSW Top 6 (无约束): {fsw_top6}')

# --- 方案 2: Naive_20 + 3.0*R (修复版真实结果) ---
print()
print('方案 2: Naive_20 + 3.0*R (修复版)')
print('-' * 50)
naive_v14 = n20 + 3.0 * r
naive_v14_ranked = np.argsort(-naive_v14).tolist()
naive_top6 = sorted(naive_v14_ranked[:6])

print(f'Naive_20 + 3.0*Repeat Top 6: {naive_top6}')
print(f'  与上期重号: {sorted(set(naive_top6) & set(last_red))} (n={len(set(naive_top6) & set(last_red))})')
print(f'  真实回测: avg 1.09/6, ≥3中 5.5% (跟随机基线 1.09 一致)')

# 对照 Naive_30 + 2.0*R
naive_30_v14 = n30 + 2.0 * r
naive_30_top6 = sorted(np.argsort(-naive_30_v14)[:6].tolist())
print(f'[对照] Naive_30 + 2.0*R Top 6: {naive_30_top6}')

# --- 方案 3: FSW Top 18 复式 ---
print()
print('方案 3: FSW Top 18 复式')
print('-' * 50)
fsw_top12 = sorted(fsw_ranked[:12])
fsw_top15 = sorted(fsw_ranked[:15])
fsw_top18 = sorted(fsw_ranked[:18])

print(f'FSW Top 12: {fsw_top12}')
print(f'FSW Top 15: {fsw_top15}')
print(f'FSW Top 18: {fsw_top18}')

# --- 蓝球 ---
print()
print('蓝球评分')
print('-' * 50)
b_s3 = blue_matrix[1:4].sum(axis=0)
b_s5 = blue_matrix[1:6].sum(axis=0)
b_s10 = blue_matrix[1:11].sum(axis=0)
b_repeat = blue_matrix[1].astype(float)
b_score = 0.5*b_s3 + 0.3*b_s5 + 0.3*b_s10 + 0.3*b_repeat
blue_top3 = [int(b) for b in np.argsort(-b_score)[:3]]
print(f'蓝球 Top 3: {blue_top3}')
print(f'上期蓝球: {data_blue[0]} (重号: {data_blue[0] in blue_top3})')


# ============= 写 JSON =============
def make_json():
    return {
        "meta": {
            "created_at": "2026-08-14 15:21 GMT+8",
            "lottery": "双色球",
            "lottery_code": "ssq",
            "target_period": "2026094",
            "prior_period": "2026093",
            "prior_red": data_red[0],
            "prior_blue": int(data_blue[0]),
            "open_time": "2026-08-16 (周日) 21:30",
            "method": "V14 修复版 — 数据方向 bug 修正后",
            "user_request": "2026-08-14 14:10 V14 三方案 + 15:21 用户发现数据方向 bug 要求验证",
            "bug_history": "修复前 actual = data_red[i+1] 跟 red_matrix[i+1] 重合, 命中率虚高 (Naive_20 + 3.0*R 虚高到 4.35)",
            "backtest_489_periods_corrected": {
                "FSW.SSQ1.F Top 6 (真实基线)": {"avg": "1.168/6", "ge3": "7.4%", "ge4": "0.6%"},
                "FSW.SSQ1.F Top 12 复式": {"avg": "2.284/12", "ge3": "40.3%", "ge4": "11.9%", "ge5": "1.2%"},
                "FSW.SSQ1.F Top 15 复式": {"avg": "2.877/15", "ge3": "62.0%", "ge4": "29.4%", "ge5": "6.3%"},
                "FSW.SSQ1.F Top 18 复式": {"avg": "3.387/18", "ge3": "80.4%", "ge4": "45.2%", "ge5": "16.0%"},
                "Naive_20 纯频次 Top 6": {"avg": "1.145/6", "ge3": "5.3%", "ge4": "0.6%"},
                "Naive_30 纯频次 Top 6": {"avg": "1.098/6", "ge3": "4.9%", "ge4": "0.4%"},
                "Naive_20 + 3.0*R Top 6 (修复前 4.35 → 真实 1.09)": {"avg": "1.086/6", "ge3": "5.5%", "ge4": "0.8%"},
                "Naive_30 + 2.0*R Top 6 (修复前 3.17 → 真实 1.07)": {"avg": "1.072/6", "ge3": "4.5%", "ge4": "0.6%"},
                "FSW + 严格约束 Top 6": {"avg": "1.129/6", "ge3": "7.0%", "ge4": "0.4%"},
                "随机基线 (1.09)": "理论上随机选 6/33 的平均命中 = 6 * 6/33 = 1.09"
            },
            "key_findings_corrected": [
                "Naive_20 + 3.0*R 修复前虚高到 4.35/6 (≥3中 99%),修复后真实 1.09/6 (≥3中 5.5%) — 跟随机基线完全一致",
                "Naive 加 R 加权对 SSQ 无效 — 1.07-1.10 都在随机基线 1.09 附近",
                "FSW Top 6 是 Top 6 单式里最佳 (1.17), 但也只是略超随机 (1.09)",
                "Top 18 复式 ≥3中率 80.4% (修复前虚高 99.8%), 仍是性价比最高的方案",
                "硬约束 (奇偶/3区/冷热) 对 FSW 几乎无影响 (-0.04/期)"
            ],
            "author": "Lucky / MiniMax-M3"
        },
        "predictions": {
            "V14_方案1_FSW_硬约束": {
                "method": "FSW.SSQ1.F + 主流形态硬约束",
                "red_top_6_dan": fsw_strict_top6,
                "morphology": {
                    "奇偶": f"{sum(1 for n in fsw_strict_top6 if n%2==1)}:{6-sum(1 for n in fsw_strict_top6 if n%2==1)}",
                    "3 区": f"z1={sum(1 for n in fsw_strict_top6 if 1<=n<=11)} z2={sum(1 for n in fsw_strict_top6 if 12<=n<=22)} z3={sum(1 for n in fsw_strict_top6 if 23<=n<=33)}",
                    "和值": sum(fsw_strict_top6),
                    "跨度": max(fsw_strict_top6) - min(fsw_strict_top6)
                },
                "blue_top_3": blue_top3,
                "with_repeat_2026093": sorted(set(fsw_strict_top6) & set(last_red)),
                "note": "修复版真实回测: FSW+约束 = 1.13/6, vs FSW 基线 1.17/6 (-0.04, 接近随机)"
            },
            "V14_方案2_Naive_20_3R": {
                "method": "Naive_20 (近 20 期频次) + 3.0*上期重号加权",
                "red_top_6_dan": naive_top6,
                "morphology": {
                    "奇偶": f"{sum(1 for n in naive_top6 if n%2==1)}:{6-sum(1 for n in naive_top6 if n%2==1)}",
                    "3 区": f"z1={sum(1 for n in naive_top6 if 1<=n<=11)} z2={sum(1 for n in naive_top6 if 12<=n<=22)} z3={sum(1 for n in naive_top6 if 23<=n<=33)}",
                    "和值": sum(naive_top6),
                    "跨度": max(naive_top6) - min(naive_top6)
                },
                "blue_top_3": blue_top3,
                "with_repeat_2026093": sorted(set(naive_top6) & set(last_red)),
                "note": "修复版真实回测: Naive_20 + 3.0*R = 1.09/6, 跟随机基线完全一致 (修复前虚高 4.35)"
            },
            "V14_方案3_Top18_复式": {
                "method": "FSW.SSQ1.F Top 18 (复式投注)",
                "red_top_12_drag": fsw_top12,
                "red_top_15_full": fsw_top15,
                "red_top_18_full": fsw_top18,
                "blue_top_3": blue_top3,
                "compound_counts": {
                    "Top 12 (924 注)": len(fsw_top12),
                    "Top 15 (5005 注)": len(fsw_top15),
                    "Top 18 (18564 注)": len(fsw_top18)
                },
                "expected_hit_rate_489_periods_corrected": {
                    "Top 12": {"avg_hits": "2.28/12", "ge3_rate": "40.3%", "ge5_rate": "1.2%"},
                    "Top 15": {"avg_hits": "2.88/15", "ge3_rate": "62.0%", "ge5_rate": "6.3%"},
                    "Top 18": {"avg_hits": "3.39/18", "ge3_rate": "80.4%", "ge5_rate": "16.0%"}
                },
                "note": "修复版真实 Top 18 ≥5 中率 16.0% (修复前虚高 99.4%)"
            }
        },
        "recommendations": {
            "保守 (6+1 单式)": {
                "红球": fsw_top6,
                "蓝球": blue_top3[0],
                "依据": "FSW Top 6 是 Top 6 单式最佳 (1.17/6), 但也只是略超随机基线 (1.09)"
            },
            "次保守 (Naive+重号)": {
                "红球": naive_top6,
                "蓝球": blue_top3[0],
                "依据": "修复版 Naive_20+3.0*R = 1.09/6 (跟随机一致), 不推荐"
            },
            "复式 (12+1)": {
                "红球": fsw_top12,
                "蓝球": blue_top3[0],
                "注数": 924,
                "依据": "真实 ≥3中率 40.3%, ≥5中率 1.2%"
            },
            "大复式 (18+1)": {
                "红球": fsw_top18,
                "蓝球": blue_top3[0],
                "注数": 18564,
                "依据": "真实 ≥3中率 80.4%, ≥5中率 16.0%"
            },
            "全复式 (18+3)": {
                "红球": fsw_top18,
                "蓝球": blue_top3,
                "注数": 55692,
                "依据": "最大覆盖"
            }
        }
    }


result = make_json()
out_path = ROOT / 'data/backtest/2026094_predictions_ssq_v14_fixed.json'
with open(out_path, 'w', encoding='utf-8') as f:
    json.dump(result, f, ensure_ascii=False, indent=2)
print(f'\n[已保存] {out_path}')
print(f'\n=== 2026094 期 V14 修复版完成 ===')