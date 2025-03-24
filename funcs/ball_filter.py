import streamlit as st
import logging
from collections import defaultdict
from itertools import combinations

def calculate_same_number_counts(data):
    """计算每期红球同号数量"""
    same_number_counts = []
    if len(data) > 1:
        for i in range(1, len(data)):
            previous_row = data.iloc[i - 1]
            current_row = data.iloc[i]
            same_count = 0
            for col in ['红球1', '红球2', '红球3', '红球4', '红球5', '红球6']:
                if current_row[col] in previous_row.values:
                    same_count += 1
            same_number_counts.append(same_count)
    return same_number_counts

def parse_bet(bet_str):
    """解析投注方案字符串，并返回排序后的红球和篮球列表"""
    try:
        red_balls = []
        blue_balls = []
        dantuo = False

        bet_str = bet_str.strip()
        if not bet_str:
            raise ValueError("投注字符串为空")

        if "+" in bet_str:
            parts = bet_str.split("+")
            if len(parts) != 2 or not parts[0]:
                raise ValueError("投注格式错误")
            red_str, blue_str = parts
            red_balls = [int(r) for r in red_str.split(",") if r.strip().isdigit()]
            blue_balls = [int(b) for b in blue_str.split(",") if b.strip().isdigit()]
        else:
            red_balls = [int(r) for r in bet_str.split(",") if r.strip().isdigit()]
            blue_balls = []  # 允许没有蓝球

        # 排序红球和篮球
        red_balls.sort()
        blue_balls.sort()

        # 范围检查和重复号码检查
        red_set = set()
        for ball in red_balls:
            if ball < 1 or ball > 33:
                raise ValueError("红球必须在 1-33 之间")
            if ball in red_set:
                raise ValueError(f"红球 {ball} 重复")
            red_set.add(ball)

        blue_set = set()
        for ball in blue_balls:
            if ball < 1 or ball > 16:
                raise ValueError("篮球必须在 1-16 之间")
            if ball in blue_set:
                raise ValueError(f"篮球 {ball} 重复")
            blue_set.add(ball)

        return red_balls, blue_balls, dantuo

    except ValueError as e:
        logging.error(f"投注解析错误: {e}, 输入: {bet_str}")
        return [], [], False

def convert_to_single_bets(red_balls, blue_balls):
    """将复式和胆拖投注转换为单注"""
    try:
        single_bets = []

        # 复式投注
        if len(red_balls) > 6:
            import itertools
            red_combinations = list(itertools.combinations(red_balls, 6))
            for red_comb in red_combinations:
                if blue_balls:
                    for blue in blue_balls:
                        single_bets.append((list(red_comb), [blue]))
                else:
                    single_bets.append((list(red_comb), []))
        # 单注
        else:
            if blue_balls:
                for blue in blue_balls:
                    single_bets.append((red_balls, [blue]))
            else:
                single_bets.append((red_balls, []))

        return single_bets

    except Exception as e:
        logging.error(f"单注转换错误: {e}, Red: {red_balls}, Blue: {blue_balls}")
        return []


def convert_bets(bets):
    """返回格式：(复式列表, 胆拖列表, 单式列表)，胆拖格式为 胆码#拖码"""
    bet_lists = [tuple(sorted(map(int, bet.split(',')))) for bet in bets]

    complex_bets = []
    dantuo_bets = []  # 格式示例：["1,5#6,9,11", ...]
    used = set()

    # ===== 胆拖处理 =====
    dantuo_candidates = defaultdict(list)
    for bet in bet_lists:
        for dan_len in range(2, 5):
            for dan in combinations(bet, dan_len):
                tuo = sorted(set(bet) - set(dan))
                if len(tuo) >= (6 - len(dan)):
                    dantuo_candidates[dan].append(tuo)

    # 按覆盖能力排序候选
    sorted_dans = sorted(dantuo_candidates.keys(),
                         key=lambda k: (-len(k), -sum(len(t) for t in dantuo_candidates[k])))

    for dan in sorted_dans:
        all_tuo = sorted({num for tuo in dantuo_candidates[dan] for num in tuo})
        cover = [bet for bet in bet_lists
                 if bet not in used
                 and set(dan).issubset(bet)
                 and all(num in all_tuo for num in bet if num not in dan)]

        if len(cover) > 1:
            # 格式化为 胆码#拖码
            dan_str = ",".join(map(str, dan))
            tuo_str = ",".join(map(str, all_tuo))
            dantuo_bets.append(f"{dan_str}#{tuo_str}")
            used.update(cover)

    # ===== 复式处理 =====
    remaining = [bet for bet in bet_lists if bet not in used]
    num_groups = defaultdict(list)

    for bet in remaining:
        for i in range(4, 7):
            for combo in combinations(bet, i):
                num_groups[combo].append(bet)

    for combo in sorted(num_groups, key=lambda x: (-len(num_groups[x]), len(x))):
        covered = [b for b in num_groups[combo] if b in remaining]
        if len(covered) < 2: continue

        all_nums = sorted({n for b in covered for n in b})
        if 6 < len(all_nums) <= 8:
            complex_bets.append(",".join(map(str, all_nums)))
            remaining = [b for b in remaining if b not in covered]
            used.update(covered)

    # ===== 单式处理 =====
    single_bets = [",".join(map(str, bet)) for bet in bet_lists if bet not in used]

    return complex_bets, dantuo_bets, single_bets