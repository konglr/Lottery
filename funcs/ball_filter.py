import streamlit as st


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
    """解析投注方案字符串"""
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

