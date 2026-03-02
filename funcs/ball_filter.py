import streamlit as st
import logging
from collections import defaultdict, Counter
import itertools
from config import LOTTERY_CONFIG

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
    """解析双色球投注方案字符串"""
    try:
        red_dan, red_tuo, blue_dan, blue_tuo = [], [], [], []
        bet_str = bet_str.strip()
        if not bet_str: return [], [], [], []
        parts = bet_str.split("+")
        red_str = parts[0]
        blue_str = parts[1] if len(parts) == 2 else ""

        if "#" in red_str:
            dan_str, tuo_str = red_str.split("#")
            red_dan = [int(r) for r in dan_str.split(",") if r.strip().isdigit()]
            red_tuo = [int(r) for r in tuo_str.split(",") if r.strip().isdigit()]
        else:
            red_tuo = [int(r) for r in red_str.split(",") if r.strip().isdigit()]

        if blue_str:
            if "#" in blue_str:
                dan_str, tuo_str = blue_str.split("#")
                blue_dan = [int(b) for b in dan_str.split(",") if b.strip().isdigit()]
                blue_tuo = [int(b) for b in tuo_str.split(",") if b.strip().isdigit()]
            else:
                blue_tuo = [int(b) for b in blue_str.split(",") if b.strip().isdigit()]
        return sorted(red_dan), sorted(red_tuo), sorted(blue_dan), sorted(blue_tuo)
    except Exception as e:
        logging.error(f"解析错误: {e}")
        return [], [], [], []

def convert_to_single_bets(red_dan, red_tuo, blue_dan, blue_tuo):
    """单注转换"""
    try:
        single_bets = []
        if red_dan:
            for tuo_comb in itertools.combinations(red_tuo, 6 - len(red_dan)):
                red_comb = sorted(red_dan + list(tuo_comb))
                if blue_tuo:
                    for blue in blue_tuo: single_bets.append((red_comb, [blue]))
                else: single_bets.append((red_comb, []))
        elif len(red_tuo) >= 6:
            for red_comb in itertools.combinations(red_tuo, 6):
                if blue_tuo:
                    for blue in blue_tuo: single_bets.append((list(red_comb), [blue]))
                else: single_bets.append((list(red_comb), []))
        else:
            if blue_tuo:
                for blue in blue_tuo: single_bets.append((red_tuo, [blue]))
            else:
                single_bets.append((red_tuo, []))
        return single_bets
    except Exception as e:
        logging.error(f"转换错误: {e}")
        return []

def calculate_morphology_score(numbers, last_period_numbers, lottery_name="双色球"):
    """
    通用形态学评分系统 (V3.1)
    """
    config = LOTTERY_CONFIG.get(lottery_name)
    if not config or "morphology_rules" not in config:
        rules = {
            "sum_range": (80, 130),
            "span_range": (20, 30),
            "ideal_repeats": (0, 2),
            "ideal_consecutive": [0, 1],
            "ideal_odd_counts": [2, 3, 4],
            "ideal_same_tails": [1, 2]
        }
    else:
        rules = config["morphology_rules"]

    score, sorted_nums = 0, sorted(numbers)
    
    # 1. 重号 (Repeats) - 20分
    repeats = len(set(numbers) & set(last_period_numbers))
    r_low, r_high = rules.get("ideal_repeats", (0, 2))
    if r_low <= repeats <= r_high: score += 20
    elif any(abs(repeats - bound) <= 1 for bound in [r_low, r_high]): score += 5
    
    # 2. 连号 (Consecutive) - 15分
    cons = 0
    for i in range(len(sorted_nums)-1):
        if sorted_nums[i+1] - sorted_nums[i] == 1: cons += 1
    ideal_cons = rules.get("ideal_consecutive", [0, 1])
    if cons in (ideal_cons if isinstance(ideal_cons, list) else [ideal_cons]): score += 15
    elif any(abs(cons - ic) <= 1 for ic in (ideal_cons if isinstance(ideal_cons, list) else [ideal_cons])): score += 5
    
    # 3. 奇偶比 (Odd-Even) - 20分
    odds = len([n for n in numbers if n % 2 != 0])
    ideal_odds = rules.get("ideal_odd_counts", [2, 3, 4])
    if odds in ideal_odds: score += 20
    elif any(abs(odds - io) <= 1 for io in ideal_odds): score += 5
        
    # 4. 同尾 (Same Tail) - 15分
    tails = [n % 10 for n in numbers]
    same_tails = len([v for v in Counter(tails).values() if v >= 2])
    ideal_tails = rules.get("ideal_same_tails", [1, 2])
    if same_tails in ideal_tails: score += 15
    elif any(abs(same_tails - it) <= 1 for it in ideal_tails): score += 5
        
    # 5. 跨度 (Span) - 15分
    span = sorted_nums[-1] - sorted_nums[0]
    s_low, s_high = rules.get("span_range", (20, 30))
    if s_low <= span <= s_high: score += 15
    elif any(abs(span - bound) <= 2 for bound in [s_low, s_high]): score += 5

    # 6. 和值 (Sum) - 15分
    s_val = sum(numbers)
    sum_low, sum_high = rules.get("sum_range", (80, 130))
    if sum_low <= s_val <= sum_high: score += 15
    elif any(abs(s_val - bound) <= 10 for bound in [sum_low, sum_high]): score += 5
        
    return score

def get_morphology_report(numbers, last_period_numbers, lottery_name="双色球"):
    sorted_nums = sorted(numbers)
    repeats = sorted(list(set(numbers) & set(last_period_numbers)))
    cons = []
    for i in range(len(sorted_nums)-1):
        if sorted_nums[i+1] - sorted_nums[i] == 1: cons.append(f"{sorted_nums[i]:02d}-{sorted_nums[i+1]:02d}")
    odds = len([n for n in numbers if n % 2 != 0])
    tails = [n % 10 for n in numbers]
    tail_counts = {t: tails.count(t) for t in set(tails)}
    same_tails = [f"{k}尾x{v}" for k, v in tail_counts.items() if v >= 2]
    span = sorted_nums[-1] - sorted_nums[0]
    s_val = sum(numbers)
    score = calculate_morphology_score(numbers, last_period_numbers, lottery_name)
    
    report = f"形态学分析报告 ({lottery_name}) [{', '.join([f'{x:02d}' for x in sorted_nums])}]:\n"
    report += f"- 重号: {len(repeats)} 个 ({', '.join([f'{x:02d}' for x in repeats])})\n"
    report += f"- 连号: {len(cons)} 组 ({', '.join(cons)})\n"
    report += f"- 奇偶比: {odds}:{len(numbers)-odds}\n"
    report += f"- 同尾号: {len(same_tails)} 组 ({', '.join(same_tails)})\n"
    report += f"- 跨度: {span}\n"
    report += f"- 和值: {s_val}\n"
    report += f"** 最终评分: {score}/100 **"
    return report

def convert_bets(bets):
    bet_lists = [tuple(sorted(map(int, bet.split(',')))) for bet in bets]
    complex_bets, dantuo_bets, used = [], [], set()
    dantuo_candidates = defaultdict(list)
    for bet in bet_lists:
        for dan_len in range(2, 5):
            for dan in combinations(bet, dan_len):
                tuo = sorted(set(bet) - set(dan))
                if len(tuo) >= (6 - len(dan)): dantuo_candidates[dan].append(tuo)
    sorted_dans = sorted(dantuo_candidates.keys(), key=lambda k: (-len(k), -len(dantuo_candidates[k])))
    for dan in sorted_dans:
        all_tuo = sorted({num for tuo in dantuo_candidates[dan] for num in tuo})
        cover = [bet for bet in bet_lists if bet not in used and set(dan).issubset(bet)]
        if len(cover) > 1:
            dantuo_bets.append(f"{','.join(map(str, dan))}#{','.join(map(str, all_tuo))}")
            used.update(cover)
    remaining = [bet for bet in bet_lists if bet not in used]
    single_bets = [",".join(map(str, bet)) for bet in remaining]
    return complex_bets, dantuo_bets, single_bets

def convert_and_display():
    if 'filtered_results' in st.session_state and st.session_state.filtered_results:
        bets = st.session_state.filtered_results
        cb, db, sb = convert_bets(bets)
        st.session_state.simplified_bets_area = f"胆拖：\n" + "\n".join(db) + "\n\n单注：\n" + "\n".join(sb)
    else:
        st.session_state.simplified_bets_area = "没有可转化的投注结果"

def check_winning(bet_str, winning_red_balls, winning_blue_ball, winning_amounts):
    try:
        parts = bet_str.split("+")
        red_balls = sorted(map(int, parts[0].split(",")))
        blue_balls = [int(parts[1])] if len(parts) > 1 else []
        red_match = len(set(red_balls) & set(winning_red_balls))
        blue_match = 1 if blue_balls and blue_balls[0] == winning_blue_ball else 0
        if red_match == 6 and blue_match == 1: return "一等奖", winning_amounts["一等奖奖金"]
        if red_match == 6: return "二等奖", winning_amounts["二等奖奖金"]
        if red_match == 5 and blue_match == 1: return "三等奖", 3000
        if red_match == 5 or (red_match == 4 and blue_match == 1): return "四等奖", 200
        if red_match == 4 or (red_match == 3 and blue_match == 1): return "五等奖", 10
        if blue_match == 1: return "六等奖", 5
        return "未中奖", 0
    except: return "格式错误", 0

def analyze_winning():
    analysis_bets = st.session_state.analysis_results
    winning_counts = defaultdict(int)
    total_winning_amount = 0
    try:
        selected_result = st.session_state.lottery_results.iloc[0]
        win_red = sorted([selected_result[f'红球{i}'] for i in range(1, 7)])
        win_blue = selected_result['蓝球']
        win_amounts = {"一等奖奖金": selected_result["一等奖奖金"], "二等奖奖金": selected_result["二等奖奖金"]}
        for line in analysis_bets:
            if isinstance(line, str) and line.strip():
                level, amt = check_winning(line.strip(), win_red, win_blue, win_amounts)
                winning_counts[level] += 1
                total_winning_amount += amt
        table_data = [{"奖项": l, "中奖数量": winning_counts[l]} for l in ["一等奖", "二等奖", "三等奖", "四等奖", "五等奖", "六等奖"]]
        st.session_state.winning_table_data = table_data
        st.session_state.winning_total_amount = total_winning_amount
    except Exception as e:
        st.error(f"分析失败: {e}")
