import streamlit as st
import logging
from collections import defaultdict
from itertools import combinations
import itertools

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
    """解析双色球投注方案字符串，并返回红球胆码、红球拖码、篮球胆码、篮球拖码"""
    try:
        red_dan = []
        red_tuo = []
        blue_dan = []
        blue_tuo = []

        bet_str = bet_str.strip()
        if not bet_str:
            raise ValueError("投注字符串为空")

        parts = bet_str.split("+")
        if len(parts) > 2:
            raise ValueError("投注格式错误：多个篮球部分")

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

        # 排序红球和篮球
        red_dan.sort()
        red_tuo.sort()
        blue_dan.sort()
        blue_tuo.sort()

        # 范围检查和重复号码检查
        red_balls = red_dan + red_tuo
        red_set = set()
        for ball in red_balls:
            if ball < 1 or ball > 33:
                raise ValueError("红球必须在 1-33 之间")
            if ball in red_set:
                raise ValueError(f"红球 {ball} 重复")
            red_set.add(ball)

        blue_balls = blue_dan + blue_tuo
        blue_set = set()
        for ball in blue_balls:
            if ball < 1 or ball > 16:
                raise ValueError("篮球必须在 1-16 之间")
            if ball in blue_set:
                raise ValueError(f"篮球 {ball} 重复")
            blue_set.add(ball)

        return red_dan, red_tuo, blue_dan, blue_tuo

    except ValueError as e:
        logging.error(f"投注解析错误: {e}, 输入: {bet_str}")
        return [], [], [], []

def convert_to_single_bets(red_dan, red_tuo, blue_dan, blue_tuo):
    """将红球和篮球的胆码和拖码转换为单注"""
    try:
        single_bets = []

        # 处理红球
        if red_dan:  # 胆拖
            if not (1 <= len(red_dan) <= 5) or len(red_dan) + len(red_tuo) < 6:
                raise ValueError("红球胆拖格式错误: 胆码数量必须为 1-5，且胆码+拖码不少于 6 个")

            for tuo_comb in itertools.combinations(red_tuo, 6 - len(red_dan)):
                red_comb = sorted(red_dan + list(tuo_comb))

                # 处理篮球
                if blue_dan:
                    for blue_comb in itertools.product(blue_dan, blue_tuo or [None]):
                        blue_final = [b for b in blue_comb if b is not None]
                        single_bets.append((red_comb, blue_final))
                elif blue_tuo:
                    for blue in blue_tuo:
                        single_bets.append((red_comb, [blue]))
                else:
                    single_bets.append((red_comb, []))

        elif len(red_tuo) > 6:  # 复式
            for red_comb in itertools.combinations(red_tuo, 6):
                if blue_dan:
                    for blue_comb in itertools.product(blue_dan, blue_tuo or [None]):
                        blue_final = [b for b in blue_comb if b is not None]
                        single_bets.append((list(red_comb), blue_final))
                elif blue_tuo:
                    for blue in blue_tuo:
                        single_bets.append((list(red_comb), [blue]))
                else:
                    single_bets.append((list(red_comb), []))
        else:  # 单注
            if blue_dan:
                for blue_comb in itertools.product(blue_dan, blue_tuo or [None]):
                    blue_final = [b for b in blue_comb if b is not None]
                    single_bets.append((red_tuo, blue_final))
            elif blue_tuo:
                for blue in blue_tuo:
                    single_bets.append((red_tuo, [blue]))
            else:
                single_bets.append((red_tuo, []))

        return single_bets

    except Exception as e:
        logging.error(f"❌ 单注转换错误: {e}, Red Dan: {red_dan}, Red Tuo: {red_tuo}, Blue Dan: {blue_dan}, Blue Tuo: {blue_tuo}")
        return []


#简化函数暂时不用
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

    return complex_bets, dantuo_bets, single_bets #sh #

def convert_and_display():
    """转换投注号码并显示结果"""
    if 'filtered_results' in st.session_state and st.session_state.filtered_results:
        bets = st.session_state.filtered_results
        complex_bets, dantuo_bets, single_bets = convert_bets(bets)

        result_str = "复式：\n" + "\n".join(complex_bets) + "\n\n"
        result_str += "胆拖：\n" + "\n".join(dantuo_bets) + "\n\n"
        result_str += "单注：\n" + "\n".join(single_bets)

        st.session_state.simplified_bets_area = result_str
    else:
        st.session_state.simplified_bets_area = "没有可转化的投注结果"

def check_winning(bet_str, winning_red_balls, winning_blue_ball):
    """计算双色球单注的中奖情况"""
    try:
        parts = bet_str.split("+")
        red_balls = sorted(map(int, parts[0].split(",")))
        blue_balls = [int(parts[1])] if len(parts) > 1 else []

        red_match = len(set(red_balls) & set(winning_red_balls))
        blue_match = 1 if blue_balls and blue_balls[0] == winning_blue_ball else 0

        if red_match == 6 and blue_match == 1:
            return "一等奖", 0  # 奖金稍后计算
        elif red_match == 6:
            return "二等奖", 0  # 奖金稍后计算
        elif red_match == 5 and blue_match == 1:
            return "三等奖", 3000
        elif red_match == 5 or (red_match == 4 and blue_match == 1):
            return "四等奖", 200
        elif red_match == 4 or (red_match == 3 and blue_match == 1):
            return "五等奖", 10
        elif blue_match == 1:
            return "六等奖", 5
        else:
            return "未中奖", 0
    except Exception as e:
        st.error(f"投注格式错误：{e}, 投注：{bet_str}")
        return "格式错误", 0

def analyze_winning():
    """分析双色球中奖情况"""
    bets_text = st.session_state.bets_text
    analysis_results = []
    total_bets = 0
    winning_counts = {
        "一等奖": 0,
        "二等奖": 0,
        "三等奖": 0,
        "四等奖": 0,
        "五等奖": 0,
        "六等奖": 0,
        "未中奖": 0,
        "格式错误": 0,
    }
    winning_amounts = {
        "一等奖": 0,
        "二等奖": 0,
        "三等奖": 0,
        "四等奖": 0,
        "五等奖": 0,
        "六等奖": 0,
        "未中奖": 0,
        "格式错误": 0
    }

    try:
        # 创建下拉菜单，显示最近 10 期开奖记录
        issue_numbers = st.session_state.lottery_results['期号'].astype(str).tolist()
        selected_issue = st.selectbox("选择开奖期号:", issue_numbers, index=len(issue_numbers) - 1)

        # 根据选择的期号，获取开奖结果
        selected_result = st.session_state.lottery_results[st.session_state.lottery_results['期号'].astype(str) == selected_issue].iloc[0]

        # 从 DataFrame 中提取开奖号码
        winning_red_balls = sorted([
            selected_result['红球1'], selected_result['红球2'], selected_result['红球3'],
            selected_result['红球4'], selected_result['红球5'], selected_result['红球6']
        ])
        winning_blue_ball = selected_result['蓝球']

        for line in bets_text.splitlines():
            if line.strip():
                winning_level, winning_amount = check_winning(line.strip(), winning_red_balls, winning_blue_ball)
                winning_counts[winning_level] += 1
                winning_amounts[winning_level] += winning_amount
                total_bets += 1

                analysis_results.append(f"{line.strip()} ({winning_level})")

        # 创建表格数据
        table_data = []
        total_winning_amount = 0
        for level in winning_counts:
            count = winning_counts[level]
            amount = winning_amounts[level]
            table_data.append({"奖项": level, "中奖数量": count, "中奖金额": amount, "奖金合记": count * amount})
            total_winning_amount += count * amount

        # 使用 session_state 将表格数据传递给 col2
        st.session_state.winning_table_data = table_data
        st.session_state.winning_total_bets = total_bets
        st.session_state.winning_total_amount = total_winning_amount

        # 将结果存储在 session_state 中
        st.session_state.all_bets_text = f"总投注数: {total_bets}\n" + "\n".join(analysis_results)
        st.session_state.analysis_results = analysis_results

    except KeyError as e:
        st.error(f"键名错误：{e}。请检查开奖记录数据。")
    except TypeError as e:
        st.error(f"数据类型错误：{e}。请检查开奖记录数据格式。")