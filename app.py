import streamlit as st
import pandas as pd
import openpyxl
import numpy as np
import io
import itertools  # 导入 itertools 库用于组合计算

def process_lottery_data(df, lottery_type):
    """根据彩票类型处理数据，将号码列转换为列表"""
    if lottery_type == "双色球":
        df['前区号码'] = df['前区号码'].apply(lambda x: [int(num) for num in str(x).split()])
        df['后区号码'] = df['后区号码'].apply(lambda x: [int(num) for num in str(x).split()])
    elif lottery_type == "快乐8":
        df['前区号码'] = df['前区号码'].apply(lambda x: [int(num) for num in str(x).split()])
    return df

def calculate_number_frequency(df, lottery_type, num_type="前区"):
    """计算号码出现次数和平均出现频率，可以指定计算前区或后区号码"""
    all_numbers = []
    num_range = range(1, 1)  # 默认空范围
    number_column = '前区号码' # 默认前区号码

    if lottery_type == "双色球":
        if num_type == "前区":
            for _, row in df.iterrows():
                all_numbers.extend(row['前区号码'])
            num_range = range(1, 34)
        elif num_type == "后区":
            for _, row in df.iterrows():
                all_numbers.extend(row['后区号码'])
            num_range = range(1, 17) # 后区号码范围是1-16，但range需要stop值+1
    elif lottery_type == "快乐8":
        if num_type == "前区":
            for _, row in df.iterrows():
                all_numbers.extend(row['前区号码'])
            num_range = range(1, 81)

    if not num_range.start < num_range.stop: # 检查是否是空范围
        return pd.DataFrame() # 返回空DataFrame

    frequency = pd.Series(all_numbers).value_counts().sort_index()
    total_draws = len(df) * (6 if lottery_type == "双色球" and num_type == "前区" else
                             (1 if lottery_type == "双色球" and num_type == "后区" else 20))  # 双色球前区6，后区1，快乐8前区20
    frequency_df = pd.DataFrame(index=['出现次数', '出现频率'], columns=num_range).fillna(0)
    for num, count in frequency.items():
        if num in num_range: # 确保号码在正确的范围内，避免快乐8和双色球号码范围冲突
            frequency_df.loc['出现次数', num] = count
            frequency_df.loc['出现频率', num] = round(count / total_draws, 3) # 保留小数点后3位
    return frequency_df

def calculate_odd_even_ratio(df, lottery_type):
    """计算奇偶占比"""
    odd_counts = []
    even_counts = []
    for _, row in df.iterrows():
        nums = row['前区号码']
        odd_count = sum(1 for num in nums if num % 2 != 0)
        even_count = len(nums) - odd_count
        odd_counts.append(odd_count)
        even_counts.append(even_counts)

    ratio_df = pd.DataFrame({'奇数个数': odd_counts, '偶数个数': even_counts})
    overall_ratio = pd.DataFrame(ratio_df.sum(), columns=['总计']).T
    overall_ratio['奇偶比例'] = overall_ratio['奇数个数'].astype(str) + ":" + overall_ratio['偶数个数'].astype(str)
    return ratio_df, overall_ratio

def calculate_small_large_ratio(df, lottery_type):
    """计算大小占比"""
    small_counts = []
    large_counts = []
    for _, row in df.iterrows():
        nums = row['前区号码']
        if lottery_type == "双色球":
            small_count = sum(1 for num in nums if 1 <= num <= 16)
            large_count = sum(1 for num in nums if 17 <= num <= 33)
        elif lottery_type == "快乐8":
            small_count = sum(1 for num in nums if 1 <= num <= 40)
            large_count = sum(1 for num in nums if 41 <= num <= 80)
        small_counts.append(small_count)
        large_counts.append(large_counts)

    ratio_df = pd.DataFrame({'小号个数': small_counts, '大号个数': large_counts})
    overall_ratio = pd.DataFrame(ratio_df.sum(), columns=['总计']).T
    overall_ratio['大小比例'] = overall_ratio['小号个数'].astype(str) + ":" + overall_ratio['大号个数'].astype(str)
    return ratio_df, overall_ratio

def analyze_consecutive_numbers(df):
    """分析连号情况"""
    consecutive_counts = []
    max_consecutive_lengths = []
    has_consecutive_issues = 0

    for _, row in df.iterrows():
        nums = sorted(row['前区号码'])
        consecutive_count = 0
        max_length = 0
        current_length = 1
        has_consecutive = False # 标记当前期是否有连号

        for i in range(len(nums) - 1):
            if nums[i+1] == nums[i] + 1:
                current_length += 1
                has_consecutive = True
            else:
                max_length = max(max_length, current_length)
                current_length = 1
        max_length = max(max_length, current_length) # 处理最后一组连号

        consecutive_counts.append(int(has_consecutive)) # 1表示有连号，0表示没有
        max_consecutive_lengths.append(max_length)
        if has_consecutive:
            has_consecutive_issues += 1

    consecutive_df = pd.DataFrame({'是否连号': consecutive_counts, '最长连号数': max_consecutive_lengths})
    overall_analysis = pd.DataFrame({
        '总期数': [len(df)],
        '有连号期数': [has_consecutive_issues],
        '连号出现频率': [f"{round(has_consecutive_issues / len(df) * 100, 2)}%"]
    })
    return consecutive_df, overall_analysis

def analyze_skipped_numbers(df, lottery_type):
    """分析跳号，即未出现的号码"""
    all_drawn_numbers = set()
    num_range = range(1, 1)
    if lottery_type == "双色球":
        num_range = range(1, 34)
    elif lottery_type == "快乐8":
        num_range = range(1, 81)

    if not num_range.start < num_range.stop: # 检查是否是空范围
        return pd.DataFrame() # 返回空DataFrame

    for _, row in df.iterrows():
        all_drawn_numbers.update(row['前区号码'])

    skipped_numbers = sorted(list(set(num_range) - all_drawn_numbers))
    skipped_counts = {num: 0 for num in num_range if num not in all_drawn_numbers} # 跳号出现次数为0
    skipped_frequency = {num: 0 for num in num_range if num not in all_drawn_numbers} # 跳号出现频率为0

    skipped_df = pd.DataFrame({
        '出现次数': pd.Series(skipped_counts),
        '出现频率': pd.Series(skipped_frequency)
    }).fillna(0).sort_index() # 按照号码排序

    return skipped_df

def analyze_tail_numbers(df):
    """分析相同尾号"""
    tail_counts = {tail: 0 for tail in range(10)} # 0-9尾号
    for _, row in df.iterrows():
        nums = row['前区号码']
        tails = [num % 10 for num in nums]
        for tail in tails:
            tail_counts[tail] += 1

    tail_df = pd.DataFrame({'尾号出现次数': pd.Series(tail_counts)}).T
    return tail_df

def calculate_blue_ball_frequency(df):
    """计算篮球号码出现次数和频率"""
    return calculate_number_frequency(df, "双色球", "后区") # 复用前区号码频率计算函数

def display_lottery_data(df, lottery_type):
    """显示彩票数据，并进行各种数据分析"""
    st.subheader(":blue[开奖记录]")
    if lottery_type == "双色球":
        st.dataframe(df[['期号', '开奖日期', 'WeekDay', '前区号码', '后区号码']])
    elif lottery_type == "快乐8":
        st.dataframe(df[['期号', '开奖日期', '前区号码']], use_container_width=True)

    # 1. 冷热号分析
    with st.expander(":blue[冷热号分析]", expanded=True):
        st.subheader("前区号码冷热号统计")
        frequency_df_front = calculate_number_frequency(df, lottery_type, "前区")
        if not frequency_df_front.empty: # 检查DataFrame是否为空
            st.dataframe(frequency_df_front, use_container_width=True, column_config={col: st.column_config.NumberColumn(width="small") for col in frequency_df_front.columns})
            st.bar_chart(frequency_df_front.loc['出现次数'])
        else:
            st.write("所选期数内，前区号码数据为空，无法进行冷热号分析。")

        if lottery_type == "双色球": # 快乐8没有后区号码
            st.subheader("后区号码冷热号统计")
            frequency_df_rear = calculate_blue_ball_frequency(df)
            if not frequency_df_rear.empty: # 检查DataFrame是否为空
                st.dataframe(frequency_df_rear, use_container_width=True, column_config={col: st.column_config.NumberColumn(width="small") for col in frequency_df_rear.columns})
                st.bar_chart(frequency_df_rear.loc['出现次数'])
            else:
                st.write("所选期数内，后区号码数据为空，无法进行冷热号分析。")

    # 2. 奇偶占比分析
    with st.expander(":blue[奇偶占比分析]"):
        st.subheader("奇偶占比统计")
        odd_even_ratio_df, overall_odd_even_ratio = calculate_odd_even_ratio(df, lottery_type)
        st.dataframe(odd_even_ratio_df)
        st.dataframe(overall_odd_even_ratio)
        st.bar_chart(odd_even_ratio_df[['奇数个数', '偶数个数']])

    # 3. 大小占比分析
    with st.expander(":blue[大小占比分析]"):
        st.subheader("大小占比统计")
        small_large_ratio_df, overall_small_large_ratio = calculate_small_large_ratio(df, lottery_type)
        st.dataframe(small_large_ratio_df)
        st.dataframe(overall_small_large_ratio)
        st.bar_chart(small_large_ratio_df[['小号个数', '大号个数']])

    # 4. 连号分析
    with st.expander(":blue[连号分析]"):
        st.subheader("连号统计分析")
        consecutive_df, overall_consecutive_analysis = analyze_consecutive_numbers(df)
        st.dataframe(consecutive_df)
        st.dataframe(overall_consecutive_analysis)
        st.bar_chart(consecutive_df[['最长连号数']])

    # 5. 跳号分析
    with st.expander(":blue[跳号分析]"):
        st.subheader("跳号（未出现号码）统计")
        skipped_df = analyze_skipped_numbers(df, lottery_type)
        if not skipped_df.empty: # 检查DataFrame是否为空，快乐8可能为空
            st.dataframe(skipped_df, use_container_width=True, column_config={col: st.column_config.NumberColumn(width="small") for col in skipped_df.columns})
            st.bar_chart(skipped_df.loc['出现次数'])
        else:
            st.write("所选彩票类型或期数，跳号数据为空，无法进行跳号分析。")

    # 6. 相同尾号分析
    with st.expander(":blue[相同尾号分析]"):
        st.subheader("相同尾号统计")
        tail_df = analyze_tail_numbers(df)
        st.dataframe(tail_df)
        st.bar_chart(tail_df.T) # 转置后绘制柱状图，方便查看尾号分布

def calculate_combinations(red_balls, blue_balls=None):
    """计算组合数量"""
    red_combinations = len(list(itertools.combinations(red_balls, 6))) if len(red_balls) >= 6 else 0
    if blue_balls and len(blue_balls) > 0 and blue_balls[0] != '': # 篮球不为空时计算篮球组合, 并排除空字符串情况
        blue_combinations = len(blue_balls)
        total_combinations = red_combinations * blue_combinations if red_combinations > 0 else 0 # 红球组合为0时，总组合也为0
    else: # 没有篮球或篮球为空时，总组合数等于红球组合数
        total_combinations = red_combinations
    return total_combinations


def main():
    st.title(":orange[Lottery 选号工具]")
    st.markdown('<h5 class="title">彩票数据分析与选号</h5>', unsafe_allow_html=True)

    # 左侧边栏 - 彩票类型选择 和  统一的分析期数选择 (下拉菜单)
    lottery_type_sidebar = st.sidebar.selectbox("选择彩票类型", ["双色球", "快乐8"]) # 侧边栏彩票类型选择
    analysis_periods = st.sidebar.selectbox("选择分析期数", [f"{i}期" for i in range(10, 110, 10)], index=4) # 下拉菜单选择分析期数, 默认50期
    display_issues_sidebar = int(analysis_periods.replace("期", "")) # 提取期数值


    # 创建 Tabs -  Tab 结构调整
    tab_info, tab_usage = st.tabs(["彩票信息", "使用说明"]) # 修改 Tab 标题

    # --- 彩票信息 Tab (第一个 Tab) ---
    with tab_info:
        if lottery_type_sidebar == "双色球":
            st.header(":red[双色球]") # Tab 内添加更醒目的标题

            # 双色球 - 数据分析区域
            st.subheader("数据分析")
            df_ssq = pd.read_excel("双色球开奖情况.xlsx")
            df_ssq['期号'] = df_ssq['期号'].astype(str).str.replace(',', '')
            df_ssq = process_lottery_data(df_ssq, "双色球")
            df_ssq_analysis = df_ssq.head(display_issues_sidebar) # 从侧边栏下拉菜单获取期数
            display_lottery_data(df_ssq_analysis, "双色球") # 传递彩票类型

            # 双色球 - 号码选择区域
            st.subheader("号码选择")
            col1_ssq, col2_ssq = st.columns(2) # 分两列布局
            with col1_ssq:
                st.subheader("红球 (前区号码): 1-33")
                # 使用 Unicode 圆圈数字作为选项，并添加一些颜色区分
                red_ball_options_ssq = [f":red[{i:02d}]" if i % 2 != 0 else f":blue[{i:02d}]" for i in range(1, 34)] # 奇数红色，偶数蓝色
                red_balls_selected_ssq_unicode = st.multiselect("请选择6个红球号码",
                                                                options=red_ball_options_ssq,
                                                                max_selections=6,
                                                                key="ssq_red_balls",
                                                                format_func=lambda option: option) # 使用 format_func 显示原始样式
                # 提取用户选择的实际数字，去除颜色标记和Unicode圆圈
                red_balls_selected_ssq = [int(''.join(filter(str.isdigit, ball))) for ball in red_balls_selected_ssq_unicode]


            with col2_ssq:
                st.subheader("篮球 (后区号码): 1-16 (可选)")
                # 使用 Unicode 圆圈数字作为选项，并添加颜色区分
                blue_ball_options_ssq = [f":green[{i:02d}]" for i in range(1, 17)] # 篮球统一绿色
                blue_balls_selected_ssq_unicode = st.multiselect("请选择1个篮球号码 (可选)",
                                                                 options=blue_ball_options_ssq,
                                                                 max_selections=1,
                                                                 key="ssq_blue_balls",
                                                                 format_func=lambda option: option) # 使用 format_func 显示原始样式
                # 提取用户选择的实际数字
                blue_balls_selected_ssq = [int(''.join(filter(str.isdigit, ball))) for ball in blue_balls_selected_ssq_unicode]


            if len(red_balls_selected_ssq) == 6: # 只有红球选满6个才计算
                combination_count_ssq = calculate_combinations(red_balls_selected_ssq, blue_balls_selected_ssq)
                st.write(f"**选定的红球组合数量为: :red[{combination_count_ssq}] 注**")
            elif len(red_balls_selected_ssq) > 0: # 至少选择了红球，但未满6个
                st.warning("请选择 6 个红球号码以计算组合数量。")

        elif lottery_type_sidebar == "快乐8":
            st.header(":green[快乐8]") # Tab 内添加更醒目的标题

            # 快乐8 - 数据分析区域
            st.subheader("数据分析")
            df_kl8 = pd.read_excel("快乐8开奖情况.xlsx")
            df_kl8['期号'] = df_kl8['期号'].astype(str).str.replace(',', '')
            df_kl8 = process_lottery_data(df_kl8, "快乐8")
            df_kl8_analysis = df_kl8.head(display_issues_sidebar) # 从侧边栏下拉菜单获取期数, 与双色球共用
            display_lottery_data(df_kl8_analysis, "快乐8") # 传递彩票类型

            # 快乐8 - 号码选择区域
            st.subheader("号码选择")
            st.subheader("快乐8 选号: 1-80")
            # 快乐8号码选项 -  使用 Unicode 圆圈数字，颜色区分
            kl8_ball_options_kl8 = [f":orange[{i:02d}]" if i % 3 == 0 else f":violet[{i:02d}]" if i % 3 == 1 else f":blue[{i:02d}]" for i in range(1, 81)] #  颜色循环
            kl8_balls_selected_kl8_unicode = st.multiselect("请选择快乐8号码 (最多20个)",
                                                            options=kl8_ball_options_kl8,
                                                            max_selections=20,
                                                            key="kl8_balls",
                                                            format_func=lambda option: option) # 使用 format_func 显示原始样式
            # 提取用户选择的实际数字
            kl8_balls_selected_kl8 = [int(''.join(filter(str.isdigit, ball))) for ball in kl8_balls_selected_kl8_unicode]


            combination_count_kl8 = calculate_combinations(kl8_balls_selected_kl8) # 快乐8没有篮球，蓝球参数为None
            st.write(f"**选定的快乐8组合数量为: :red[{combination_count_kl8}] 注**")

    # --- 使用说明 Tab (第二个 Tab) ---
    with tab_usage:
        st.subheader(":blue[使用说明]")
        st.markdown("**:orange[彩票信息]** Tab:")
        st.markdown("  *  在左侧边栏选择 **彩票类型** 和 **分析期数**，本 Tab 页将显示对应彩票的开奖记录和各项数据分析结果。")
        st.markdown("  *  **数据分析模块：**")
        st.markdown("    *  **冷热号分析:**  统计号码出现次数和频率，辅助判断冷热号码。")
        st.markdown("    *  **奇偶占比分析:**  分析开奖号码中奇偶数的比例。")
        st.markdown("    *  **大小占比分析:**  分析开奖号码中大小号的比例。")
        st.markdown("    *  **连号分析:**  分析连号出现的情况，包括是否连号和最长连号数。")
        st.markdown("    *  **跳号分析:**  统计在选定期数内未出现的号码（跳号）。")
        st.markdown("    *  **相同尾号分析:**  分析开奖号码中相同尾号的出现次数。")
        if lottery_type_sidebar == "双色球": # 仅双色球显示篮球冷热号分析说明, 使用侧边栏的彩票类型判断
            st.markdown("    *  **篮球冷热号分析:**  （仅双色球）统计篮球号码出现次数和频率。")

        st.markdown("\n  **号码选择区域：**")
        st.markdown("    *  **双色球：** 必须选择 **6个红球**，篮球为可选。")
        st.markdown("    *  **快乐8：** 最多可以选择 **20个号码**。")
        st.markdown("    *  选号后，下方会显示您所选号码的组合数量。")

    # 侧边栏下载按钮保留，不再区分彩票类型，直接提供两个彩票类型的数据下载
    st.sidebar.subheader("下载数据")
    with open("双色球开奖情况.xlsx", "rb") as f:
        bytes_data_ssq = f.read()
    st.sidebar.download_button(label="下载双色球数据", data=bytes_data_ssq, file_name="双色球开奖情况.xlsx", mime="application/vnd.ms-excel")

    with open("快乐8开奖情况.xlsx", "rb") as f:
        bytes_data_kl8 = f.read()
    st.sidebar.download_button(label="下载快乐8数据", data=bytes_data_kl8, file_name="快乐8开奖情况.xlsx", mime="application/vnd.ms-excel")


if __name__ == "__main__":
    main()