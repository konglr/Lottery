import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from datetime import datetime
import itertools
import altair as alt

# Page configuration
st.set_page_config(
    page_title="双色球分析与选号工具",
    layout="wide",
    initial_sidebar_state="expanded"
)
# 设置 matplotlib 使用 SimHei 字体
plt.rcParams['font.sans-serif'] = ['Hei']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示为方块的问题

# CSS styling
st.markdown("""
<style>
    .main {
        padding: 1rem;
    }
    .lottery-ball {
        display: inline-block;
        width: 40px;
        height: 40px;
        border-radius: 50%;
        text-align: center;
        line-height: 40px;
        margin: 5px;
        font-weight: bold;
        cursor: pointer;
        user-select: none; /* Prevent text selection during click */
    }
    .red-ball {
        background-color: #ff5b5b;
        color: white;
    }
    .blue-ball {
        background-color: #5b9fff;
        color: white;
    }
    .selected {
        border: 3px solid #ffcc00;
    }
    .latest-draw {
        padding: 10px;
        background-color: #f8f9fa;
        border-radius: 5px;
        margin-bottom: 15px;
    }
    .header {
        font-size: 24px;
        font-weight: bold;
        margin-bottom: 15px;
    }
    .subheader {
        font-size: 14px;
        font-weight: bold;
        margin: 10px 0;
    }
    .selected-numbers-display {
        margin-top: 20px;
        padding: 10px;
        border: 1px solid #ced4da;
        border-radius: 5px;
    }
 
    .red-ball-button {
        display: inline-block; /* Make it behave like lottery balls */
        width: 40px; /* Same width as lottery balls */
        height: 40px; /* Same height as lottery balls */
        border-radius: 50%; /* Make it round */
        background-color: #007bff;
        color: white;
        border: none;
        cursor: pointer;
        font-size: 10px;
        margin: 5px; /* Same margin as lottery balls */
        line-height: 40px; /* Center text vertically */
        text-align: center;
    }
    .clear-button:hover {
        background-color: #0056b3;
    }

    .selected-button { /* Add this class for selected buttons */
        background-color: gray;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.title("双色球分析选项")

    # Lottery type selection
    lottery_type = st.selectbox(
        "彩票类型",
        ["双色球"]
    )

    # Analysis period selection
    analysis_period = st.slider(
        "分析期数",
        min_value=10,
        max_value=100,
        value=20,
        step=10
    )

    st.divider()

    # Filter settings (These filters are currently only UI elements and not yet applied to data analysis or number selection logic)
    st.subheader("筛选条件设置 (暂未应用)") # Added clarification that filters are not yet functional

    odd_even_filter = st.checkbox("奇偶比例筛选")
    if odd_even_filter:
        odd_count = st.slider("红球奇数个数", 0, 6, (2, 4))

    consecutive_filter = st.checkbox("连号筛选")
    if consecutive_filter:
        consecutive_count = st.slider("最多连号数量", 0, 6, 2)

    same_tail_filter = st.checkbox("同尾号筛选")
    if same_tail_filter:
        max_same_tail = st.slider("最多同尾号数量", 0, 6, 2)

    sum_filter = st.checkbox("和值筛选")
    if sum_filter:
        sum_range = st.slider("红球和值范围", 21, 183, (70, 130))

@st.cache_data
def load_historical_data(analysis_period):
    try:
        # 直接读取需要的列
        df = pd.read_excel('双色球开奖情况.xlsx', usecols=["期号", "开奖日期", "红球1", "红球2", "红球3", "红球4", "红球5", "红球6", "蓝球"])
        if analysis_period > 0:
            df = df.head(analysis_period)

        # 将 "开奖日期" 列重命名为 "日期"
        df = df.rename(columns={"开奖日期": "日期"})

        # 确保所有球号码列为整数类型
        ball_columns = ["红球1", "红球2", "红球3", "红球4", "红球5", "红球6", "蓝球"]
        for col in ball_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)

        return df

    except FileNotFoundError:
        st.error("找不到 Excel 文件 '双色球开奖情况.xlsx'，请确保文件与代码在同一目录下，并检查文件名是否正确。")
        return pd.DataFrame(columns=["期号", "日期", "红球1", "红球2", "红球3", "红球4", "红球5", "红球6", "蓝球"])
    except Exception as e:
        st.error(f"加载或处理数据时出错: {e}")
        return pd.DataFrame(columns=["期号", "日期", "红球1", "红球2", "红球3", "红球4", "红球5", "红球6", "蓝球"])

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

# Main content area
st.markdown("<div class='header'>双色球分析工具</div>", unsafe_allow_html=True)



# Display tabs for different analyses
tab1, tab2, tab3 = st.tabs(["号码分析", "选号工具", "历史数据"])

with tab1:
    # 加载历史数据，并根据分析期数筛选
    filtered_data = load_historical_data(analysis_period)

    if not filtered_data.empty:
        latest_draw = filtered_data.iloc[0]
        st.markdown("<div class='subheader'>最新开奖信息</div>", unsafe_allow_html=True)
        latest_draw_html = f"""
            <div class='latest-draw'>
                <p>期号: {latest_draw.get('期号', 'N/A')} &nbsp;&nbsp;&nbsp; 开奖日期: {latest_draw.get('日期', 'N/A')}</p>
                <div>
            """

        for i in range(1, 7):
            ball_value = latest_draw.get(f'红球{i}', 0)
            latest_draw_html += f"<span class='lottery-ball red-ball'>{ball_value}</span>"

        latest_draw_html += f"<span class='lottery-ball blue-ball'>{latest_draw.get('蓝球', 0)}</span>"
        latest_draw_html += "</div></div>"

        st.markdown(latest_draw_html, unsafe_allow_html=True)
    else:
        st.warning("没有可显示的开奖数据。请检查Excel文件。")

    if not filtered_data.empty:
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("红球冷热分析")
            # Calculate frequency for red balls
            red_frequency = {}
            for i in range(1, 34):  # Corrected range to 1-33 in previous version, now 1-34 to align with UI
                count = 0
                for j in range(1, 7):
                    if f'红球{j}' in filtered_data.columns:
                        count += (filtered_data[f'红球{j}'] == i).sum()
                red_frequency[i] = count

            red_freq_df = pd.DataFrame({'号码': red_frequency.keys(), '出现次数': red_frequency.values()})
            red_freq_df = red_freq_df.sort_values('出现次数', ascending=False)

            # fig, ax = plt.subplots(figsize=(10, 5))
            # ax.bar(red_freq_df['号码'].astype(str), red_freq_df['出现次数'], color='red')
            # ax.set_title('红球出现频率')
            # ax.set_xlabel('号码')
            # ax.set_ylabel('出现次数')
            # st.pyplot(fig)

            # 创建 Altair 柱状图
            chart = alt.Chart(red_freq_df).mark_bar(color='red').encode(
                x=alt.X('号码:O', title='红球号码', sort='-y',
                        axis=alt.Axis(labelAngle=-45, labelOverlap=False, labelFontSize=10)),
                # 使用 Ordinal 类型，将号码作为离散值处理
                y=alt.Y('出现次数:Q', title='出现次数'),  # 使用 Quantitative 类型，将出现次数作为数值处理
                tooltip=['号码', '出现次数']  # 添加鼠标悬停提示
            ).properties(
                title='红球出现频率',
                width=800,  # 设置图表宽度
                height=300  # 设置图表高度
            )

            # 在 Streamlit 中显示 Altair 图表
            st.altair_chart(chart, use_container_width=True)

            # 获取出现频率最高的前 5 个号码
            top_5_freq = red_freq_df['出现次数'].iloc[4]
            hot_red_df = red_freq_df[red_freq_df['出现次数'] >= top_5_freq]
            hot_red = hot_red_df['号码'].tolist()

            # 获取出现频率最低的后 5 个号码
            bottom_5_freq = red_freq_df['出现次数'].iloc[-5]
            cold_red_df = red_freq_df[red_freq_df['出现次数'] <= bottom_5_freq]
            cold_red = cold_red_df['号码'].tolist()

            # 格式化输出
            hot_red_str = ', '.join(map(str, hot_red))
            cold_red_str = ', '.join(map(str, cold_red))

            # 显示热门和冷门号码
            st.write(f"热门号码: {hot_red_str}")
            st.write(f"冷门号码: {cold_red_str}")

        with col2:
            st.subheader("蓝球冷热分析")
            # Calculate frequency for blue balls
            if '蓝球' in filtered_data.columns:
                blue_frequency = {}
                for i in range(1, 17):
                    blue_frequency[i] = (filtered_data['蓝球'] == i).sum()

                blue_freq_df = pd.DataFrame(
                    {'号码': list(blue_frequency.keys()), '出现次数': list(blue_frequency.values())})
                blue_freq_df = blue_freq_df.sort_values('出现次数', ascending=False)

                # 创建 Altair 柱状图
                chart = alt.Chart(blue_freq_df).mark_bar(color='blue').encode(
                    x=alt.X('号码:O', title='篮球号码', sort='-y', axis=alt.Axis(labelAngle=0, labelFontSize=10)),
                    y=alt.Y('出现次数:Q', title='出现次数'),
                    tooltip=['号码', '出现次数']
                ).properties(
                    title='蓝球出现频率',
                    width=800,
                    height=300
                )

                # 在 Streamlit 中显示 Altair 图表
                st.altair_chart(chart, use_container_width=True)

                # 获取出现频率最高的前 3 个号码
                top_3_freq = blue_freq_df['出现次数'].iloc[2]
                hot_blue_df = blue_freq_df[blue_freq_df['出现次数'] >= top_3_freq]
                hot_blue = hot_blue_df['号码'].tolist()

                # 获取出现频率最低的后 3 个号码
                bottom_3_freq = blue_freq_df['出现次数'].iloc[-3]
                cold_blue_df = blue_freq_df[blue_freq_df['出现次数'] <= bottom_3_freq]
                cold_blue = cold_blue_df['号码'].tolist()

                # 格式化输出
                hot_blue_str = ', '.join(map(str, hot_blue))
                cold_blue_str = ', '.join(map(str, cold_blue))

                # 显示热门和冷门号码
                st.write(f"热门蓝球号码: {hot_blue_str}")
                st.write(f"冷门蓝球号码: {cold_blue_str}")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("大小比例分析")
            # Calculate big-small ratio for red balls in each draw
            big_small_ratios = []
            for _, row in filtered_data.iterrows():
                big_count = 0
                for i in range(1, 7):
                    col_name = f'红球{i}'
                    if col_name in row and row[col_name] > 16:  # 假设17及以上为大号
                        big_count += 1
                big_small_ratios.append((big_count, 6 - big_count))

            big_small_df = pd.DataFrame(big_small_ratios, columns=['大号', '小号'])
            big_small_counts = big_small_df.groupby(['大号', '小号']).size().reset_index(name='次数')

            fig, ax = plt.subplots(figsize=(10, 5))
            labels = [f"{row['大号']}大{row['小号']}小" for _, row in big_small_counts.iterrows()]
            ax.bar(labels, big_small_counts['次数'])
            ax.set_title('红球大小比例分布')
            ax.set_xlabel('大小比例')
            ax.set_ylabel('出现次数')
            st.pyplot(fig)

        with col2:
            st.subheader("奇偶比例分析")
            # Calculate odd-even ratio for red balls in each draw
            odd_even_counts = {
                '1:5': 0,
                '2:4': 0,
                '3:3': 0,
                '4:2': 0,
                '5:1': 0
            }

            for _, row in filtered_data.iterrows():
                odd_count = 0
                for i in range(1, 7):
                    col_name = f'红球{i}'
                    if col_name in row and row[col_name] % 2 == 1:
                        odd_count += 1

                ratio = f"{odd_count}:{6 - odd_count}"
                if ratio in odd_even_counts:
                    odd_even_counts[ratio] += 1

            ratios = list(odd_even_counts.keys())
            counts = list(odd_even_counts.values())

            fig, ax = plt.subplots(figsize=(10, 5))
            ax.bar(ratios, counts)
            ax.set_title('红球奇偶比例分布')
            ax.set_xlabel('奇偶比例')
            ax.set_ylabel('出现次数')
            st.pyplot(fig)

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("连号分析")
            # Analyze consecutive numbers
            consecutive_counts = []
            for _, row in filtered_data.iterrows():
                red_balls = [row.get(f'红球{i}', 0) for i in range(1, 7)]
                red_balls = [b for b in red_balls if b > 0]  # Filter out zeros or missing values

                if len(red_balls) < 2:
                    continue

                red_balls.sort()
                max_consecutive = 1
                current_consecutive = 1

                for i in range(1, len(red_balls)):
                    if red_balls[i] == red_balls[i - 1] + 1:
                        current_consecutive += 1
                    else:
                        max_consecutive = max(max_consecutive, current_consecutive)
                        current_consecutive = 1

                max_consecutive = max(max_consecutive, current_consecutive)
                consecutive_counts.append(max_consecutive)

            if consecutive_counts:
                consecutive_df = pd.DataFrame({'最大连号数': consecutive_counts})
                consecutive_count = consecutive_df['最大连号数'].value_counts().sort_index()

                fig, ax = plt.subplots(figsize=(8, 4))
                ax.bar(consecutive_count.index.astype(str), consecutive_count.values)
                ax.set_title('最大连号数分布')
                ax.set_xlabel('连号数')
                ax.set_ylabel('出现次数')
                st.pyplot(fig)
            else:
                st.warning("连号分析数据不足")

        with col2:
            st.subheader("同尾号分析")
            # Analyze same tail numbers
            same_tail_counts = []
            for _, row in filtered_data.iterrows():
                red_balls = [row.get(f'红球{i}', 0) for i in range(1, 7)]
                red_balls = [b for b in red_balls if b > 0]  # Filter out zeros or missing values

                if len(red_balls) < 2:
                    continue

                tails = [num % 10 for num in red_balls]
                tail_counts = {}

                for tail in tails:
                    if tail in tail_counts:
                        tail_counts[tail] += 1
                    else:
                        tail_counts[tail] = 1

                if tail_counts:
                    max_same_tail = max(tail_counts.values())
                    same_tail_counts.append(max_same_tail)

            if same_tail_counts:
                same_tail_df = pd.DataFrame({'最大同尾数': same_tail_counts})
                same_tail_count = same_tail_df['最大同尾数'].value_counts().sort_index()

                fig, ax = plt.subplots(figsize=(8, 4))
                ax.bar(same_tail_count.index.astype(str), same_tail_count.values)
                ax.set_title('最大同尾号数分布')
                ax.set_xlabel('同尾号数')
                ax.set_ylabel('出现次数')
                st.pyplot(fig)
            else:
                st.warning("同尾号分析数据不足")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("红球同号统计")  # 由于缺少前一期的数据，统计中少一期数据

            same_number_counts = calculate_same_number_counts(filtered_data)
            # 统计同号数量的频率
            frequency = {}
            for count in same_number_counts:
                frequency[count] = frequency.get(count, 0) + 1

            frequency_df = pd.DataFrame(
                {'同号数量': list(frequency.keys()), '出现次数': list(frequency.values())})

            # 创建 Altair 柱状图
            chart = alt.Chart(frequency_df).mark_bar().encode(
                x=alt.X('同号数量:O', title='同号数量', axis=alt.Axis(labelAngle=0, labelOverlap=False)),
                y=alt.Y('出现次数:Q', title='出现次数'),
                tooltip=['同号数量', '出现次数']
            ).properties(
                title='红球同号数量统计',
                width=800,
                height=300
            )

            # 在 Streamlit 中显示 Altair 图表
            st.altair_chart(chart, use_container_width=True)

        with col2:
            st.subheader("红球同号分析")

            same_number_counts = calculate_same_number_counts(filtered_data)
            # 修改 same_numbers_df 的创建方式，使用 filtered_data 的期号
            same_numbers_df = pd.DataFrame({
                '期号': filtered_data['期号'].tolist()[1:],  # 从第二期开始，获取期号
                '同号数量': same_number_counts
            })

            # 创建 Altair 折线图
            chart = alt.Chart(same_numbers_df).mark_line().encode(
                x=alt.X('期号:O', title='期号', axis=alt.Axis(labelAngle=-45, labelFontSize=10)),  # 使用期号作为 x 轴
                y=alt.Y('同号数量:Q', title='同号数量', axis=alt.Axis(tickCount=same_numbers_df['同号数量'].max() + 1)),
                # 设置 y 轴标尺为整数
                tooltip=['期号', '同号数量']
            ).properties(
                title='红球同号趋势图',
                width=800,
                height=300
            )

            # 在 Streamlit 中显示 Altair 图表
            st.altair_chart(chart, use_container_width=True)


    else:
        st.warning("没有足够的数据进行分析。请检查Excel文件。")


import streamlit as st

with st.container():
    with st.container():
        st.subheader("选号工具")

        # 初始化会话状态
        if 'selected_red_balls' not in st.session_state:
            st.session_state.selected_red_balls = []
        if 'selected_blue_balls' not in st.session_state:
            st.session_state.selected_blue_balls = []
        if 'bets' not in st.session_state:
            st.session_state.bets = []

        # 创建占位符
        placeholder = st.empty()

        def display_selected_numbers():
            selected_numbers_html = "<div class='selected-numbers-display'>"
            selected_numbers_html += "<p>已选红球:</p><div>"
            for ball in sorted(st.session_state.selected_red_balls):
                selected_numbers_html += f"<span class='lottery-ball red-ball selected'>{ball}</span>"
            selected_numbers_html += "</div>"

            selected_numbers_html += "<p>已选蓝球:</p><div>"
            for ball in sorted(st.session_state.selected_blue_balls):
                selected_numbers_html += f"<span class='lottery-ball blue-ball selected'>{ball}</span>"
            selected_numbers_html += "</div>"

            if not st.session_state.selected_red_balls and not st.session_state.selected_blue_balls:
                selected_numbers_html += "<p>尚未选择任何号码</p>"

            selected_numbers_html += "</div>"
            placeholder.markdown(selected_numbers_html, unsafe_allow_html=True)

        display_selected_numbers()

        def toggle_red_ball(ball):
            if ball in st.session_state.selected_red_balls:
                st.session_state.selected_red_balls.remove(ball)
            else:
                st.session_state.selected_red_balls.append(ball)
            display_selected_numbers()

        def toggle_blue_ball(ball):
            if ball in st.session_state.selected_blue_balls:
                st.session_state.selected_blue_balls.remove(ball)
            else:
                st.session_state.selected_blue_balls.append(ball)
            display_selected_numbers()

        def clear_selection():
            st.session_state.selected_red_balls = []
            st.session_state.selected_blue_balls = []
            display_selected_numbers()

        def add_bet():
            if st.session_state.selected_red_balls and st.session_state.selected_blue_balls:
                red_balls_str = ",".join(map(str, sorted(st.session_state.selected_red_balls)))
                blue_balls_str = ",".join(map(str, sorted(st.session_state.selected_blue_balls)))
                bet_str = f"{red_balls_str} + {blue_balls_str}"
                st.session_state.bets.append(bet_str)
            else:
                st.warning("请选择红球和蓝球号码")

        # 红球选择
        st.markdown("<div class='subheader'>选择红球 (1-33)</div>", unsafe_allow_html=True)
        cols_red = st.columns(16)

        for i in range(1, 34):
            col_index = (i - 1) % 16
            with cols_red[col_index]:
                ball_key = f"red_{i}"
                if st.button(f"{i}", key=ball_key, on_click=toggle_red_ball, args=(i,), use_container_width=True):
                    pass

        # 蓝球选择
        st.markdown("<div class='subheader'>选择蓝球 (1-16)</div>", unsafe_allow_html=True)
        cols_blue = st.columns(16)

        for i in range(1, 17):
            col_index = (i - 1) % 16
            with cols_blue[col_index]:
                ball_key = f"blue_{i}"
                if st.button(f"{i}", key=ball_key, on_click=toggle_blue_ball, args=(i,), use_container_width=True):
                    pass

        st.button("清除所有选择", on_click=clear_selection, type="primary", key="clear_all_button")
        st.button("增加一个投注", on_click=add_bet, type="primary", key="add_bet_button")

    with st.container():
        # 显示投注记录
        st.subheader("投注记录")
        if st.session_state.bets:
            for bet in st.session_state.bets:
                st.text(f"复式: {bet}")
        else:
            st.text("尚未添加任何投注")

with tab3:
    st.subheader("历史开奖数据")
    historical_data = load_historical_data(100) # 获取所有数据
    if not historical_data.empty:
        st.dataframe(historical_data, width=1000, height=500) # Display historical data in a table
    else:
        st.warning("没有历史开奖数据可以显示。")