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
        font-size: 10px;
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
        max_value=500,
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
        # 读取 Excel 并加载所有需要的列，包括新增的分析字段
        df = pd.read_excel(
            "双色球开奖情况.xlsx",
            usecols=[
                "期号", "开奖日期", "红球1", "红球2", "红球3", "红球4", "红球5", "红球6", "蓝球",
                "奇数", "偶数", "小号", "大号", "一区", "二区", "三区",
                "重号", "邻号", "孤号", "和值", "AC", "跨度",
                "二连", "三连", "四连", "五连", "六连",
                "二跳", "三跳", "四跳", "五跳", "六跳"
            ]
        )

        # 只保留最近 `analysis_period` 期的数据
        if analysis_period > 0:
            df = df.head(analysis_period)

        # 重命名 "开奖日期" 列为 "日期"
        df = df.rename(columns={"开奖日期": "日期"})

        # 确保所有数字列为整数类型
        ball_columns = [
            "红球1", "红球2", "红球3", "红球4", "红球5", "红球6", "蓝球",
            "奇数", "偶数", "小号", "大号", "一区", "二区", "三区",
            "重号", "邻号", "孤号", "和值", "AC", "跨度",
            "二连", "三连", "四连", "五连", "六连",
            "二跳", "三跳", "四跳", "五跳", "六跳"
        ]
        df[ball_columns] = df[ball_columns].apply(pd.to_numeric, errors="coerce").fillna(0).astype(int)

        return df

    except FileNotFoundError:
        st.error("找不到 Excel 文件 '双色球开奖情况.xlsx'，请确保文件在当前目录，并检查文件名是否正确。")
        return pd.DataFrame(columns=[
            "期号", "日期", "红球1", "红球2", "红球3", "红球4", "红球5", "红球6", "蓝球",
            "奇数", "偶数", "小号", "大号", "一区", "二区", "三区",
            "重号", "邻号", "孤号", "和值", "AC", "跨度",
            "二连", "三连", "四连", "五连", "六连",  # 新增的字段
            "二跳", "三跳", "四跳", "五跳", "六跳"  # 新增的字段
        ])
    except Exception as e:
        st.error(f"加载或处理数据时出错: {e}")
        return pd.DataFrame(columns=[
            "期号", "日期", "红球1", "红球2", "红球3", "红球4", "红球5", "红球6", "蓝球",
            "奇数", "偶数", "小号", "大号", "一区", "二区", "三区",
            "重号", "邻号", "孤号", "和值", "AC", "跨度",
            "二连", "三连", "四连", "五连", "六连",  # 新增的字段
            "二跳", "三跳", "四跳", "五跳", "六跳"  # 新增的字段
        ])


# 加载历史数据，并根据分析期数筛选
filtered_data = load_historical_data(analysis_period)
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
# 获取出现频率最高的前 5 个号码
top_5_freq = red_freq_df['出现次数'].iloc[4]
hot_red_df = red_freq_df[red_freq_df['出现次数'] >= top_5_freq]
hot_red = hot_red_df['号码'].tolist()

# 获取出现频率最低的后 5 个号码
bottom_5_freq = red_freq_df['出现次数'].iloc[-5]
cold_red_df = red_freq_df[red_freq_df['出现次数'] <= bottom_5_freq]
cold_red = cold_red_df['号码'].tolist()

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

def analyze_red_balls(red_balls):
    """
    分析红球号码。
    Args:
        red_balls (list): 选中的红球号码列表。
    Returns:
        dict: 包含分析结果的字典。
    """
    if not red_balls:
        return {}  # 如果没有红球，返回空字典
    results = {}

    # 1. 热号冷号数量
    cold_count = sum(1 for ball in red_balls if ball in cold_red)
    hot_count = sum(1 for ball in red_balls if ball in hot_red)
    results['热冷'] = f"{hot_count}:{cold_count}"

    # 2. 奇偶号比例
    odd_count = sum(1 for ball in red_balls if ball % 2 != 0)
    even_count = sum(1 for ball in red_balls if ball % 2 == 0)
    results['奇偶'] = f"{odd_count}:{even_count}"

    # 3. 大小号比例（1-16为小，17-33为大）
    small_count = sum(1 for ball in red_balls if 1 <= ball <= 16)
    large_count = sum(1 for ball in red_balls if 17 <= ball <= 33)
    results['大小'] = f"{large_count}:{small_count}"

    # 4. 连号
    red_balls.sort()
    consecutive_counts = {}
    count = 1
    for i in range(1, len(red_balls)):
        if red_balls[i] == red_balls[i - 1] + 1:
            count += 1
        else:
            if count > 1:
                consecutive_counts[count] = consecutive_counts.get(count, 0) + 1
            count = 1
    if count > 1:
        consecutive_counts[count] = consecutive_counts.get(count, 0) + 1
    results['连号'] = ", ".join([f"{length}连={c}" for length, c in consecutive_counts.items()])

    # 5. 同尾号
    tails = [ball % 10 for ball in red_balls] # 取模操作，取个位数的数字
    tail_counts = {}
    for tail in tails:
        tail_counts[tail] = tail_counts.get(tail, 0) + 1

    same_tail_counts = {}
    for tail, count in tail_counts.items():
        if count > 1:
            same_tail_counts[count] = same_tail_counts.get(count, 0) + 1
    results['同尾'] = ", ".join([f"={count}" for count in same_tail_counts.keys()])

    # 6. 最新一期同号 (修改部分 - 仅替换 #6 部分)
    last_draw_reds = [latest_draw[f'红球{i}'] for i in range(1, 7)]  # 提取 latest_draw 的红球
    if red_balls == last_draw_reds:  # <---  判断 red_balls 是否与 latest_draw_reds 完全相同
        if len(filtered_data) > 1:  # <---  确保 filtered_data 中至少有两期数据，防止 IndexError
            previous_draw = filtered_data.iloc[1]  # 获取前一期数据
            previous_draw_reds = [previous_draw[f'红球{i}'] for i in range(1, 7)]  # 提取前一期红球
            same_count = sum(1 for ball in red_balls if ball in previous_draw_reds)  # 与前一期红球比较
        else:
            same_count = 0  # 如果只有一期或没有数据，则同号数量为 0
    else:
        same_count = sum(1 for ball in red_balls if ball in last_draw_reds)  # 否则，与最新一期红球比较
    results['同号'] = f"={same_count}"

    return results


# Main content area
st.markdown("<div class='header'>双色球分析工具</div>", unsafe_allow_html=True)


# Display tabs for different analyses
tab1, tab2, tab3 = st.tabs(["号码分析", "选号工具", "历史数据"])

with tab1:

    latest_draw = filtered_data.iloc[0]
    st.markdown("<div class='subheader'>最新开奖信息</div>", unsafe_allow_html=True)
    latest_draw_html = f"""
            <div class='latest-draw'>
                <p>期号: {latest_draw.get('期号', 'N/A')} &nbsp;&nbsp;&nbsp; 开奖日期: {latest_draw.get('日期', 'N/A')}</p>
                <div>
            """

    red_balls_latest_draw = []  # Collect red balls for analysis
    for i in range(1, 7):
        ball_value = latest_draw.get(f'红球{i}', 0)
        latest_draw_html += f"<span class='lottery-ball red-ball'>{ball_value}</span>"
        red_balls_latest_draw.append(ball_value)  # Add red ball to list

    latest_draw_html += f"<span class='lottery-ball blue-ball'>{latest_draw.get('蓝球', 0)}</span>"
    latest_draw_html += "</div>"  # 结束 开奖号码 div

    # 调用 analyze_red_balls 函数获取分析结果 HTML
    red_analysis_result_html = analyze_red_balls(red_balls_latest_draw)

    # 将分析结果 HTML 追加到 latest_draw_html
    latest_draw_html += f"""
                <div class='analysis-box'>  <div class='analysis-title'>红球分析:
                    {red_analysis_result_html}  </div>
            </div> """

    st.markdown(latest_draw_html, unsafe_allow_html=True)  # 显示完整 HTML 代码，包含分析结果


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

        # 计算总出现次数
        total_counts = red_freq_df['出现次数'].sum()

        # 计算百分比并添加到 DataFrame
        red_freq_df['百分比'] = (red_freq_df['出现次数'] / total_counts)

        # 创建 Altair 柱状图和文本图层
        bars = alt.Chart(red_freq_df).mark_bar(color='red').encode(
            x=alt.X('号码:O', title='红球号码', sort='-y',
                    axis=alt.Axis(labelAngle=-45, labelOverlap=False, labelFontSize=10)),
            y=alt.Y('出现次数:Q', title='出现次数'),
            color= alt.Color('出现次数:Q',scale=alt.Scale(scheme='reds'), legend=None),
            tooltip=['号码', '出现次数', alt.Tooltip('百分比', format=".1%")]  # Tooltip 中也显示百分比
        )

        text = alt.Chart(red_freq_df).mark_text(
            dy=-10
        ).encode(
            x=alt.X('号码:O', sort='-y'),
            y=alt.Y('出现次数:Q'),
            text=alt.Text(
                '百分比:Q',
                format=".1%'",  # 使用 format=".1f" 格式化数值部分
                formatType='number',  # 显式指定 formatType 为 'number'
            )
        )

        # 将柱状图和文本标签图层叠加
        chart = (bars).properties(
            title='红球出现频率 (出现次数及百分比)',
            width=800,
            height=300
        )

        # 在 Streamlit 中显示 Altair 图表
        st.altair_chart(chart, use_container_width=True)


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

            # 计算蓝球总出现次数
            blue_total_counts = blue_freq_df['出现次数'].sum()

            # 计算百分比并添加到 DataFrame
            blue_freq_df['百分比'] = (blue_freq_df['出现次数'] / blue_total_counts)

            # 创建 Altair 柱状图
            blue_bars = alt.Chart(blue_freq_df).mark_bar(color='blue').encode(
                x=alt.X('号码:O', title='蓝球号码', sort='-y',
                        axis=alt.Axis(labelAngle=-45, labelOverlap=False, labelFontSize=10)),
                y=alt.Y('出现次数:Q', title='出现次数'),
                tooltip=['号码', '出现次数', alt.Tooltip('百分比', format=".1%")]
                # Tooltip 中也显示百分比, 并添加百分号
            )

            # 文本标签图层
            blue_text = alt.Chart(blue_freq_df).mark_text(
                dy=-10  # 调整文本垂直位置
            ).encode(
                x=alt.X('号码:O', sort='-y'),
                y=alt.Y('出现次数:Q'),
                text=alt.Text(
                    '百分比:Q',
                    format=".1%",  # 使用 format=".1f" 格式化数值部分
                    formatType='number',  # 显式指定 formatType 为 'number'
                )
            )

            # 将柱状图和文本标签图层叠加
            blue_chart = (blue_bars).properties(
                title='蓝球出现频率 (出现次数及百分比)',
                width=800,
                height=300
            )

            st.altair_chart(blue_chart, use_container_width=True)

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
        st.subheader("奇偶比例分析")

        # 初始化奇偶比例的计数字典
        odd_even_counts = {
            '1:5': 0,
            '2:4': 0,
            '3:3': 0,
            '4:2': 0,
            '5:1': 0
        }

        # 统计每种奇偶组合的出现次数
        for _, row in filtered_data.iterrows():
            # 获取奇数数量
            odd_count = row['奇数']
            # 偶数数量是 6 - 奇数数量
            even_count = 6 - odd_count

            # 计算奇偶比例字符串
            ratio = f"{odd_count}:{even_count}"

            # 如果这个比例在字典中，则计数加 1
            if ratio in odd_even_counts:
                odd_even_counts[ratio] += 1

        # 将计数结果转为 DataFrame
        odd_even_df = pd.DataFrame(list(odd_even_counts.items()), columns=["奇偶比例", "出现次数"])

        # 计算百分比
        total_count = odd_even_df["出现次数"].sum()
        odd_even_df["百分比"] = (odd_even_df["出现次数"] / total_count)

        # 使用 Altair 绘制柱状图
        chart = alt.Chart(odd_even_df).mark_bar().encode(
            x=alt.X('奇偶比例:O', title='奇偶比例', axis=alt.Axis(labelAngle=0)),  # 旋转X轴标签
            y=alt.Y('出现次数:Q', title='出现次数'),
            color=alt.Color('出现次数:Q', legend=None),  # 使用奇偶比例的不同值显示不同颜色
            tooltip=['奇偶比例', '出现次数',
                     alt.Tooltip('百分比:Q', format='.1%', title='百分比')]  # 鼠标悬停显示的内容
        ).properties(
            title='奇偶比例分布',  # 图表标题
            width=800,  # 图表宽度
            height=300  # 图表高度
        )

        # 添加百分比文本标签
        text = chart.mark_text(
            align='center',
            baseline='bottom',
            dy=-5  # 调整文字位置
        ).encode(
            text=alt.Text('百分比:Q', format='.1%')
        )

        # 组合图表并显示
        st.altair_chart(chart + text, use_container_width=True)

    with col2:
        st.subheader("奇数和偶数变化趋势")

        # 进行数据转换，将 '奇数' 和 '偶数' 两列折叠为两列：类别 ('奇数' 或 '偶数') 和 数量
        chart_data = filtered_data.melt(id_vars=["期号"], value_vars=["奇数", "偶数"], var_name="类别",
                                        value_name="数量")

        # 将 '偶数' 的数量转换为负数，使其显示在中轴之下
        chart_data.loc[chart_data['类别'] == '偶数', '数量'] *= -1

        # 绘制柱状图
        chart = alt.Chart(chart_data).mark_bar().encode(
            x=alt.X('期号:O', title='期号',
                    axis=alt.Axis(labelAngle=-45)),  # X轴为期号
            y=alt.Y('数量:Q', title='数量', axis=alt.Axis(grid=True)),  # Y轴为数量，定量数据
            color=alt.Color('类别:N', legend=None),  # 根据类别（奇数、偶数）颜色区分
            tooltip=['期号', '类别', '数量']  # 鼠标悬停显示期号、类别和数量
        ).properties(
            title='奇偶号趋势',  # 图表标题
            width=800,  # 图表宽度
            height=300  # 图表高度
        )

        # 设置 Y 轴范围，使奇数在中轴之上，偶数在中轴之下
        chart = chart.encode(
            y=alt.Y('数量:Q', title='数量', scale=alt.Scale(domain=[-6, 6]))  # 设定 Y 轴范围，确保对称
        )

        # 显示图表
        st.altair_chart(chart, use_container_width=True)

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
        st.subheader("大小比例分析")

        # 初始化大小比例的计数字典
        size_ratio_counts = {
            '0:6': 0,
            '1:5': 0,
            '2:4': 0,
            '3:3': 0,
            '4:2': 0,
            '5:1': 0,
            '6:0': 0
        }

        # 统计每种大小组合的出现次数
        for _, row in filtered_data.iterrows():
            # 获取大号数量
            large_count = row['大号']
            # 小号数量是 6 - 大号数量
            small_count = 6 - large_count

            # 计算大小比例字符串
            ratio = f"{large_count}:{small_count}"

            # 如果这个比例在字典中，则计数加 1
            if ratio in size_ratio_counts:
                size_ratio_counts[ratio] += 1

        # 将计数结果转为 DataFrame
        size_ratio_df = pd.DataFrame(list(size_ratio_counts.items()), columns=["大小比例", "出现次数"])

        # 计算百分比
        total_count = size_ratio_df["出现次数"].sum()
        size_ratio_df["百分比"] = (size_ratio_df["出现次数"] / total_count)

        # 使用 Altair 绘制柱状图
        chart = alt.Chart(size_ratio_df).mark_bar().encode(
            x=alt.X('大小比例:O', title='大小比例', axis=alt.Axis(labelAngle=0)),  # 旋转X轴标签
            y=alt.Y('出现次数:Q', title='出现次数'),
            color=alt.Color('出现次数:Q', legend=None),  # 使用大小比例的不同值显示不同颜色
            tooltip=['大小比例', '出现次数',
                     alt.Tooltip('百分比:Q', format='.1%', title='百分比')]  # 鼠标悬停显示的内容
        ).properties(
            title='大小比例分布',  # 图表标题
            width=800,  # 图表宽度
            height=300  # 图表高度
        )

        # 添加百分比文本标签
        text = chart.mark_text(
            align='center',
            baseline='bottom',
            dy=-5  # 调整文字位置
        ).encode(
            text=alt.Text('百分比:Q', format='.1%')
        )

        # 组合图表并显示
        st.altair_chart(chart + text, use_container_width=True)

    with col2:
        st.subheader("大号和小号变化趋势")
        # 进行数据转换
        # 将 '大号' 和 '小号' 两列折叠为两列：类别 ('大号' 或 '小号') 和 数量
        chart_data = filtered_data.melt(id_vars=["期号"], value_vars=["大号", "小号"], var_name="类别",
                                        value_name="数量")

        # 将 '小号' 的数量转换为负数
        chart_data.loc[chart_data['类别'] == '小号', '数量'] *= -1

        # 绘制柱状图
        chart = alt.Chart(chart_data).mark_bar().encode(
            x=alt.X('期号:O', title='期号',
                    axis=alt.Axis(labelAngle=-45)),  # X轴为期号
            y=alt.Y('数量:Q', title='数量', axis=alt.Axis(grid=True)),  # Y轴为数量，定量数据
            color=alt.Color('类别:N', legend=None),  # 根据类别（大号、小号）颜色区分
            tooltip=['期号', '类别', '数量']  # 鼠标悬停显示期号、类别和数量
        ).properties(
            title='大小号趋势',  # 图表标题
            width=800,  # 图表宽度
            height=300  # 图表高度
        )

        # 将大号显示在中轴之上，小号显示在中轴之下
        chart = chart.encode(
            y=alt.Y('数量:Q', title='数量', scale=alt.Scale(domain=[-6, 6]))  # 将 Y 轴设置为对称的区间
        )

        # 显示图表
        st.altair_chart(chart, use_container_width=True)

    col1,col2 = st.columns(2)
    with col1:
        st.subheader("区间数字汇总分析")

        # 假设 filtered_data 是一个包含期号、一区、二区、三区等列的 DataFrame
        # 创建区间统计字典
        area_counts = {
            '一区': 0,
            '二区': 0,
            '三区': 0
        }

        # 统计每个区出现的数字数量，假设每个区的数字是已统计好的数量
        for _, row in filtered_data.iterrows():
            # 获取每个区出现的数字数量
            area_counts['一区'] += row['一区']  # 假设 '一区' 列是已经统计了出现数字的数量
            area_counts['二区'] += row['二区']  # 假设 '二区' 列是已经统计了出现数字的数量
            area_counts['三区'] += row['三区']  # 假设 '三区' 列是已经统计了出现数字的数量

        # 将计数结果转为 DataFrame
        area_df = pd.DataFrame(list(area_counts.items()), columns=["区间", "出现次数"])

        # 计算百分比
        total_count = area_df["出现次数"].sum()
        area_df["百分比"] = (area_df["出现次数"] / total_count) # 转换为百分比

        # 固定顺序，确保 X轴按 位置顺序排列
        order = ['一区', '二区', '三区']
        area_df['区间'] = pd.Categorical(area_df['区间'], categories=order, ordered=True)

        # 使用 Altair 绘制柱状图
        chart = alt.Chart(area_df).mark_bar().encode(
            x=alt.X('区间:O', title='区间', axis=alt.Axis(labelAngle=0),sort=order),  # X轴为区间
            y=alt.Y('出现次数:Q', title='出现次数'),
            color=alt.Color('区间:N', legend=None),  # 使用不同区间显示不同颜色
            tooltip=['区间', '出现次数',
                     alt.Tooltip('百分比:Q', format='.1%', title='百分比')]  # 鼠标悬停显示区间和出现次数
        ).properties(
            title='区间出现数字次数(一区：1-11；二区：12-24；三区：25-33)',  # 图表标题
            width=800,  # 图表宽度
            height=300  # 图表高度
        )

        # 添加百分比文本标签
        text = chart.mark_text(
            align='center',
            baseline='bottom',
            dy=-5  # 调整文字位置
        ).encode(
            text=alt.Text('百分比:Q', format='.1%')
        )

        # 组合图表并显示
        st.altair_chart(chart + text, use_container_width=True)


    with col2:
        st.subheader("区间数字走势")

        # 假设 filtered_data 是一个包含期号、一区、二区、三区等列的 DataFrame
        # 创建一个新的 DataFrame 用来存储每一期的区间数据
        area_data = pd.DataFrame({
            '期号': filtered_data['期号'],  # 假设有期号列
            '一区': filtered_data['一区'],  # 假设 '一区' 列是已统计了出现数字的数量
            '二区': filtered_data['二区'],  # 假设 '二区' 列是已统计了出现数字的数量
            '三区': filtered_data['三区']  # 假设 '三区' 列是已统计了出现数字的数量
        })

        # 将数据转化为长格式 (long format)，方便绘制折线图
        area_long_df = area_data.melt(id_vars=['期号'], value_vars=['一区', '二区', '三区'],
                                      var_name='区间', value_name='出现次数')

        # 使用 Altair 绘制折线图
        chart = alt.Chart(area_long_df).mark_line().encode(
            x=alt.X('期号:O', title='期号',axis=alt.Axis(labelAngle=-45)),  # X轴为期号
            y=alt.Y('出现次数:Q', title='出现次数', axis=alt.Axis(format='d')),
            color=alt.Color('区间:N', title='区间'),  # 用不同颜色表示不同的区间
            tooltip=['期号', '区间', '出现次数']  # 鼠标悬停显示期号、区间和出现次数
        ).properties(
            title='区间出现数字走势',  # 图表标题
            width=800,  # 图表宽度
            height=300  # 图表高度
        )

        # 显示折线图
        st.altair_chart(chart, use_container_width=True)

    col1, col2 = st.columns(2)  #重号分析
    with col1:
        st.subheader("红球重号统计")

        # 预处理数据
        filtered_data['重号'] = filtered_data['重号'].fillna(0).astype(int)

        # 生成完整数据范围
        all_values = pd.DataFrame({'同号数量': list(range(6))})
        frequency_df = (
            filtered_data['重号'].value_counts()
            .reindex(range(6), fill_value=0)  # 直接重建索引更高效
            .reset_index()
        )
        frequency_df.columns = ['同号数量', '出现次数']

        # 转换数据类型并计算百分比
        frequency_df['出现次数'] = frequency_df['出现次数'].astype(int)  # 确保为整数
        total_count = frequency_df["出现次数"].sum()
        frequency_df["百分比"] = frequency_df["出现次数"] / total_count if total_count > 0 else 0

        # 创建Altair图表
        chart = alt.Chart(frequency_df).mark_bar().encode(
            x=alt.X('同号数量:N',  # 使用名义类型
                    title='同号数量（个）',
                    sort=alt.EncodingSortField('同号数量', order='ascending'),  # 明确排序字段
                    axis=alt.Axis(labelAngle=0)),
            y=alt.Y('出现次数:Q',
                    title='出现次数',
                    axis=alt.Axis(format='d'),  # 整数格式化
                    scale=alt.Scale(domainMin=0)),  # 强制Y轴从0开始
            tooltip=[
                '同号数量:N',
                '出现次数:Q',
                alt.Tooltip('百分比:Q', format='.1%', title='百分比')
            ]
        ).properties(
            title='红球同号数量统计(与上一期相同的号码)',
            width=800,
            height=300
        )

        # 添加文本标签（仅显示非零值）
        text = chart.mark_text(
            align='center',
            baseline='bottom',
            dy=-5,
            color='#333333'
        ).encode(
            text=alt.condition(
                alt.datum.出现次数 > 0,  # 仅当次数>0时显示
                alt.Text('百分比:Q', format='.1%'),
                alt.value('')
            )
        )

        st.altair_chart(chart + text, use_container_width=True)

    with col2:
        st.subheader("红球重号分析")

        # 创建 DataFrame，包含 "期号" 和 "重号"
        same_numbers_df = pd.DataFrame({
            '期号': filtered_data['期号'],  # 期号
            '重号数量': filtered_data['重号']  # 已计算的重号数量
        })

        # 创建 Altair 折线图
        chart = alt.Chart(same_numbers_df).mark_line().encode(
            x=alt.X('期号:O', title='期号', axis=alt.Axis(labelAngle=-45, labelFontSize=10)),  # X 轴
            y=alt.Y('重号数量:Q', title='重号数量',axis=alt.Axis(format='d')),  # Y 轴显示整数
            #color=alt.value(''),  # 设定线条颜色
            tooltip=['期号', '重号数量']
        ).properties(
            title='红球重号趋势图',
            width=800,
            height=300
        )

        # 显示折线图
        st.altair_chart(chart, use_container_width=True)

    col1,col2 = st.columns(2) #邻号分析

    with col1:
        st.subheader("红球邻号统计")

        # 数据预处理
        filtered_data['邻号'] = filtered_data['邻号'].fillna(0).astype(int)

        # 生成完整数据范围并统计
        all_values = pd.DataFrame({'邻号数量': list(range(7))})  # 0-6
        frequency_df = (
            filtered_data['邻号']
            .value_counts()
            .reindex(all_values['邻号数量'], fill_value=0)
            .reset_index()
        )
        frequency_df.columns = ['邻号数量', '出现次数']

        # 强制转换数据类型（关键修复）
        frequency_df['出现次数'] = frequency_df['出现次数'].astype(int)  # 确保为整数

        # 计算百分比（保持小数形式）
        total_count = frequency_df["出现次数"].sum()
        frequency_df["百分比"] = frequency_df["出现次数"] / total_count if total_count > 0 else 0

        # 创建Altair图表（修复编码问题）
        chart = alt.Chart(frequency_df).mark_bar().encode(
            x=alt.X('邻号数量:N',  # 使用名义类型
                    title='邻号数量（个）',
                    sort=alt.EncodingSortField('邻号数量', order='ascending'),  # 明确排序
                    axis=alt.Axis(labelAngle=0)),
            y=alt.Y('出现次数:Q',
                    title='出现次数',
                    axis=alt.Axis(format='d'),
                    scale=alt.Scale(domainMin=0)),  # 强制从0开始

            tooltip=[
                alt.Tooltip('邻号数量:N', title='邻号数'),
                alt.Tooltip('出现次数:Q', title='出现次数'),
                alt.Tooltip('百分比:Q', format='.1%', title='占比')  # 正确百分比格式
            ]
        ).properties(
            title='红球邻号数量统计',
            width=800,
            height=300
        )

        # 添加文本标签（修复显示条件）
        text = chart.mark_text(
            align='center',
            baseline='bottom',
            dy=-5,
            color='black'
        ).encode(
            text=alt.condition(
                alt.datum.出现次数 > 0,  # 仅显示非零值
                alt.Text('百分比:Q', format='.1%'),
                alt.value('')
            )
        )

        st.altair_chart(chart + text, use_container_width=True)
    with col2:
        st.subheader("红球邻号分析")

        # **创建 DataFrame，包含 "期号" 和 "邻号"**
        neighbor_numbers_df = pd.DataFrame({
            '期号': filtered_data['期号'],  # 期号
            '邻号数量': filtered_data['邻号']  # 已计算的邻号数量
        })

        # **创建 Altair 折线图**
        chart = alt.Chart(neighbor_numbers_df).mark_line().encode(
            x=alt.X('期号:O', title='期号', axis=alt.Axis(labelAngle=-45, labelFontSize=10)),  # **X 轴，期号**
            y=alt.Y('邻号数量:Q', title='邻号数量', axis=alt.Axis(format='d')),  # **Y 轴，整数**
            color=alt.value('#1E90FF'),  # **设定折线颜色为蓝色**
            tooltip=['期号', '邻号数量']  # **鼠标悬停显示数据**
        ).properties(
            title='红球邻号趋势图',  # **图表标题**
            width=800,  # **宽度**
            height=300  # **高度**
        )

        # **显示折线图**
        st.altair_chart(chart, use_container_width=True)

    col1,col2 = st.columns(2)
    with col1:
        st.subheader("红球孤号统计")

        # **数据预处理**
        filtered_data['孤号'] = filtered_data['孤号'].fillna(0).astype(int)

        # **生成完整数据范围并统计**
        all_values = pd.DataFrame({'孤号数量': list(range(7))})  # 假设最大孤号数量为 6
        frequency_df = (
            filtered_data['孤号']
            .value_counts()
            .reindex(all_values['孤号数量'], fill_value=0)
            .reset_index()
        )
        frequency_df.columns = ['孤号数量', '出现次数']

        # **强制转换数据类型（关键修复）**
        frequency_df['出现次数'] = frequency_df['出现次数'].astype(int)  # 确保为整数

        # **计算百分比（保持小数形式）**
        total_count = frequency_df["出现次数"].sum()
        frequency_df["百分比"] = frequency_df["出现次数"] / total_count if total_count > 0 else 0

        # **创建 Altair 图表**
        chart = alt.Chart(frequency_df).mark_bar().encode(
            x=alt.X('孤号数量:N',  # **使用名义类型**
                    title='孤号数量（个）',
                    sort=alt.EncodingSortField('孤号数量', order='ascending'),  # **明确排序**
                    axis=alt.Axis(labelAngle=0)),
            y=alt.Y('出现次数:Q',
                    title='出现次数',
                    axis=alt.Axis(format='d'),
                    scale=alt.Scale(domainMin=0)),  # **强制从0开始**

            tooltip=[
                alt.Tooltip('孤号数量:N', title='孤号数'),
                alt.Tooltip('出现次数:Q', title='出现次数'),
                alt.Tooltip('百分比:Q', format='.1%', title='占比')  # **正确百分比格式**
            ]
        ).properties(
            title='红球孤号数量统计',
            width=800,
            height=300
        )

        # **添加文本标签**
        text = chart.mark_text(
            align='center',
            baseline='bottom',
            dy=-5,
            color='black'
        ).encode(
            text=alt.condition(
                alt.datum.出现次数 > 0,  # **仅显示非零值**
                alt.Text('百分比:Q', format='.1%'),
                alt.value('')
            )
        )

        # **显示图表**
        st.altair_chart(chart + text, use_container_width=True)

    with col2:

        st.subheader("红球孤号分析")

        # **创建 DataFrame，包含 "期号" 和 "孤号"**
        isolated_numbers_df = pd.DataFrame({
            '期号': filtered_data['期号'],  # 期号
            '孤号数量': filtered_data['孤号']  # 已计算的孤号数量
        })

        # **创建 Altair 折线图**
        chart = alt.Chart(isolated_numbers_df).mark_line().encode(
            x=alt.X('期号:O', title='期号', axis=alt.Axis(labelAngle=-45, labelFontSize=10)),  # **X 轴，期号**
            y=alt.Y('孤号数量:Q', title='孤号数量', axis=alt.Axis(format='d')),  # **Y 轴，整数**
            #color=alt.value('blue'),  #
            tooltip=['期号', '孤号数量']  # **鼠标悬停显示数据**
        ).properties(
            title='红球孤号趋势图',  # **图表标题**
            width=800,  # **宽度**
            height=300  # **高度**
        )

        # **显示折线图**
        st.altair_chart(chart, use_container_width=True)

with tab2:
    col1, col2 = st.columns(2)  # 创建两列布局

    with col1:
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
            selected_numbers_html = "<div class='selected-numbers-display' style='width: 500px;height: 200px'>"
            selected_numbers_html += "<p>已选红球:</p><div>"
            for ball in sorted(st.session_state.selected_red_balls):
                selected_numbers_html += f"<span class='lottery-ball red-ball selected'>{ball}</span>"
            selected_numbers_html += "</div>"

            selected_numbers_html += "<p>已选蓝球:</p><div>"
            for ball in sorted(st.session_state.selected_blue_balls):
                selected_numbers_html += f"<span class='lottery-ball blue-ball selected'>{ball}</span>"
            selected_numbers_html += "</div>"

            if not st.session_state.selected_red_balls and not st.session_state.selected_blue_balls:
                selected_numbers_html += "<p></p>"

            selected_numbers_html += "</div>"
            placeholder.markdown(selected_numbers_html, unsafe_allow_html=True)

        display_selected_numbers()

    with col2:
        st.subheader("选号记录")
        if 'bets_text' not in st.session_state:
            st.session_state.bets_text = ""  # 初始化 bets_text

        if st.session_state.bets:
            bets_text = "\n".join([f" {bet}" for bet in st.session_state.bets])
            st.session_state.bets_text = bets_text  # 更新 bets_text
        else:
           pass

        st.text_area(label="你的选号:", value=st.session_state.bets_text, height=200,
                     key="bets_text_area")  # 更新 text_area

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
        if len(st.session_state.selected_red_balls) >= 6 and len(st.session_state.selected_blue_balls) >= 1:
            red_balls_analysis = analyze_red_balls(st.session_state.selected_red_balls)

            # 格式化分析结果
            analysis_str = ", ".join([f"{key}: {value}" for key, value in red_balls_analysis.items()])
            red_balls_str = ",".join(map(str, sorted(st.session_state.selected_red_balls)))
            blue_balls_str = ",".join(map(str, sorted(st.session_state.selected_blue_balls)))
            bet_str = f"{red_balls_str}+{blue_balls_str} ({analysis_str})"  # 将分析结果加在号码后面

            st.session_state.bets.append(bet_str)

            # 更新 text_area 的值
            bets_text = "\n".join([f" {bet}" for bet in st.session_state.bets])
            st.session_state.bets_text = bets_text
        else:
            if len(st.session_state.selected_red_balls) < 6:
                st.warning("红球选择至少6个")
            elif len(st.session_state.selected_blue_balls) < 1:
                st.warning("蓝球选择至少1个")
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


with tab3:
    st.subheader("历史开奖数据")
    historical_data = load_historical_data(100)
    historical_data=historical_data.style.set_properties(**{'text-align': 'left'})
 # 获取所有数据

    st.dataframe(historical_data, width=1000, height=500) # Display historical data in a table