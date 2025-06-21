import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from collections import Counter
import logging
from funcs.functions import analyze_top_companion_pairs,analyze_top_triples

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler( 'my_log_file.log')  # 修改这行
    ]
)

# Page configuration
st.set_page_config(
    page_title="双色球分析与选号工具",
    layout="wide",
    initial_sidebar_state="expanded"
)

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
        min_value=5,
        max_value=100,
        value=5,
        step=5
    )
    # 确保在使用 session_state 前进行初始化
    if 'hot_nums_filter' not in st.session_state:
        st.session_state.hot_nums_filter = False
    if 'cold_nums_filter' not in st.session_state:
        st.session_state.cold_nums_filter = False
    if 'odd_even_filter' not in st.session_state:
        st.session_state.odd_even_filter = False
    if 'small_big_filter' not in st.session_state:
        st.session_state.small_big_filter = False
    if 'same_nums_filter' not in st.session_state:
        st.session_state.same_nums_filter = False
    if 'neigh_nums_filter' not in st.session_state:
        st.session_state.neigh_nums_filter = False
    if 'sep_nums_filter' not in st.session_state:
        st.session_state.sep_nums_filter = False
    if 'consecutive_filter' not in st.session_state:
        st.session_state.consecutive_filter = False
    if 'same_tail_filter' not in st.session_state:
        st.session_state.same_tail_filter = False
    if 'two_tail_filter' not in st.session_state:
        st.session_state.two_tail_filter = False
    if 'three_tail_filter' not in st.session_state:
        st.session_state.three_tail_filter = False
    if 'sum_filter' not in st.session_state:
        st.session_state.sum_filter = False
    if 'span_filter' not in st.session_state:
        st.session_state.span_filter = False
    if 'two_consecutive_filter' not in st.session_state:
        st.session_state.two_consecutive_filter = False
    if 'three_consecutive_filter' not in st.session_state:
        st.session_state.three_consecutive_filter = False
    if 'skip_nums_filter' not in st.session_state:
        st.session_state.skip_nums_filter = False
    if 'two_skip_nums_filter' not in st.session_state:
        st.session_state.two_skip_nums_filter = False
    if 'three_skip_nums_filter' not in st.session_state:
        st.session_state.three_skip_nums_filter = False
    if 'zone_filter' not in st.session_state:
        st.session_state.zone_filter = False
    if 'zone_count' not in st.session_state:
        st.session_state.zone_count = None

            # 显示筛选器并将选中的值保存在 session_state 中
    st.divider()
    st.subheader("红球筛选条件设置")

    st.session_state.hot_nums_filter = st.checkbox("热号筛选", value=st.session_state.hot_nums_filter)
    if st.session_state.hot_nums_filter:
        st.session_state.hot_nums = st.slider("红球热号个数", 0, 6, (1, 2))

    st.session_state.cold_nums_filter = st.checkbox("冷号筛选", value=st.session_state.cold_nums_filter)
    if st.session_state.cold_nums_filter:
        st.session_state.cold_nums = st.slider("红球冷号个数", 0, 6, (1, 2))

    st.session_state.zone_filter = st.checkbox("区域号码数筛选", value=st.session_state.zone_filter)
    if st.session_state.zone_filter:
        st.session_state.zone1_count = st.slider("一区号码数范围 (1-11)", 0, 6, (0, 6))
        st.session_state.zone2_count = st.slider("二区号码数范围 (12-22)", 0, 6, (0, 6))
        st.session_state.zone3_count = st.slider("三区号码数范围 (23-33)", 0, 6, (0, 6))

    st.session_state.odd_even_filter = st.checkbox("奇偶筛选", value=st.session_state.odd_even_filter)
    if st.session_state.odd_even_filter:
        st.session_state.odd_count = st.slider("红球奇数个数", 0, 6, (2, 4))

    st.session_state.small_big_filter = st.checkbox("大小筛选", value=st.session_state.small_big_filter)
    if st.session_state.small_big_filter:
        st.session_state.small_big = st.slider("红球小号个数", 0, 6, (2, 4))

    st.session_state.same_nums_filter = st.checkbox("重号筛选", value=st.session_state.same_nums_filter)
    if st.session_state.same_nums_filter:
        st.session_state.same_nums = st.slider("红球重号个数（与上期）", 0, 6, (0, 2))

    st.session_state.neigh_nums_filter = st.checkbox("邻号筛选", value=st.session_state.neigh_nums_filter)
    if st.session_state.neigh_nums_filter:
        st.session_state.neigh_nums = st.slider("红球邻号个数（与上期）", 0, 6, (0, 2))

    st.session_state.sep_nums_filter = st.checkbox("孤号筛选", value=st.session_state.sep_nums_filter)
    if st.session_state.sep_nums_filter:
        st.session_state.sep_nums = st.slider("红球孤号个数（与上期）", 0, 6, (0, 2))

    st.session_state.consecutive_filter = st.checkbox("连号筛选", value=st.session_state.consecutive_filter)
    if st.session_state.consecutive_filter:
        st.session_state.consecutive_count = st.slider("最多连号数量", 0, 6, 2)

    st.session_state.two_consecutive_filter = st.checkbox("二连组数", value=st.session_state.two_consecutive_filter)
    if st.session_state.two_consecutive_filter:
        st.session_state.two_consecutive_count = st.slider("二连组数范围", 0, 3, (0, 1))

    st.session_state.three_consecutive_filter = st.checkbox("三连组数", value=st.session_state.three_consecutive_filter)
    if st.session_state.three_consecutive_filter:
        st.session_state.three_consecutive_count = st.slider("三连组数范围", 0, 2, (0, 1))

    st.session_state.skip_nums_filter = st.checkbox("跳号", value=st.session_state.skip_nums_filter)
    if st.session_state.skip_nums_filter:
        st.session_state.skip_nums = st.slider("跳号个数", 0, 3, (0, 1))

    st.session_state.two_skip_nums_filter = st.checkbox("二跳组数", value=st.session_state.two_skip_nums_filter)
    if st.session_state.two_skip_nums_filter:
        st.session_state.two_skip_nums = st.slider("二跳组数范围", 0, 3, (0, 1))

    st.session_state.three_skip_nums_filter = st.checkbox("三跳组数", value=st.session_state.three_skip_nums_filter)
    if st.session_state.three_skip_nums_filter:
        st.session_state.three_skip_nums = st.slider("三跳组数范围", 0, 6, (0, 2))

    st.session_state.same_tail_filter = st.checkbox("同尾号筛选", value=st.session_state.same_tail_filter)
    if st.session_state.same_tail_filter:
        st.session_state.max_same_tail = st.slider("最多同尾号数量", 0, 6, 2)

    st.session_state.two_tail_filter = st.checkbox("二尾组数", value=st.session_state.two_tail_filter)
    if st.session_state.two_tail_filter:
        st.session_state.two_tail_count = st.slider("二尾组数范围", 0, 3, (0, 1))

    st.session_state.three_tail_filter = st.checkbox("三尾组数", value=st.session_state.three_tail_filter)
    if st.session_state.three_tail_filter:
        st.session_state.three_tail_count = st.slider("三尾组数范围", 0, 2, (0, 1))

    st.session_state.sum_filter = st.checkbox("和值筛选", value=st.session_state.sum_filter)
    if st.session_state.sum_filter:
        st.session_state.sum_range = st.slider("红球和值范围", 20, 200, (70, 130))

    st.session_state.span_filter = st.checkbox("跨度筛选", value=st.session_state.span_filter)
    if st.session_state.span_filter:
        st.session_state.span_range = st.slider("红球跨度范围", 10, 33, (15, 25))



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
                "二跳", "三跳", "四跳", "五跳", "六跳","一等奖奖金","二等奖奖金"
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
    results['连号'] = ", ".join([f"{length}连{c}" for length, c in consecutive_counts.items()])

    # 5. 同尾号
    tails = [ball % 10 for ball in red_balls] # 取模操作，取个位数的数字
    tail_counts = {}
    for tail in tails:
        tail_counts[tail] = tail_counts.get(tail, 0) + 1

    same_tail_counts = {}
    for tail, count in tail_counts.items():
        if count > 1:
            same_tail_counts[count] = same_tail_counts.get(count, 0) + 1
    results['同尾'] = ", ".join([f"{count}" for count in same_tail_counts.keys()])

    # 6. 最新一期同号
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
    results['重号'] = f"{same_count}"

    return results

# 加载历史数据，并根据分析期数筛选
filtered_data = load_historical_data(analysis_period)
st.session_state.lottery_results=load_historical_data(10)
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

# Main content area
st.markdown("<div class='header'>双色球分析工具</div>", unsafe_allow_html=True)

# Display tabs for different analyses
tab1, tab2, tab3,tab4= st.tabs(["号码分析", "选号工具","全量筛选","历史数据"])

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

    # 调用 analyze_red_balls 函数获取分析结果字典
    red_analysis_result_dict = analyze_red_balls(red_balls_latest_draw)

    # 构建逗号分隔的分析结果字符串
    red_analysis_str = ", ".join([f"{key}: {value}" for key, value in red_analysis_result_dict.items()])

    # 将分析结果字符串添加到 latest_draw_html
    latest_draw_html += f"""
                <div class='analysis-box'>
                    <div class='analysis-title'>红球分析:
                        {red_analysis_str}
                    </div>
                </div>
            """

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

    col1, col2 = st.columns(2) #连号
    with col1:
        st.subheader("连号分析")

        # 假设 filtered_data 是一个包含连号统计列的 DataFrame
        # 创建连号统计字典
        consecutive_counts = {
            '二连': 0,
            '三连': 0,
            '四连': 0,
            '五连': 0,
            '六连': 0
        }

        # 统计每个连号出现的次数
        for _, row in filtered_data.iterrows():
            consecutive_counts['二连'] += row['二连']
            consecutive_counts['三连'] += row['三连']
            consecutive_counts['四连'] += row['四连']
            consecutive_counts['五连'] += row['五连']
            consecutive_counts['六连'] += row['六连']

        # 将计数结果转换为 DataFrame
        consecutive_df = pd.DataFrame(list(consecutive_counts.items()), columns=["连号", "出现次数"])

        # 计算百分比
        total_count = consecutive_df["出现次数"].sum()
        consecutive_df["百分比"] = (consecutive_df["出现次数"] / total_count)

        # 固定顺序，确保 X 轴按连号顺序排列
        order = ['二连', '三连', '四连', '五连', '六连']
        consecutive_df['连号'] = pd.Categorical(consecutive_df['连号'], categories=order, ordered=True)

        # 使用 Altair 绘制柱状图
        chart = alt.Chart(consecutive_df).mark_bar().encode(
            x=alt.X('连号:O', title='连号', axis=alt.Axis(labelAngle=0), sort=order),
            y=alt.Y('出现次数:Q', title='出现次数'),
            color=alt.Color('出现次数:Q', legend=None),
            tooltip=['连号', '出现次数', alt.Tooltip('百分比:Q', format='.1%', title='百分比')]
        ).properties(
            title='连号出现次数统计',
            width=800,
            height=300
        )

        # 添加百分比文本标签
        text = chart.mark_text(
            align='center',
            baseline='bottom',
            dy=-5
        ).encode(
            text=alt.Text('百分比:Q', format='.1%')
        )

        # 组合图表并显示
        st.altair_chart(chart + text, use_container_width=True)

    with col2:
        st.subheader("连号趋势分析")

        # 从 filtered_data 读取连号数据
        trend_data = []
        for _, row in filtered_data.iterrows():
            issue_no = row['期号']
            two_consecutive = row['二连']
            three_consecutive = row['三连']
            four_consecutive = row['四连']
            five_consecutive = row['五连']
            six_consecutive = row['六连']

            # 添加数据点（如果存在连号）
            if two_consecutive > 0:
                trend_data.append({
                    '期号': int(issue_no),
                    '连号类型': '二连',
                    '连号组数': two_consecutive
                })
            if three_consecutive > 0:
                trend_data.append({
                    '期号': int(issue_no),
                    '连号类型': '三连',
                    '连号组数': three_consecutive
                })
            if four_consecutive > 0:
                trend_data.append({
                    '期号': int(issue_no),
                    '连号类型': '四连',
                    '连号组数': four_consecutive
                })
            if five_consecutive > 0:
                trend_data.append({
                    '期号': int(issue_no),
                    '连号类型': '五连',
                    '连号组数': five_consecutive
                })
            if six_consecutive > 0:
                trend_data.append({
                    '期号': int(issue_no),
                    '连号类型': '六连',
                    '连号组数': six_consecutive
                })

        # 创建趋势分析DataFrame
        trend_df = pd.DataFrame(trend_data)

        # 创建交互式趋势图
        base = alt.Chart(trend_df).properties(
            width=800,
            height=300
        )

        points = base.mark_circle(size=60).encode(
            x=alt.X('期号:O',
                    title='期号',
                    axis=alt.Axis(labelAngle=-45)),
            y=alt.Y('连号类型:N',
                    title='连号类型',
                    sort=['二连', '三连', '四连', '五连', '六连']),
            color=alt.Color('连号类型:N',
                            scale=alt.Scale(
                                domain=['二连', '三连', '四连', '五连', '六连'],
                                range=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
                            )),
            tooltip=[
                '期号:O',
                '连号类型:N',
                '连号组数:Q'
            ]
        )

        text = base.mark_text(
            align='center',
            baseline='bottom',
            dx=-5
        ).encode(
            x='期号:O',
            y='连号类型:N',
            text='连号组数:Q',
            color=alt.value('black')
        )

        st.altair_chart(
            (points + text).properties(title='红球连号趋势'),
            use_container_width=True)

    col1, col2 = st.columns(2) #跳号
    with col1:
        st.subheader("跳号分析")

        # 假设 filtered_data 是一个包含跳号统计列的 DataFrame
        # 创建跳号统计字典
        jump_counts = {
            '二跳': 0,
            '三跳': 0,
            '四跳': 0,
            '五跳': 0,
            '六跳': 0
        }

        # 统计每个跳号出现的次数
        for _, row in filtered_data.iterrows():
            jump_counts['二跳'] += row['二跳']
            jump_counts['三跳'] += row['三跳']
            jump_counts['四跳'] += row['四跳']
            jump_counts['五跳'] += row['五跳']
            jump_counts['六跳'] += row['六跳']

        # 将计数结果转换为 DataFrame
        jump_df = pd.DataFrame(list(jump_counts.items()), columns=["跳号", "出现次数"])

        # 计算百分比
        total_count = jump_df["出现次数"].sum()
        jump_df["百分比"] = (jump_df["出现次数"] / total_count)

        # 固定顺序，确保 X 轴按跳号顺序排列
        order = ['二跳', '三跳', '四跳', '五跳', '六跳']
        jump_df['跳号'] = pd.Categorical(jump_df['跳号'], categories=order, ordered=True)

        # 使用 Altair 绘制柱状图
        chart = alt.Chart(jump_df).mark_bar().encode(
            x=alt.X('跳号:O', title='跳号', axis=alt.Axis(labelAngle=0), sort=order),
            y=alt.Y('出现次数:Q', title='出现次数'),
            color=alt.Color('出现次数:Q', legend=None),
            tooltip=['跳号', '出现次数', alt.Tooltip('百分比:Q', format='.1%', title='百分比')]
        ).properties(
            title='跳号出现次数统计',
            width=800,
            height=300
        )

        # 添加百分比文本标签
        text = chart.mark_text(
            align='center',
            baseline='bottom',
            dy=-5
        ).encode(
            text=alt.Text('百分比:Q', format='.1%')
        )

        # 组合图表并显示
        st.altair_chart(chart + text, use_container_width=True)
    with col2:
        st.subheader("跳号趋势分析")

        # 从 filtered_data 读取跳号数据
        trend_data = []
        for _, row in filtered_data.iterrows():
            issue_no = row['期号']
            two_jump = row['二跳']
            three_jump = row['三跳']
            four_jump = row['四跳']
            five_jump = row['五跳']
            six_jump = row['六跳']

            # 添加数据点（如果存在跳号）
            if two_jump > 0:
                trend_data.append({
                    '期号': int(issue_no),
                    '跳号类型': '二跳',
                    '跳号组数': two_jump
                })
            if three_jump > 0:
                trend_data.append({
                    '期号': int(issue_no),
                    '跳号类型': '三跳',
                    '跳号组数': three_jump
                })
            if four_jump > 0:
                trend_data.append({
                    '期号': int(issue_no),
                    '跳号类型': '四跳',
                    '跳号组数': four_jump
                })
            if five_jump > 0:
                trend_data.append({
                    '期号': int(issue_no),
                    '跳号类型': '五跳',
                    '跳号组数': five_jump
                })
            if six_jump > 0:
                trend_data.append({
                    '期号': int(issue_no),
                    '跳号类型': '六跳',
                    '跳号组数': six_jump
                })

        # 创建趋势分析DataFrame
        trend_df = pd.DataFrame(trend_data)

        # 创建交互式趋势图
        base = alt.Chart(trend_df).properties(
            width=800,
            height=300
        )

        points = base.mark_circle(size=60).encode(
            x=alt.X('期号:O',
                    title='期号',
                    axis=alt.Axis(labelAngle=-45)),
            y=alt.Y('跳号类型:N',
                    title='跳号类型',
                    sort=['二跳', '三跳', '四跳', '五跳', '六跳']),
            color=alt.Color('跳号类型:N',
                            scale=alt.Scale(
                                domain=['二跳', '三跳', '四跳', '五跳', '六跳'],
                                range=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
                            )),
            tooltip=[
                '期号:O',
                '跳号类型:N',
                '跳号组数:Q'
            ]
        )

        text = base.mark_text(
            align='center',
            baseline='middle',
            dx=15
        ).encode(
            x='期号:O',
            y='跳号类型:N',
            text='跳号组数:Q',
            color=alt.value('black')
        )

        st.altair_chart(
            (points + text).properties(title='红球跳号趋势'),
            use_container_width=True)

    col1,col2 = st.columns(2)
    with col1:
        st.subheader("同尾号分析")

        # 统计同尾号（排除重复计数）
        tail_categories = ['二尾', '三尾', '四尾', '五尾', '六尾']
        category_map = {2: '二尾', 3: '三尾', 4: '四尾', 5: '五尾', 6: '六尾'}

        # 统计每期最大同尾数
        max_tail_counts = []
        for _, row in filtered_data.iterrows():
            red_balls = [row.get(f'红球{i}', 0) for i in range(1, 7)]
            red_balls = [b for b in red_balls if b > 0]  # 过滤无效值

            if len(red_balls) < 2:
                continue

            # 统计尾号频率
            tail_counter = Counter(num % 10 for num in red_balls)
            max_count = max(tail_counter.values(), default=0)

            # 只记录最大同尾数（≥2的情况）
            if max_count >= 2:
                max_tail_counts.append(min(max_count, 6))  # 最大限制为六尾

        # 生成统计DataFrame
        count_series = pd.Series(max_tail_counts).value_counts().reindex(range(2, 7), fill_value=0)
        stats_df = pd.DataFrame({
            '同尾类型': [category_map.get(i, f'{i}尾') for i in count_series.index],
            '出现次数': count_series.values
        })

        # 计算百分比
        total = stats_df['出现次数'].sum()
        stats_df['百分比'] = stats_df['出现次数'] / total if total > 0 else 0

        # 创建Altair图表
        chart = alt.Chart(stats_df).mark_bar(color='#4C78A8').encode(
            x=alt.X('同尾类型:N',
                    title='同尾类型',
                    sort=tail_categories,
                    axis=alt.Axis(labelAngle=0)),
            y=alt.Y('出现次数:Q',
                    title='出现次数',
                    axis=alt.Axis(format='d'),
                    scale=alt.Scale(domainMin=0)),
            color=alt.Color('出现次数:Q', legend=None),  # 颜色区分
            tooltip=[
                alt.Tooltip('同尾类型:N', title='类型'),
                alt.Tooltip('出现次数:Q', title='次数'),
                alt.Tooltip('百分比:Q', format='.1%', title='占比')
            ]
        ).properties(
            title='红球同尾号分布统计',
            width=600,
            height=300
        )

        # 添加百分比标签
        text = chart.mark_text(
            align='center',
            baseline='bottom',
            dy=-5,
            color='white'
        ).encode(
            text=alt.Text('百分比:Q', format='.1%')
        )

        st.altair_chart(chart + text, use_container_width=True)

        with col2:
            st.subheader("同尾号趋势分析")

            # 确保 filtered_data 按照期号排序
            filtered_data = filtered_data.sort_values(by='期号')

            # 生成趋势分析数据
            trend_data = []
            for _, row in filtered_data.iterrows():
                issue_no = row['期号']
                red_balls = [row.get(f'红球{i}', 0) for i in range(1, 7)]
                red_balls = [b for b in red_balls if b > 0]

                if len(red_balls) < 2:
                    continue

                # 统计尾号并获取同尾数组
                tail_counter = Counter(num % 10 for num in red_balls)
                same_tails = [count for count in tail_counter.values() if count >= 2]

                if same_tails:
                    for count in same_tails:
                        trend_data.append({
                            '期号': int(issue_no),
                            '最大同尾数': min(count, 6),
                            '同尾类型': category_map.get(min(count, 6), f'{min(count, 6)}尾'),
                            '同尾组数': same_tails.count(count)
                        })

            # 创建趋势分析DataFrame
            trend_df = pd.DataFrame(trend_data)

            # 创建交互式趋势图
            base = alt.Chart(trend_df).properties(
                width=800,
                height=300
            )

            points = base.mark_circle(size=60).encode(
                x=alt.X('期号:O',
                        title='期号',
                        axis=alt.Axis(labelAngle=-45)),
                y=alt.Y('最大同尾数:Q',
                        title='最大同尾数',
                        axis=alt.Axis(format='d'),
                        scale=alt.Scale(domain=[1, 6])),
                color=alt.Color('同尾类型:N',
                                scale=alt.Scale(
                                    domain=['二尾', '三尾', '四尾', '五尾', '六尾'],
                                    range=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
                                )),
                tooltip=[
                    '期号:O',
                    '同尾类型:N',
                    '最大同尾数:Q',
                    '同尾组数:Q'
                ]
            )

            text = base.mark_text(
                align='center',
                baseline='bottom',
                dy=-5
            ).encode(
                x='期号:O',
                y='最大同尾数:Q',
                text='同尾组数:Q',
                color=alt.value('black')
            )

            st.altair_chart(
                (points + text).properties(title='同尾类型历史趋势'),
                use_container_width=True)
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
        st.subheader("大小变化趋势")
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

        # 假设 filtered_data 已经定义，并且包含 "期号" 和 "重号" 列

        # 创建 DataFrame，包含 "期号" 和 "重号"
        same_numbers_df = pd.DataFrame({
            '期号': filtered_data['期号'],
            '重号数量': filtered_data['重号']
        })

        # 设置 Y 轴 domain 为 0 到 4
        chart = alt.Chart(same_numbers_df).mark_line().encode(
            x=alt.X('期号:O', title='期号', axis=alt.Axis(labelAngle=-45, labelFontSize=10)),
            y=alt.Y('重号数量:Q', title='重号数量',
                    axis=alt.Axis(format='d'),  # 确保y轴显示整数
                    scale=alt.Scale(domain=[0, 4], nice=False)),  # 设置 Y 轴 domain 并关闭 nice
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
        # **添加数据标签**
        text = chart.mark_text(
            align='center',
            baseline='bottom',
            dy=-5,  # **调整标签位置**
            fontSize=10,  # **设置字体大小**
            #color='black'  # **标签颜色**
        ).encode(
            text=alt.Text('孤号数量:Q', format='.0f')  # **数值格式化为整数**
        )

        # **显示折线图**
        st.altair_chart(chart, use_container_width=True)

    col1,col2 = st.columns(2)
    with col1:
        st.subheader("红球和值分析")

        # **创建 DataFrame，包含 "期号" 和 "和值"**
        sum_numbers_df = pd.DataFrame({
            '期号': filtered_data['期号'],  # 期号
            '和值': filtered_data['和值']  # 已计算的和值
        })

        # **创建 Altair 折线图**
        chart = alt.Chart(sum_numbers_df).mark_line().encode(
            x=alt.X('期号:O', title='期号', axis=alt.Axis(labelAngle=-45, labelFontSize=10)),  # **X 轴，期号**
            y=alt.Y('和值:Q', title='和值', axis=alt.Axis(format='d')),  # **Y 轴，整数**
            color=alt.value('#FF5733'),  # **设定折线颜色为橙色**
            tooltip=['期号', '和值']  # **鼠标悬停显示数据**
        ).properties(
            title='红球和值趋势图',  # **图表标题**
            width=800,  # **宽度**
            height=300  # **高度**
        )
        # **添加数据标签**
        text = chart.mark_text(
            align='center',
            baseline='bottom',
            dy=-5,  # **调整标签位置**
            fontSize=10,  # **设置字体大小**
            color='black'  # **标签颜色**
        ).encode(
            text=alt.Text('和值:Q', format='.0f')  # **数值格式化为整数**
        )

        # **显示折线图**
        st.altair_chart(chart + text, use_container_width=True)

    with col2:
        st.subheader("红球跨度分析")

        # **创建 DataFrame，包含 "期号" 和 "跨度"**
        range_numbers_df = pd.DataFrame({
            '期号': filtered_data['期号'],  # 期号
            '跨度': filtered_data['跨度']  # 已计算的跨度
        })

        # **创建 Altair 折线图**
        chart = alt.Chart(range_numbers_df).mark_line().encode(
            x=alt.X('期号:O', title='期号', axis=alt.Axis(labelAngle=-45, labelFontSize=10)),  # **X 轴，期号**
            y=alt.Y('跨度:Q', title='跨度', axis=alt.Axis(format='d')),  # **Y 轴，整数**
            color=alt.value('#1E90FF'),  # **设定折线颜色为蓝色**
            tooltip=['期号', '跨度']  # **鼠标悬停显示数据**
        ).properties(
            title='红球跨度趋势图',  # **图表标题**
            width=800,  # **宽度**
            height=300  # **高度**
        )
        # **添加数据标签**
        text = chart.mark_text(
            align='center',
            baseline='bottom',
            dy=-5,  # **调整标签位置**
            fontSize=10,  # **设置字体大小**
            color='black'  # **标签颜色**
        ).encode(
            text=alt.Text('跨度:Q', format='.0f')  # **数值格式化为整数**
        )

        # **显示折线图**
        st.altair_chart(chart + text, use_container_width=True)

    col1,col2 = st.columns(2)
    with col1:
        st.subheader("红球 AC 值分析")

        # **创建 DataFrame，包含 "期号" 和 "AC"**
        ac_numbers_df = pd.DataFrame({
            '期号': filtered_data['期号'],  # 期号
            'AC 值': filtered_data['AC']  # 已计算的 AC 值
        })

        # **创建 Altair 折线图**
        chart = alt.Chart(ac_numbers_df).mark_line().encode(
            x=alt.X('期号:O', title='期号', axis=alt.Axis(labelAngle=-45, labelFontSize=10)),  # **X 轴，期号**
            y=alt.Y('AC 值:Q', title='AC 值', axis=alt.Axis(format='d'),
                    scale=alt.Scale(domain=[4, 12])), # **Y 轴范围 4 到 12**)
            #color=alt.value('#FF5733'),  # **设定折线颜色为橙色**
            tooltip=['期号', 'AC 值']  # **鼠标悬停显示数据**
        ).properties(
            title='红球 AC 值趋势图',  # **图表标题**
            width=800,  # **宽度**
            height=300  # **高度**
        )

        # **添加数据标签**
        text = chart.mark_text(
            align='center',
            baseline='bottom',
            dy=-5,  # **调整标签位置**
            fontSize=10,  # **设置字体大小**
            color='black'  # **标签颜色**
        ).encode(
            text=alt.Text('AC 值:Q', format='.0f')  # **数值格式化为整数**
        )

        # **显示折线图**
        st.altair_chart(chart + text, use_container_width=True)

    with col2:
        st.subheader("🔥 热门号码对")

        # **获取热门号码对**
        freq_df = analyze_top_companion_pairs(filtered_data, top_n=10)

        # **绘制柱状图**
        bars = alt.Chart(freq_df).mark_bar().encode(
            x=alt.X('号码对:O', title='热门号码对', sort='-y',
                    axis=alt.Axis(labelAngle=-45, labelOverlap=False, labelFontSize=10)),
            y=alt.Y('出现次数:Q', title='出现次数',axis=alt.Axis(format='d')),
            #color=alt.Color('出现次数:Q', scale=alt.Scale(scheme='blues'), legend=None),
            tooltip=['号码对', '出现次数', alt.Tooltip('百分比:Q', format=".1%")]
        )

        # **添加百分比文本**
        text = bars.mark_text(
            align='center',
            baseline='bottom',
            dy=-10
        ).encode(
            text=alt.Text('百分比:Q', format=".1%")
        )

        # **显示图表**
        st.altair_chart(bars + text, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🔥热门号码三元组")

        # **获取热门号码三元组**
        freq_df = analyze_top_triples(filtered_data, top_n=10)

        # **绘制柱状图**
        bars = alt.Chart(freq_df).mark_bar(color='red').encode(
            x=alt.X('号码三元组:O', title='热门号码三元组', sort='-y',
                    axis=alt.Axis(labelAngle=-45, labelOverlap=False, labelFontSize=10)),
            y=alt.Y('出现次数:Q', title='出现次数', axis=alt.Axis(format='d')),
            tooltip=['号码三元组', '出现次数', alt.Tooltip('百分比:Q', format=".1%")]
        )

        # **添加百分比文本**
        text = bars.mark_text(
            align='center',
            baseline='bottom',
            dy=-10
        ).encode(
            text=alt.Text('百分比:Q', format=".1%")
        )

        # **显示图表**
        st.altair_chart(bars + text, use_container_width=True)

with (tab2):
    from funcs.ball_filter import convert_to_single_bets, parse_bet,convert_bets,check_winning,analyze_winning

    def filter_bets():
        """根据筛选条件过滤投注方案"""
        if 'analysis_results' not in st.session_state:
            st.warning("请先分析投注方案")
            return

        filtered_results = []
        filter_info = []  # 用于存储筛选条件的列表
        filter_results = {}  # 用于存储每个筛选条件的结果

        for result in st.session_state.analysis_results:
            # 去除分析结果部分
            if "(" in result:
                bet_str = result.split("(")[0].strip()
            else:
                bet_str = result.strip()

            # 解析红球和篮球部分
            if "+" in bet_str:
                red_balls_str, blue_balls_str = bet_str.split("+")
                red_balls = [int(ball) for ball in red_balls_str.split(",") if ball.strip()]  # 过滤掉空字符串
                blue_balls = [int(blue_balls_str.strip())]  # 提取篮球部分，并去除多余空格
            else:
                red_balls = [int(ball) for ball in bet_str.split(",") if ball.strip()]  # 过滤掉空字符串
                blue_balls = []  # 如果没有篮球，则设置为空列表

            # 筛选条件检查
            hot_cold_match = True
            odd_even_match = True
            size_match = True
            same_nums_match = True
            neigh_nums_match = True
            sep_nums_match = True
            consecutive_match = True
            same_tail_match = True
            sum_match = True
            span_match = True
            two_consecutive_match = True
            three_consecutive_match = True
            skip_nums_match = True
            two_skip_nums_match = True
            three_skip_nums_match = True
            two_tail_match = True
            three_tail_match = True
            zone_match = True

            # 热号筛选
            if st.session_state.hot_nums_filter:
                hot_count = sum(1 for ball in red_balls if ball in hot_red)  # 假设 hot_red 已定义
                hot_cold_match = st.session_state.hot_nums[0] <= hot_count <= st.session_state.hot_nums[1]
                if "hot_nums" not in filter_results:
                    filter_results["hot_nums"] = hot_cold_match

            # 冷号筛选
            if st.session_state.cold_nums_filter:
                cold_count = sum(1 for ball in red_balls if ball in cold_red)  # 假设 cold_red 已定义
                hot_cold_match = hot_cold_match and (
                            st.session_state.cold_nums[0] <= cold_count <= st.session_state.cold_nums[1])
                if "cold_nums" not in filter_results:
                    filter_results["cold_nums"] = hot_cold_match

            # 奇偶筛选
            if st.session_state.odd_even_filter:
                odd_count_range = st.session_state.odd_count  # 将滑块的值存储在 odd_count_range 中
                odd_count_actual = sum(1 for ball in red_balls if ball % 2 != 0)  # 计算实际奇数个数
                odd_even_match = odd_count_actual >= odd_count_range[0] and odd_count_actual <= odd_count_range[1]
                if "odd_count" not in filter_results:
                    filter_results["odd_count"] = odd_even_match

            # 大小筛选
            if st.session_state.small_big_filter:
                small_count = sum(1 for ball in red_balls if 1 <= ball <= 16)
                small_big_match = small_count >= st.session_state.small_big[0] and small_count <= \
                                  st.session_state.small_big[1]
                if "small_big" not in filter_results:
                    filter_results["small_big"] = small_big_match

            # 重号筛选
            if st.session_state.same_nums_filter:
                last_draw_reds = [latest_draw[f'红球{i}'] for i in range(1, 7)]  # 假设 latest_draw 已定义
                same_count = sum(1 for ball in red_balls if ball in last_draw_reds)
                same_nums_match = same_count >= st.session_state.same_nums[0] and same_count <= \
                                  st.session_state.same_nums[1]
                if "same_nums" not in filter_results:
                    filter_results["same_nums"] = same_nums_match

            # 邻号筛选
            if st.session_state.neigh_nums_filter:
                last_draw_reds = [latest_draw[f'红球{i}'] for i in range(1, 7)]  # 假设 latest_draw 已定义
                neigh_count = sum(1 for ball in red_balls if ball + 1 in last_draw_reds or ball - 1 in last_draw_reds)
                neigh_nums_match = neigh_count >= st.session_state.neigh_nums[0] and neigh_nums_match <= \
                                   st.session_state.neigh_nums[1]
                if "neigh_nums" not in filter_results:
                    filter_results["neigh_nums"] = neigh_nums_match

            # 孤号筛选
            if st.session_state.sep_nums_filter:
                last_draw_reds = [latest_draw[f'红球{i}'] for i in range(1, 7)]  # 假设 latest_draw 已定义
                sep_count = sum(
                    1 for ball in red_balls if ball + 1 not in last_draw_reds and ball - 1 not in last_draw_reds)
                sep_nums_match = sep_count >= st.session_state.sep_nums[0] and sep_nums_match <= \
                                 st.session_state.sep_nums[1]
                if "sep_nums" not in filter_results:
                    filter_results["sep_nums"] = sep_nums_match

            # 连号筛选
            if st.session_state.consecutive_filter:
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
                max_consecutive = max(consecutive_counts.keys()) if consecutive_counts else 0
                consecutive_match = max_consecutive <= st.session_state.consecutive_count
                if "consecutive_count" not in filter_results:
                    filter_results["consecutive_count"] = consecutive_match

            # 同尾号筛选
            if st.session_state.same_tail_filter:
                tails = [ball % 10 for ball in red_balls]
                tail_counts = {}
                for tail in tails:
                    tail_counts[tail] = tail_counts.get(tail, 0) + 1
                max_same_tail_count = max(tail_counts.values()) if tail_counts else 0
                same_tail_match = max_same_tail_count <= st.session_state.max_same_tail
                if "max_same_tail" not in filter_results:
                    filter_results["max_same_tail"] = same_tail_match

            # 和值筛选
            if st.session_state.sum_filter:
                red_sum = sum(red_balls)
                sum_match = st.session_state.sum_range[0] <= red_sum <= st.session_state.sum_range[1]
                if "sum_range" not in filter_results:
                    filter_results["sum_range"] = sum_match

            # 跨度筛选
            if st.session_state.span_filter:
                red_span = max(red_balls) - min(red_balls)
                span_match = st.session_state.span_range[0] <= red_span <= st.session_state.span_range[1]
                if "span_range" not in filter_results:
                    filter_results["span_range"] = span_match

            # 二连组数筛选
            if st.session_state.two_consecutive_filter:
                red_balls.sort()
                two_consecutive_groups = 0
                i = 0
                while i < len(red_balls) - 1:
                    if red_balls[i + 1] == red_balls[i] + 1:
                        two_consecutive_groups += 1
                        i += 2  # 跳过已计入二连组的两个数
                    else:
                        i += 1
                two_consecutive_match = st.session_state.two_consecutive_count[0] <= two_consecutive_groups <= \
                                        st.session_state.two_consecutive_count[1]
                if "two_consecutive_count" not in filter_results:
                    filter_results["two_consecutive_count"] = two_consecutive_match

            # 三连组数筛选
            if st.session_state.three_consecutive_filter:
                red_balls.sort()
                three_consecutive_groups = 0
                i = 0
                while i < len(red_balls) - 2:
                    if red_balls[i + 2] == red_balls[i + 1] + 1 and red_balls[i + 1] == red_balls[i] + 1:
                        three_consecutive_groups += 1
                        i += 3  # 跳过已计入三连组的三个数
                    else:
                        i += 1
                three_consecutive_match = st.session_state.three_consecutive_count[
                                              0] <= three_consecutive_groups <= \
                                          st.session_state.three_consecutive_count[1]
                if "three_consecutive_count" not in filter_results:
                    filter_results["three_consecutive_count"] = three_consecutive_match

            # 跳号个数筛选
            if st.session_state.skip_nums_filter:
                skip_count = 0
                for i in range(1, len(red_balls)):
                    if abs(red_balls[i] - red_balls[i - 1]) == 2:  # 修改跳号规则
                        skip_count += 1
                skip_nums_match = st.session_state.skip_nums[0] <= skip_count <= st.session_state.skip_nums[1]
                if "skip_nums" not in filter_results:
                    filter_results["skip_nums"] = skip_nums_match

            # 二跳组数筛选
            if st.session_state.two_skip_nums_filter:
                red_balls.sort()
                two_skip_groups = 0
                i = 0
                while i < len(red_balls) - 1:
                    if abs(red_balls[i + 1] - red_balls[i]) == 2:
                        two_skip_groups += 1
                        i += 2  # 跳过已计入二跳组的两个数
                    else:
                        i += 1
                two_skip_nums_match = st.session_state.two_skip_nums[0] <= two_skip_groups <= \
                                      st.session_state.two_skip_nums[1]
                if "two_skip_nums" not in filter_results:
                    filter_results["two_skip_nums"] = two_skip_nums_match

            # 三跳组数筛选
            if st.session_state.three_skip_nums_filter:
                red_balls.sort()
                three_skip_groups = 0
                i = 0
                while i < len(red_balls) - 2:  # 修改循环条件
                    if red_balls[i + 2] - red_balls[i + 1] == 2 and red_balls[i + 1] - red_balls[
                        i] == 2:  # 修改三跳号规则
                        three_skip_groups += 1
                        i += 3  # 跳过已计入三跳组的三个数
                    else:
                        i += 1
                three_skip_nums_match = st.session_state.three_skip_nums[0] <= three_skip_groups <= \
                                        st.session_state.three_skip_nums[1]
                if "three_skip_nums" not in filter_results:
                    filter_results["three_skip_nums"] = three_skip_nums_match

            # 二尾组数筛选
            if st.session_state.two_tail_filter:
                two_tail_groups = 0
                tail_counts = {}
                for ball in red_balls:
                    tail = ball % 10
                    tail_counts[tail] = tail_counts.get(tail, 0) + 1
                for count in tail_counts.values():
                    if count >= 2:
                        two_tail_groups += 1
                two_tail_match = st.session_state.two_tail_count[0] <= two_tail_groups <= \
                                 st.session_state.two_tail_count[1]
                if "two_tail_count" not in filter_results:
                    filter_results["two_tail_count"] = two_tail_match

            # 三尾组数筛选
            if st.session_state.three_tail_filter:
                three_tail_groups = 0
                tail_counts = {}
                for ball in red_balls:
                    tail = ball % 10
                    tail_counts[tail] = tail_counts.get(tail, 0) + 1
                for count in tail_counts.values():
                    if count >= 3:
                        three_tail_groups += 1
                three_tail_match = st.session_state.three_tail_count[0] <= three_tail_groups <= \
                                   st.session_state.three_tail_count[1]
                if "three_tail_count" not in filter_results:
                    filter_results["three_tail_count"] = three_tail_match

            # 区域号码数筛选
            if st.session_state.zone_filter:
                zone1_count = 0
                zone2_count = 0
                zone3_count = 0
                for ball in red_balls:
                    if 1 <= ball <= 11:
                        zone1_count += 1
                    elif 12 <= ball <= 22:
                        zone2_count += 1
                    elif 23 <= ball <= 33:
                        zone3_count += 1
                zone_match = (
                        st.session_state.zone1_count[0] <= zone1_count <= st.session_state.zone1_count[1] and
                        st.session_state.zone2_count[0] <= zone2_count <= st.session_state.zone2_count[1] and
                        st.session_state.zone3_count[0] <= zone3_count <= st.session_state.zone3_count[1]
                )
                # 确保初始化 filter_results["zone_count"]
                if "zone_count" not in filter_results:
                    filter_results["zone_count"] = zone_match
                # 确保初始化 st.session_state.zone_count
                st.session_state.zone_count = zone_match

            if (hot_cold_match and odd_even_match and size_match and same_nums_match and
                    neigh_nums_match and sep_nums_match and consecutive_match and same_tail_match and
                    sum_match and span_match and two_consecutive_match and three_consecutive_match and skip_nums_match and two_skip_nums_match and three_skip_nums_match and two_tail_match and three_tail_match and zone_match):
                filtered_results.append(result)

        st.session_state.filtered_results = filtered_results
        if 'filtered_results' in st.session_state:
            all_bets_text = (
                    f"总投注数: {len(st.session_state.analysis_results)}\n"
                    f"筛选条件：{', '.join([f'{k}:{st.session_state[k]}' for k in filter_results.keys()])}\n"
                    f"筛选后注数: {len(st.session_state.filtered_results)}\n\n"
                    + "\n".join(st.session_state.filtered_results)
            )
            st.session_state.all_bets_text = all_bets_text


    def analyze_bets():
        """分析投注方案"""
        bets_text = st.session_state.bets_text
        bets = []
        analysis_results = []
        total_bets = 0
        unique_bets = set()  # 添加一个集合，用于存储唯一的投注组合

        for line in bets_text.splitlines():
            if line.strip():
                red_dan, red_tuo, blue_dan, blue_tuo = parse_bet(line.strip())
                single_bets = convert_to_single_bets(red_dan, red_tuo, blue_dan, blue_tuo)
                total_bets += len(single_bets)
                bets.extend(single_bets)

        for red_balls, blue_balls in bets:  # 从 single_bets 中提取红球和篮球
            bet_tuple = tuple(sorted(red_balls)), tuple(sorted(blue_balls))  # 创建元组用于判断
            if bet_tuple not in unique_bets:  # 检查投注组合是否已经存在
                unique_bets.add(bet_tuple)  # 将投注组合添加到集合中
                if blue_balls:
                    analysis_results.append(
                        f"{','.join(map(str, red_balls))}+{','.join(map(str, blue_balls))} "
                    )
                else:
                    analysis_results.append(
                        f"{','.join(map(str, red_balls))} "
                    )

        # 创建包含所有投注结果和总投注数的字符串
        all_bets_text = f"总投注数: {total_bets}\n" + "\n".join(analysis_results)

        # 将结果存储在 session_state 中
        st.session_state.all_bets_text = all_bets_text
        st.session_state.analysis_results = analysis_results

    col1, col2 = st.columns(2)  # 创建两列布局

    with col1:
        st.subheader("选号工具")

        # 初始化会话状态
        if 'selected_red_balls' not in st.session_state:
            st.session_state.selected_red_balls = []
        if 'selected_blue_balls' not in st.session_state:
            st.session_state.selected_blue_balls = []

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
            placeholder.markdown(selected_numbers_html, unsafe_allow_html=True)  # 修改后的代码。


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

    with col2:
        st.subheader("选号方案")

        if 'bets_text' not in st.session_state:
            st.session_state.bets_text = ""  # 初始化 bets_text

        st.text_area(label="你的方案（每行一个方案，支持单注，复式和胆拖）:",
                     value=st.session_state.bets_text, height=200,
                     key="bets_text")  # 更新 text_area

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


    st.button("分析投注", on_click=analyze_bets)

    col1, col2 = st.columns(2)  # 创建两列布局

    with col1:
        # 初始化 session_state
        if "all_bets_text" not in st.session_state:
            st.session_state.all_bets_text = "请增加你的投注方案"

        # 显示投注结果
        st.text_area("投注结果", value=st.session_state.all_bets_text, height=300)
        st.button("选号方案筛选", on_click=filter_bets)

    with col2:
        if 'filtered_results' not in st.session_state:
            st.session_state.filtered_results = []

        if 'simplified_bets_area' not in st.session_state:
            st.session_state.simplified_bets_area = "请转化你的投注结果"

        # 如果 session_state 中存在表格数据，则显示表格
        if 'winning_table_data' in st.session_state:

            st.table(st.session_state.winning_table_data)
            st.write(f"总投注数: {st.session_state.winning_total_bets}")
            st.write(f"总奖金：{st.session_state.winning_total_amount}")
       # st.selectbox("选择开奖期号:", issue_numbers, index=5)
        st.button("投注对奖", on_click=analyze_winning)

with tab3:
    st.subheader("全量筛选")


with tab4:
    st.subheader("历史开奖数据")
    historical_data = load_historical_data(100)
    historical_data=historical_data.style.set_properties(**{'text-align': 'center'})
 # 获取所有数据

    st.dataframe(historical_data, width=1000, height=500) # Display historical data in a table