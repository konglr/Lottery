import streamlit as st
import pandas as pd
import openpyxl
import numpy as np
import io

def process_lottery_data(df, lottery_type):
    """根据彩票类型处理数据"""
    if lottery_type == "双色球":
        df['前区号码'] = df['前区号码'].apply(lambda x: [int(num) for num in str(x).split()])
        df['后区号码'] = df['后区号码'].apply(lambda x: [int(num) for num in str(x).split()])
    elif lottery_type == "快乐8":
        df['前区号码'] = df['前区号码'].apply(lambda x: [int(num) for num in str(x).split()])
    return df

def calculate_number_frequency(df, lottery_type):
    """计算号码出现次数和平均出现频率"""
    all_numbers = []
    if lottery_type == "双色球":
        for _, row in df.iterrows():
            all_numbers.extend(row['前区号码'])
            all_numbers.extend(row['后区号码'])
        num_range = range(1, 33)
    elif lottery_type == "快乐8":
        for _, row in df.iterrows():
            all_numbers.extend(row['前区号码'])
        num_range = range(1, 80)

    frequency = pd.Series(all_numbers).value_counts().sort_index()
    total_draws = len(df) * (6 if lottery_type == "双色球" else 20)  # 双色球每期6个号码，快乐8每期20个号码
    frequency_df = pd.DataFrame(index=['出现次数', '出现频率'], columns=num_range).fillna(0)
    for num, count in frequency.items():
        frequency_df.loc['出现次数', num] = count
        frequency_df.loc['出现频率', num] = round(count / total_draws,3) # 保留小数点后3位
    return frequency_df

def main():
    st.title(":orange[Lottery 彩票开奖]")    # 使用 CSS 设置 st.title() 样式
    st.markdown(
        """
        <style>
        .title {
            text-align: left;
            color: #FF5733; /* 设置颜色为橙红色 */
            font-size: 0.8 em; /* 设置文字大小 */
            font-family: 'Georgia', serif; /* 设置字体 */
            font-weight: bold; /* 设置粗体 */
            text-shadow: 1px 1px 2px #888888; /* 设置文字阴影 */
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # 显示标题
    st.markdown('<h5 class="title">Lottery 彩票开奖</h5>', unsafe_allow_html=True)


    # 左侧边栏
    lottery_type = st.sidebar.selectbox("选择彩票类型", ["双色球", "快乐8"])
    display_issues = st.sidebar.selectbox("选择显示期数", [10, 20, 30, 50])

    # 读取Excel数据
    if lottery_type == "双色球":
        df = pd.read_excel("双色球开奖情况.xlsx")
    elif lottery_type == "快乐8":
        df = pd.read_excel("快乐8开奖情况.xlsx")

    # 移除期号中的千分位分隔符
    df['期号'] = df['期号'].astype(str).str.replace(',', '')

    # 数据预处理
    df = process_lottery_data(df, lottery_type)
    df = df.head(display_issues)

    # 显示开奖记录
    st.subheader(":blue[开奖记录]")
    if lottery_type == "双色球":
        st.dataframe(df[['期号', '开奖日期', 'WeekDay', '前区号码', '后区号码']])
    elif lottery_type == "快乐8":
        # 使用方法一：增加表格宽度
        st.markdown(
            """
            <style>
            .stContainer {
                width: 100% !important;
                height: 500px; /* 调整行高度 */
                margin: 0 auto;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )
        with st.container():
            st.dataframe(df[['期号', '开奖日期', '前区号码']], use_container_width=True)


    # 计算号码出现次数
    frequency_df = calculate_number_frequency(df, lottery_type)

    # 显示号码统计信息
    st.subheader("号码统计")
    column_config = {col: st.column_config.NumberColumn(width=None) for col in frequency_df.columns}
    st.dataframe(frequency_df, use_container_width=True, column_config=column_config)

    # 绘制柱状图
    st.subheader("号码出现次数统计图")
    st.bar_chart(frequency_df.loc['出现次数'])

    # 下载按钮
    st.sidebar.subheader("下载数据")
    with open("双色球开奖情况.xlsx", "rb") as f:
        bytes_data = f.read()
    st.sidebar.download_button(label="下载双色球数据", data=bytes_data, file_name="双色球开奖情况.xlsx", mime="application/vnd.ms-excel")

    with open("快乐8开奖情况.xlsx", "rb") as f:
        bytes_data = f.read()
    st.sidebar.download_button(label="下载快乐8数据", data=bytes_data, file_name="快乐8开奖情况.xlsx", mime="application/vnd.ms-excel")

if __name__ == "__main__":
    main()