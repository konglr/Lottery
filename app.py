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
        num_range = range(1, 81)

    frequency = pd.Series(all_numbers).value_counts().sort_index()
    total_draws = len(df) * (6 if lottery_type == "双色球" else 20)  # 双色球每期6个号码，快乐8每期20个号码
    frequency_df = pd.DataFrame(index=['出现次数', '出现频率'], columns=num_range).fillna(0)
    for num, count in frequency.items():
        frequency_df.loc['出现次数', num] = count
        frequency_df.loc['出现频率', num] = round(count / total_draws,3) # 保留小数点后3位
    return frequency_df

def display_lottery_data(df, lottery_type):
    """显示彩票数据"""
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

def main():
    st.title(":orange[Lottery 彩票开奖]")
    st.markdown(
        """
        <style>
        .title {
            text-align: left;
            color: #FF5733;
            font-size: 0.8 em;
            font-family: 'Georgia', serif;
            font-weight: bold;
            text-shadow: 1px 1px 2px #888888;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.markdown('<h5 class="title">Lottery 彩票开奖</h5>', unsafe_allow_html=True)

    # 创建 Tabs
    tab1, tab2 = st.tabs(["双色球", "快乐8"])

    # 左侧边栏
    display_issues = st.sidebar.selectbox("选择显示期数", [10, 20, 30, 50])

    # 双色球 Tab
    with tab1:
        df_ssq = pd.read_excel("双色球开奖情况.xlsx")
        df_ssq['期号'] = df_ssq['期号'].astype(str).str.replace(',', '')
        df_ssq = process_lottery_data(df_ssq, "双色球")
        df_ssq = df_ssq.head(display_issues)
        display_lottery_data(df_ssq, "双色球")

    # 快乐8 Tab
    with tab2:
        df_kl8 = pd.read_excel("快乐8开奖情况.xlsx")
        df_kl8['期号'] = df_kl8['期号'].astype(str).str.replace(',', '')
        df_kl8 = process_lottery_data(df_kl8, "快乐8")
        df_kl8 = df_kl8.head(display_issues)
        display_lottery_data(df_kl8, "快乐8")

    # 下载按钮
    st.sidebar.subheader("下载数据")
    with open("双色球开奖情况.xlsx", "rb") as f:
        bytes_data_ssq = f.read()
    st.sidebar.download_button(label="下载双色球数据", data=bytes_data_ssq, file_name="双色球开奖情况.xlsx", mime="application/vnd.ms-excel")

    with open("快乐8开奖情况.xlsx", "rb") as f:
        bytes_data_kl8 = f.read()
    st.sidebar.download_button(label="下载快乐8数据", data=bytes_data_kl8, file_name="快乐8开奖情况.xlsx", mime="application/vnd.ms-excel")

if __name__ == "__main__":
    main()