from itertools import combinations
import pandas as pd

def analyze_top_companion_pairs(filtered_data, top_n=5):
    """
    计算红球号码对的伴随出现频率，并返回前 top_n 热门号码对的 DataFrame。
    """
    # **提取所有红球列**
    red_ball_columns = ["红球1", "红球2", "红球3", "红球4", "红球5", "红球6"]

    # **检查数据是否为空**
    if filtered_data.empty:
        st.warning("❌ 警告: 历史数据为空，无法分析号码对频率。")
        return pd.DataFrame(columns=['号码对', '出现次数', '百分比'])

    # **获取所有可能的号码**
    all_numbers = sorted(set(
        int(num) for col in red_ball_columns
        for num in filtered_data[col].dropna().astype(int)  # 确保转换为整数
    ))

    # **检查是否获取到了数据**
    if not all_numbers:
        st.warning("⚠️ 无法获取红球号码，请检查数据格式。")
        return pd.DataFrame(columns=['号码对', '出现次数', '百分比'])

    # **生成所有可能的号码对**
    pairs = list(combinations(all_numbers, 2))

    # **计算号码对的伴随出现频率**
    total_issues = len(filtered_data)
    frequency_dict = {f"{num1}-{num2}": 0 for num1, num2 in pairs}

    for _, row in filtered_data.iterrows():
        red_balls = set(row[red_ball_columns].dropna().astype(int))
        for num1, num2 in pairs:
            if num1 in red_balls and num2 in red_balls:
                frequency_dict[f"{num1}-{num2}"] += 1

    # **转换为 DataFrame 并排序**
    freq_df = pd.DataFrame([
        {'号码对': key, '出现次数': value, '百分比': value / total_issues if total_issues else 0}
        for key, value in frequency_dict.items()
    ])

    freq_df = freq_df.sort_values(by='出现次数', ascending=False).head(top_n)  # 仅取前 top_n

    return freq_df  # **返回 DataFrame**

def analyze_top_triples(filtered_data, top_n=5):
    """
    计算红球三个号码组合的伴随出现频率，并返回前 top_n 热门三元组。
    """
    # **提取所有红球列**
    red_ball_columns = ["红球1", "红球2", "红球3", "红球4", "红球5", "红球6"]

    # **检查数据是否为空**
    if filtered_data.empty:
        st.warning("❌ 警告: 历史数据为空，无法分析号码组合频率。")
        return pd.DataFrame(columns=['号码三元组', '出现次数', '百分比'])

    # **获取所有可能的号码**
    all_numbers = sorted(set(
        int(num) for col in red_ball_columns
        for num in filtered_data[col].dropna().astype(int)  # 确保转换为整数
    ))

    # **检查是否获取到了数据**
    if not all_numbers:
        st.warning("⚠️ 无法获取红球号码，请检查数据格式。")
        return pd.DataFrame(columns=['号码三元组', '出现次数', '百分比'])

    # **生成所有可能的三元组**
    triplets = list(combinations(all_numbers, 3))

    # **计算号码三元组的伴随出现频率**
    total_issues = len(filtered_data)
    frequency_dict = {f"{num1}-{num2}-{num3}": 0 for num1, num2, num3 in triplets}

    for _, row in filtered_data.iterrows():
        red_balls = set(row[red_ball_columns].dropna().astype(int))  # 获取该期的红球号码
        for num1, num2, num3 in triplets:
            if num1 in red_balls and num2 in red_balls and num3 in red_balls:
                frequency_dict[f"{num1}-{num2}-{num3}"] += 1

    # **转换为 DataFrame 并排序**
    freq_df = pd.DataFrame([
        {'号码三元组': key, '出现次数': value, '百分比': value / total_issues if total_issues else 0}
        for key, value in frequency_dict.items()
    ])

    freq_df = freq_df.sort_values(by='出现次数', ascending=False).head(top_n)  # 仅取前 top_n

    return freq_df  # **返回 DataFrame**