from itertools import combinations
from collections import Counter
import pandas as pd
import streamlit as st

@st.cache_data
def analyze_top_companion_pairs(id_data, red_cols, top_n=5):
    """
    计算红球号码对的伴随出现频率，并返回前 top_n 热门号码对的 DataFrame。
    """
    if id_data is None or len(id_data) == 0:
        return pd.DataFrame(columns=['号码对', '出现次数', '百分比'])

    total_issues = len(id_data)
    frequency_dict = Counter()
    
    # Extract values as a NumPy array for fast access
    vals = id_data[red_cols].values

    for row in vals:
        try:
            # Filter NaNs and sort
            balls = sorted([int(v) for v in row if pd.notna(v)])
            if len(balls) < 2:
                continue
            for pair in combinations(balls, 2):
                frequency_dict[f"{pair[0]}-{pair[1]}"] += 1
        except:
            continue

    if not frequency_dict:
        return pd.DataFrame(columns=['号码对', '出现次数', '百分比'])

    freq_df = pd.DataFrame([
        {'号码对': key, '出现次数': value, '百分比': value / total_issues if total_issues else 0}
        for key, value in frequency_dict.items()
    ])

    return freq_df.sort_values(by='出现次数', ascending=False).head(top_n)

@st.cache_data
def analyze_top_triples(id_data, red_cols, top_n=5):
    """
    计算红球三个号码组合的伴随出现频率，并返回前 top_n 热门三元组。
    """
    if id_data is None or len(id_data) == 0:
        return pd.DataFrame(columns=['号码三元组', '出现次数', '百分比'])

    total_issues = len(id_data)
    frequency_dict = Counter()

    vals = id_data[red_cols].values

    for row in vals:
        try:
            balls = sorted([int(v) for v in row if pd.notna(v)])
            if len(balls) < 3:
                continue
            for triple in combinations(balls, 3):
                frequency_dict[f"{triple[0]}-{triple[1]}-{triple[2]}"] += 1
        except:
            continue

    if not frequency_dict:
        return pd.DataFrame(columns=['号码三元组', '出现次数', '百分比'])

    freq_df = pd.DataFrame([
        {'号码三元组': key, '出现次数': value, '百分比': value / total_issues if total_issues else 0}
        for key, value in frequency_dict.items()
    ])

    return freq_df.sort_values(by='出现次数', ascending=False).head(top_n)