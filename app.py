import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from collections import Counter
import logging
import os
import itertools
from funcs.functions import analyze_top_companion_pairs, analyze_top_triples
import time
import json
import ast
from funcs.ai_helper import (
    load_renviron,
    get_brand_models,
    prepare_lottery_data_text,
    generate_ai_prediction,
    parse_ai_recommendations,
    format_ai_response
)
from funcs.ball_filter import calculate_morphology_score, get_morphology_report

# Load environment variables from .Renviron
load_renviron()

# --- Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('my_log_file.log')
    ]
)

from config import LOTTERY_CONFIG

# --- Helper Functions ---

@st.cache_data
def load_full_data(lottery_name):
    """Loads matching data for the selected lottery type (limited to 100 for performance)."""
    config = LOTTERY_CONFIG.get(lottery_name)
    if not config: return pd.DataFrame()
    file_path = config["data_file"]
    try:
        if not os.path.exists(file_path):
            logging.error(f"File not found: {file_path}")
            return pd.DataFrame()
            
        df = pd.read_csv(file_path)
        column_mapping = {}
        if 'issue' in df.columns: column_mapping['issue'] = '期号'
        if 'openTime' in df.columns: column_mapping['openTime'] = '开奖日期'
        if 'period' in df.columns: column_mapping['period'] = '期号'
        if 'saleMoney' in df.columns: column_mapping['saleMoney'] = '本期销售金额'
        if 'prizePoolMoney' in df.columns: column_mapping['prizePoolMoney'] = '奖池累计金额'
        if column_mapping: df = df.rename(columns=column_mapping)
        if '期号' in df.columns:
            # 统一处理期号：转为数值，处理大乐透 26019 这种格式
            df['期号'] = pd.to_numeric(df['期号'], errors='coerce').fillna(0).astype(int).astype(str)
            
            # 如果是开奖日期存在，优先按日期排序，确保最新开奖在首行
            if '开奖日期' in df.columns:
                df['开奖日期'] = pd.to_datetime(df['开奖日期'], errors='coerce')
                df = df.sort_values(['开奖日期', '期号'], ascending=[False, False])
            else:
                df = df.sort_values('期号', ascending=False)
            
        # Limit to 100 as the UI slider max is 100
        return df.head(100)
    except Exception as e:
        logging.error(f"Error loading {lottery_name}: {e}")
        return pd.DataFrame()

def get_filtered_data(df, limit=None):
    """Returns a slice of the data."""
    if df.empty: return df
    return df.head(limit) if limit else df

# --- Modular Chart Functions (Exact Styling from User) ---

def render_chart_red_freq(df, config):
    st.subheader("红球冷热分析")
    min_v, max_v = config['red_range']
    red_cols = [f"{config['red_col_prefix']}{i}" for i in range(1, config['red_count']+1)]
    if not all(c in df.columns for c in red_cols): return
    
    # Matching user logic: range(1, 34) for ssq etc.
    # We use config range but user specifically mentioned range(1, 34) for ssq UI alignment
    calc_range = range(min_v, max_v + 1)
    
    # Optimized calculation
    counts = df[red_cols].stack().value_counts()
    red_frequency = {i: int(counts.get(i, 0)) for i in calc_range}

    df_plot = pd.DataFrame({'号码': list(red_frequency.keys()), '出现次数': list(red_frequency.values())})
    df_plot = df_plot.sort_values('出现次数', ascending=False)
    total = df_plot['出现次数'].sum()
    df_plot['百分比'] = (df_plot['出现次数'] / total) if total > 0 else 0

    bars = alt.Chart(df_plot).mark_bar(color='red').encode(
        x=alt.X('号码:O', title='红球号码', sort='-y', axis=alt.Axis(labelAngle=-45, labelOverlap=False, labelFontSize=10)),
        y=alt.Y('出现次数:Q', title='出现次数'),
        color=alt.Color('出现次数:Q', scale=alt.Scale(scheme='reds'), legend=None),
        tooltip=['号码', '出现次数', alt.Tooltip('百分比', format=".1%")]
    )
    text = alt.Chart(df_plot).mark_text(dy=-10).encode(
        x=alt.X('号码:O', sort='-y'),
        y=alt.Y('出现次数:Q'),
        text=alt.Text('百分比:Q', format=".1%", formatType='number')
    )
    st.altair_chart((bars + text).properties(title='红球出现频率 (出现次数及百分比)', width=800, height=300), use_container_width=True)

    counts_sorted = sorted(red_frequency.items(), key=lambda x: x[1], reverse=True)
    hot_nums = [item[0] for item in counts_sorted[:6]]
    cold_nums = [item[0] for item in counts_sorted[-6:]]
    st.write(f"热门号码: {', '.join(map(str, hot_nums))}")
    st.write(f"冷门号码: {', '.join(map(str, cold_nums))}")

def render_chart_blue_freq(df, config):
    if not config['has_blue']: return
    st.subheader("蓝球冷热分析")
    
    # Identify blue ball columns
    blue_cols = []
    if config['blue_count'] == 1:
        # Fallback for SSQ/QLC/XQXC which might use '蓝球' or '篮球'
        col = config['blue_col_name'] if config['blue_col_name'] in df.columns else '蓝球'
        if col in df.columns: blue_cols.append(col)
        elif '篮球' in df.columns: blue_cols.append('篮球')
    else:
        # For lotteries like DLT (蓝球1, 蓝球2)
        base_name = config['blue_col_name']
        for i in range(1, config['blue_count'] + 1):
            col = f"{base_name}{i}"
            if col in df.columns:
                blue_cols.append(col)
            elif f"篮球{i}" in df.columns: # Fallback for naming variations
                blue_cols.append(f"篮球{i}")
                
    if not blue_cols: return

    # Optimized calculation
    counts = df[blue_cols].stack().value_counts()
    min_v, max_v = config['blue_range']
    blue_frequency = {i: int(counts.get(i, 0)) for i in range(min_v, max_v + 1)}

    df_plot = pd.DataFrame({'号码': list(blue_frequency.keys()), '出现次数': list(blue_frequency.values())})
    df_plot = df_plot.sort_values('出现次数', ascending=False)
    total = df_plot['出现次数'].sum()
    df_plot['百分比'] = (df_plot['出现次数'] / total) if total > 0 else 0

    bars = alt.Chart(df_plot).mark_bar(color='blue').encode(
        x=alt.X('号码:O', title='蓝球号码', sort='-y', axis=alt.Axis(labelAngle=-45, labelOverlap=False, labelFontSize=10)),
        y=alt.Y('出现次数:Q', title='出现次数'),
        tooltip=['号码', '出现次数', alt.Tooltip('百分比', format=".1%")]
    )
    text = alt.Chart(df_plot).mark_text(dy=-10).encode(
        x=alt.X('号码:O', sort='-y'),
        y=alt.Y('出现次数:Q'),
        text=alt.Text('百分比:Q', format=".1%", formatType='number')
    )
    st.altair_chart((bars + text).properties(title='蓝球出现频率 (出现次数及百分比)', width=800, height=300), use_container_width=True)
    
    counts_sorted = sorted(blue_frequency.items(), key=lambda x: x[1], reverse=True)
    hot = [item[0] for item in counts_sorted[:3]]
    cold = [item[0] for item in counts_sorted[-3:]]
    st.write(f"热门蓝球号码: {', '.join(map(str, hot))}")
    st.write(f"冷门蓝球号码: {', '.join(map(str, cold))}")

def render_chart_odd_even_ratio(df, config):
    st.subheader("奇偶比例分析")
    if '奇数' not in df.columns: 
        st.caption("暂无奇偶数据")
        return
        
    red_count = int(config['red_count'])
    # Dynamically generate all possible ratios based on red count
    # e.g., for 6 balls: 0:6, 1:5, 2:4, 3:3, 4:2, 5:1, 6:0
    possible_ratios = []
    ratio_counts = {}
    
    for i in range(red_count + 1): # 0 to red_count
        ratio_str = f"{i}:{red_count - i}"
        possible_ratios.append(ratio_str)
        ratio_counts[ratio_str] = 0
        
    for val in df['奇数']:
        if pd.isna(val): continue
        odds = int(val)
        evens = red_count - odds
        # Ensure we don't go out of bounds if data is weird
        if 0 <= odds <= red_count:
            ratio = f"{odds}:{evens}"
            ratio_counts[ratio] = ratio_counts.get(ratio, 0) + 1
            
    # Create DataFrame for plotting
    data_list = [{"奇偶比例": k, "出现次数": v} for k, v in ratio_counts.items()]
    df_plot = pd.DataFrame(data_list)
    
    # Sort by the logical order (0:N -> N:0) or by frequency? Usually logical is better for X-axis
    # Let's ensure the order matches possible_ratios
    df_plot['sort_key'] = df_plot['奇偶比例'].apply(lambda x: int(x.split(':')[0]))
    df_plot = df_plot.sort_values('sort_key')
    
    total = df_plot["出现次数"].sum()
    df_plot["百分比"] = df_plot["出现次数"] / total if total > 0 else 0
    
    chart = alt.Chart(df_plot).mark_bar().encode(
        x=alt.X('奇偶比例:O', title='奇偶比例 (奇:偶)', sort=possible_ratios, axis=alt.Axis(labelAngle=0)),
        y=alt.Y('出现次数:Q', title='出现次数'),
        color=alt.Color('出现次数:Q', legend=None),
        tooltip=['奇偶比例', '出现次数', alt.Tooltip('百分比:Q', format='.1%', title='百分比')]
    ).properties(title=f'奇偶比例分布 (红球总数: {red_count})', width=800, height=300)
    
    text = chart.mark_text(align='center', baseline='bottom', dy=-5).encode(text=alt.Text('百分比:Q', format='.1%'))
    st.altair_chart(chart + text, use_container_width=True)

def render_chart_odd_even_trend(df, config):
    st.subheader("奇数和偶数变化趋势")
    if '奇数' not in df.columns: return
    df_plot = df.copy()
    df_plot['偶数'] = config['red_count'] - df_plot['奇数']
    chart_data = df_plot.melt(id_vars=["期号"], value_vars=["奇数", "偶数"], var_name="类别", value_name="数量")
    chart_data.loc[chart_data['类别'] == '偶数', '数量'] *= -1
    chart = alt.Chart(chart_data).mark_bar().encode(
        x=alt.X('期号:O', title='期号', axis=alt.Axis(labelAngle=-45)),
        y=alt.Y('数量:Q', title='数量', scale=alt.Scale(domain=[-config['red_count'], config['red_count']])),
        color=alt.Color('类别:N', legend=None),
        tooltip=['期号', '类别', '数量']
    ).properties(title='奇偶号趋势', width=800, height=300)
    st.altair_chart(chart, use_container_width=True)

def _get_consecutive_cols(df, red_count):
    CN_KEYS = ["", "", "二", "三", "四", "五", "六", "七", "八", "九", "十", 
               "十一", "十二", "十三", "十四", "十五", "十六", "十七", "十八", "十九", "二十"]
    # Generate list based on red_count and check existence in df (limited to 20 per request_process_all_data.py)
    potential_cols = [f"{CN_KEYS[k]}连" for k in range(2, min(int(red_count) + 1, 21))]
    return [c for c in potential_cols if c in df.columns]

def render_chart_consecutive_dist(df, config):
    st.subheader("连号分布分析")
    cols = _get_consecutive_cols(df, config['red_count'])
    if not cols: 
        st.info("暂无连号统计数据")
        return
        
    counts = {c: df[c].sum() for c in cols}
    
    # 找到有数据出现的最大连号类型索引
    max_idx = -1
    for i, c in enumerate(cols):
        if counts[c] > 0:
            max_idx = i
            
    if max_idx == -1:
        st.info("本期所选数据范围内未发现连号组合")
        return
        
    # 截断列表，只保留到最大有数据的字段为止
    active_cols = cols[:max_idx + 1]
    
    df_plot = pd.DataFrame([{"连号": c, "出现次数": counts[c]} for c in active_cols])
    total = df_plot["出现次数"].sum()
    
    df_plot["百分比"] = df_plot["出现次数"] / total if total > 0 else 0
    df_plot['连号'] = pd.Categorical(df_plot['连号'], categories=active_cols, ordered=True)
    
    chart = alt.Chart(df_plot).mark_bar().encode(
        x=alt.X('连号:O', title='连号类型', axis=alt.Axis(labelAngle=0), sort=active_cols),
        y=alt.Y('出现次数:Q', title='出现总次数'),
        color=alt.Color('出现次数:Q', scale=alt.Scale(scheme='blues'), legend=None),
        tooltip=['连号', '出现次数', alt.Tooltip('百分比:Q', format='.1%', title='百分比')]
    ).properties(title='连号出现次数分布（历史统计）', width=800, height=300)
    
    text = chart.mark_text(align='center', baseline='bottom', dy=-5, color='black').encode(text=alt.Text('百分比:Q', format='.1%'))
    st.altair_chart(chart + text, use_container_width=True)

def render_chart_consecutive_trend(df, config):
    st.subheader("连号趋势分析")
    cols = _get_consecutive_cols(df, config['red_count'])
    if not cols: return
    
    trend_data = []
    # Show last 50 issues for clarity
    plot_df = df.head(50)
    
    for _, row in plot_df.iterrows():
        for c in cols:
            val = row.get(c, 0)
            if val > 0:
                trend_data.append({'期号': str(row['期号']), '连号类型': c, '连号组数': int(val)})
                
    if not trend_data:
        st.info("近期未发现连号组合")
        return
        
    df_trend = pd.DataFrame(trend_data)
    
    base = alt.Chart(df_trend).properties(width=800, height=300)
    
    points = base.mark_circle(size=60).encode(
        x=alt.X('期号:O', title='期号', axis=alt.Axis(labelAngle=-45)),
        y=alt.Y('连号类型:N', title='连号类型', sort=cols),
        color=alt.Color('连号类型:N', title='连号类型', scale=alt.Scale(domain=cols), legend=None),
        tooltip=['期号', '连号类型', '连号组数']
    )
    
    text = base.mark_text(
        align='center', 
        baseline='bottom', 
        dx=15
    ).encode(
        x='期号:O',
        y='连号类型:N',
        text='连号组数:Q',
        color=alt.value('black')
    )
    
    st.altair_chart((points + text).properties(title='红球连号趋势'), use_container_width=True)

def _get_jump_cols(df, red_count):
    CN_KEYS = ["", "", "二", "三", "四", "五", "六", "七", "八", "九", "十", 
               "十一", "十二", "十三", "十四", "十五", "十六", "十七", "十八", "十九", "二十"]
    # Generate list based on red_count (max jump length is equal to ball count)
    potential_cols = [f"{CN_KEYS[k]}跳" for k in range(2, min(int(red_count) + 1, 21))]
    return [c for c in potential_cols if c in df.columns]

def render_chart_jump_dist(df, config):
    st.subheader("跳号分布分析")
    cols = _get_jump_cols(df, config['red_count'])
    if not cols:
        st.info("暂无跳号统计数据")
        return
        
    counts = {c: df[c].sum() for c in cols}
    
    # 找到有数据出现的最大跳号类型索引
    max_idx = -1
    for i, c in enumerate(cols):
        if counts[c] > 0:
            max_idx = i
            
    if max_idx == -1:
        st.info("所选数据范围内未发现跳号组合")
        return
    
    # 截断列表，只保留到最大有数据的字段为止
    active_cols = cols[:max_idx + 1]
    
    df_plot = pd.DataFrame([{"跳号": c, "出现次数": counts[c]} for c in active_cols])
    total = df_plot["出现次数"].sum()
    
    df_plot["百分比"] = df_plot["出现次数"] / total if total > 0 else 0
    df_plot['跳号'] = pd.Categorical(df_plot['跳号'], categories=active_cols, ordered=True)
    
    chart = alt.Chart(df_plot).mark_bar().encode(
        x=alt.X('跳号:O', title='跳号类型', axis=alt.Axis(labelAngle=0), sort=active_cols),
        y=alt.Y('出现次数:Q', title='出现总次数'),
        color=alt.Color('出现次数:Q', scale=alt.Scale(scheme='oranges'), legend=None),
        tooltip=['跳号', '出现次数', alt.Tooltip('百分比:Q', format='.1%', title='百分比')]
    ).properties(title='跳号出现次数分布（历史统计）', width=800, height=300)
    
    text = chart.mark_text(align='center', baseline='bottom', dy=-5, color='black').encode(text=alt.Text('百分比:Q', format='.1%'))
    st.altair_chart(chart + text, use_container_width=True)

def render_chart_jump_trend(df, config):
    st.subheader("跳号趋势分析")
    cols = _get_jump_cols(df, config['red_count'])
    if not cols: return
    
    trend_data = []
    # Show last 50 issues for clarity
    plot_df = df.head(50)
    
    for _, row in plot_df.iterrows():
        for c in cols:
            val = row.get(c, 0)
            if val > 0:
                trend_data.append({'期号': str(row['期号']), '跳号类型': c, '跳号组数': int(val)})
                
    if not trend_data:
        st.info("近期未发现跳号组合")
        return
        
    df_trend = pd.DataFrame(trend_data)
    
    # Optimize Y-axis: only show jump types present in the data
    present_cols = [c for c in cols if c in df_trend['跳号类型'].unique()]
    
    base = alt.Chart(df_trend).properties(width=800, height=300)
    
    points = base.mark_circle(size=60).encode(
        x=alt.X('期号:O', title='期号', axis=alt.Axis(labelAngle=-45)),
        y=alt.Y('跳号类型:N', title='跳号类型', sort=present_cols),
        color=alt.Color('跳号类型:N', title='跳号类型', scale=alt.Scale(domain=present_cols), legend=None),
        tooltip=['期号', '跳号类型', '跳号组数']
    )
    
    text = base.mark_text(
        align='center', 
        baseline='bottom', 
        dx=15
    ).encode(
        x='期号:O',
        y='跳号类型:N',
        text='跳号组数:Q',
        color=alt.value('black')
    )
    
    st.altair_chart((points + text).properties(title='红球跳号趋势'), use_container_width=True)

def render_chart_tail_dist(df, config):
    st.subheader("同尾号分析")
    tail_map = {2: '二尾', 3: '三尾', 4: '四尾', 5: '五尾', 6: '六尾'}
    max_tails = []
    red_cols = [f"红球{i}" for i in range(1, config['red_count']+1)]
    if not all(c in df.columns for c in red_cols): return
    for _, row in df.iterrows():
        balls = [int(row[c]) for c in red_cols if pd.notnull(row[c])]
        if len(balls) < 2: continue
        counts = Counter(b % 10 for b in balls)
        best = max(counts.values(), default=0)
        if best >= 2: max_tails.append(min(best, 6))
    s = pd.Series(max_tails).value_counts().reindex(range(2, 7), fill_value=0)
    df_p = pd.DataFrame({'同尾类型': [tail_map[i] for i in s.index], '出现次数': s.values})
    total = df_p['出现次数'].sum()
    df_p['百分比'] = df_p['出现次数'] / total if total > 0 else 0
    chart = alt.Chart(df_p).mark_bar(color='#4C78A8').encode(
        x=alt.X('同尾类型:N', sort=list(tail_map.values()), axis=alt.Axis(labelAngle=0)),
        y=alt.Y('出现次数:Q'),
        color=alt.Color('出现次数:Q', legend=None),
        tooltip=['同尾类型', '出现次数', alt.Tooltip('百分比:Q', format='.1%')]
    ).properties(title='红球同尾号分布统计', width=600, height=300)
    text = chart.mark_text(align='center', baseline='bottom', dy=-5, color='white').encode(text=alt.Text('百分比:Q', format='.1%'))
    st.altair_chart(chart + text, use_container_width=True)

def render_chart_tail_trend(df, config):
    st.subheader("同尾号趋势分析")
    tail_map = {2: '二尾', 3: '三尾', 4: '四尾', 5: '五尾', 6: '六尾'}
    red_cols = [f"红球{i}" for i in range(1, config['red_count']+1)]
    if not all(c in df.columns for c in red_cols): return
    trend_data = []
    for _, row in df.iterrows():
        balls = [int(row[c]) for c in red_cols if pd.notnull(row[c])]
        if len(balls) < 2: continue
        counts = Counter(b % 10 for b in balls)
        valid = [c for c in counts.values() if c >= 2]
        if valid:
            for v in valid:
                trend_data.append({'期号': row['期号'], '最大同尾数': min(v, 6), '同尾类型': tail_map.get(min(v, 6)), '同尾组数': list(counts.values()).count(v)})
    if not trend_data: return
    df_t = pd.DataFrame(trend_data)
    base = alt.Chart(df_t).properties(width=800, height=300)
    points = base.mark_circle(size=60).encode(
        x=alt.X('期号:O', title='期号', axis=alt.Axis(labelAngle=-45)),
        y=alt.Y('最大同尾数:Q', title='最大同尾数', scale=alt.Scale(domain=[1, 6])),
        color=alt.Color('同尾类型:N', scale=alt.Scale(domain=list(tail_map.values()), range=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'])),
        tooltip=['期号', '同尾类型', '最大同尾数', '同尾组数']
    )
    text = base.mark_text(align='center', baseline='bottom', dy=-5).encode(x='期号:O', y='最大同尾数:Q', text='同尾组数:Q', color=alt.value('black'))
    st.altair_chart((points + text).properties(title='同尾类型历史趋势'), use_container_width=True)

def render_chart_size_ratio(df, config):
    st.subheader("大小比例分析")
    if '大号' not in df.columns: return
    ratios = {f"{i}:{config['red_count']-i}": 0 for i in range(config['red_count'] + 1)}
    for val in df['大号']:
        r = f"{int(val)}:{int(config['red_count']-val)}"
        ratios[r] = ratios.get(r, 0) + 1
    df_p = pd.DataFrame(list(ratios.items()), columns=["大小比例", "出现次数"])
    total = df_p["出现次数"].sum()
    df_p["百分比"] = df_p["出现次数"] / total if total > 0 else 0
    chart = alt.Chart(df_p).mark_bar().encode(
        x=alt.X('大小比例:O', axis=alt.Axis(labelAngle=0)),
        y=alt.Y('出现次数:Q'),
        color=alt.Color('出现次数:Q', legend=None),
        tooltip=['大小比例', '出现次数', alt.Tooltip('百分比:Q', format='.1%')]
    ).properties(title='大小比例分布', width=800, height=300)
    text = chart.mark_text(align='center', baseline='bottom', dy=-5).encode(text=alt.Text('百分比:Q', format='.1%'))
    st.altair_chart(chart + text, use_container_width=True)

def render_chart_size_trend(df, config):
    st.subheader("大小变化趋势")
    if '大号' not in df.columns: return
    df_plot = df.copy()
    df_plot['小号'] = config['red_count'] - df_plot['大号']
    c_data = df_plot.melt(id_vars=["期号"], value_vars=["大号", "小号"], var_name="类别", value_name="数量")
    c_data.loc[c_data['类别'] == '小号', '数量'] *= -1
    chart = alt.Chart(c_data).mark_bar().encode(
        x=alt.X('期号:O', axis=alt.Axis(labelAngle=-45)),
        y=alt.Y('数量:Q', scale=alt.Scale(domain=[-config['red_count'], config['red_count']])),
        color=alt.Color('类别:N', legend=None),
        tooltip=['期号', '类别', '数量']
    ).properties(title='大小号趋势', width=800, height=300)
    st.altair_chart(chart, use_container_width=True)

def render_chart_zone_dist(df, config):
    st.subheader("区间数字汇总分析")
    zones = ['一区', '二区', '三区']
    if not all(z in df.columns for z in zones): return
    counts = {z: df[z].sum() for z in zones}
    df_p = pd.DataFrame(list(counts.items()), columns=["区间", "出现次数"])
    total = df_p["出现次数"].sum()
    df_p["百分比"] = df_p["出现次数"] / total if total > 0 else 0
    df_p['区间'] = pd.Categorical(df_p['区间'], categories=zones, ordered=True)
    chart = alt.Chart(df_p).mark_bar().encode(
        x=alt.X('区间:O', sort=zones, axis=alt.Axis(labelAngle=0)),
        y=alt.Y('出现次数:Q'),
        color=alt.Color('区间:N', legend=None),
        tooltip=['区间', '出现次数', alt.Tooltip('百分比:Q', format='.1%')]
    ).properties(title='区间出现数字次数(一区:;二区:;三区:)', width=800, height=300)
    text = chart.mark_text(align='center', baseline='bottom', dy=-5).encode(text=alt.Text('百分比:Q', format='.1%'))
    st.altair_chart(chart + text, use_container_width=True)

def render_chart_zone_trend(df, config):
    st.subheader("区间数字走势")
    zones = ['一区', '二区', '三区']
    if not all(z in df.columns for z in zones): return
    df_l = df.melt(id_vars=['期号'], value_vars=zones, var_name='区间', value_name='出现次数')
    chart = alt.Chart(df_l).mark_line().encode(
        x=alt.X('期号:O', axis=alt.Axis(labelAngle=-45)),
        y=alt.Y('出现次数:Q'),
        color=alt.Color('区间:N'),
        tooltip=['期号', '区间', '出现次数']
    ).properties(title='区间出现数字走势', width=800, height=300)
    st.altair_chart(chart, use_container_width=True)

def render_chart_repeat_dist(df, config):
    st.subheader("红球重号统计")
    if '重号' not in df.columns: return
    s = df['重号'].fillna(0).astype(int).value_counts().reindex(range(config['red_count']+1), fill_value=0).reset_index()
    s.columns = ['同号数量', '出现次数']
    total = s["出现次数"].sum()
    s["百分比"] = s["出现次数"] / total if total > 0 else 0
    chart = alt.Chart(s).mark_bar().encode(
        x=alt.X('同号数量:N', axis=alt.Axis(labelAngle=0)),
        y=alt.Y('出现次数:Q'),
        tooltip=['同号数量', '出现次数', alt.Tooltip('百分比:Q', format='.1%')]
    ).properties(title='红球同号数量统计(与上一期相同的号码)', width=800, height=300)
    text = chart.mark_text(align='center', baseline='bottom', dy=-5).encode(text=alt.condition(alt.datum.出现次数 > 0, alt.Text('百分比:Q', format='.1%'), alt.value('')))
    st.altair_chart(chart + text, use_container_width=True)

def render_chart_repeat_trend(df, config):
    st.subheader("红球重号分析")
    if '重号' not in df.columns: return
    chart = alt.Chart(df).mark_line().encode(
        x=alt.X('期号:O', axis=alt.Axis(labelAngle=-45)),
        y=alt.Y('重号:Q', scale=alt.Scale(domain=[0, config['red_count']])),
        tooltip=['期号', '重号']
    ).properties(title='红球重号趋势图', width=800, height=300)
    text = chart.mark_text(align='center', baseline='bottom', dy=-5, color='black').encode(text=alt.Text('重号:Q', format='.0f'))
    st.altair_chart(chart + text, use_container_width=True)

def render_chart_neighbor_dist(df, config):
    st.subheader("红球邻号统计")
    if '邻号' not in df.columns: return
    s = df['邻号'].fillna(0).astype(int).value_counts().reindex(range(config['red_count']+1), fill_value=0).reset_index()
    s.columns = ['邻号数量', '出现次数']
    total = s["出现次数"].sum()
    s["百分比"] = s["出现次数"] / total if total > 0 else 0
    chart = alt.Chart(s).mark_bar().encode(
        x=alt.X('邻号数量:N', axis=alt.Axis(labelAngle=0)),
        y=alt.Y('出现次数:Q'),
        tooltip=['邻号数量', '出现次数', alt.Tooltip('百分比:Q', format='.1%')]
    ).properties(title='红球邻号数量统计', width=800, height=300)
    text = chart.mark_text(align='center', baseline='bottom', dy=-5).encode(text=alt.condition(alt.datum.出现次数 > 0, alt.Text('百分比:Q', format='.1%'), alt.value('')))
    st.altair_chart(chart + text, use_container_width=True)

def render_chart_neighbor_trend(df, config):
    st.subheader("红球邻号分析")
    if '邻号' not in df.columns: return
    chart = alt.Chart(df).mark_line().encode(
        x=alt.X('期号:O', axis=alt.Axis(labelAngle=-45)),
        y=alt.Y('邻号:Q'),
        color=alt.value('#1E90FF'),
        tooltip=['期号', '邻号']
    ).properties(title='红球邻号趋势图', width=800, height=300)
    text = chart.mark_text(align='center', baseline='bottom', dy=-5, color='black').encode(text=alt.Text('邻号:Q', format='.0f'))
    st.altair_chart(chart + text, use_container_width=True)

def render_chart_isolated_dist(df, config):
    st.subheader("红球孤号统计")
    if '孤号' not in df.columns: return
    s = df['孤号'].fillna(0).astype(int).value_counts().reindex(range(config['red_count']+1), fill_value=0).reset_index()
    s.columns = ['孤号数量', '出现次数']
    total = s["出现次数"].sum()
    s["百分比"] = s["出现次数"] / total if total > 0 else 0
    chart = alt.Chart(s).mark_bar().encode(
        x=alt.X('孤号数量:N', axis=alt.Axis(labelAngle=0)),
        y=alt.Y('出现次数:Q'),
        tooltip=['孤号数量', '出现次数', alt.Tooltip('百分比:Q', format='.1%')]
    ).properties(title='红球孤号数量统计', width=800, height=300)
    text = chart.mark_text(align='center', baseline='bottom', dy=-5, color='black').encode(text=alt.condition(alt.datum.出现次数 > 0, alt.Text('百分比:Q', format='.1%'), alt.value('')))
    st.altair_chart(chart + text, use_container_width=True)

def render_chart_isolated_trend(df, config):
    st.subheader("红球孤号分析")
    if '孤号' not in df.columns: return
    chart = alt.Chart(df).mark_line().encode(
        x=alt.X('期号:O', axis=alt.Axis(labelAngle=-45)),
        y=alt.Y('孤号:Q'),
        color=alt.value('#1E90FF'),
        tooltip=['期号', '孤号']
    ).properties(title='红球孤号趋势图', width=800, height=300)
    text = chart.mark_text(align='center', baseline='bottom', dy=-5, color='black').encode(text=alt.Text('孤号:Q', format='.0f'))
    st.altair_chart(chart + text, use_container_width=True)

def render_chart_sum_trend(df, config):
    st.subheader("红球和值分析")
    if '和值' not in df.columns: return
    chart = alt.Chart(df).mark_line().encode(
        x=alt.X('期号:O', axis=alt.Axis(labelAngle=-45)),
        y=alt.Y('和值:Q'),
        color=alt.value('#FF5733'),
        tooltip=['期号', '和值']
    ).properties(title='红球和值趋势图', width=800, height=300)
    text = chart.mark_text(align='center', baseline='bottom', dy=-5, color='black').encode(text=alt.Text('和值:Q', format='.0f'))
    st.altair_chart(chart + text, use_container_width=True)

def render_chart_span_trend(df, config):
    st.subheader("红球跨度分析")
    if '跨度' not in df.columns: return
    chart = alt.Chart(df).mark_line().encode(
        x=alt.X('期号:O', axis=alt.Axis(labelAngle=-45)),
        y=alt.Y('跨度:Q'),
        color=alt.value('#1E90FF'),
        tooltip=['期号', '跨度']
    ).properties(title='红球跨度趋势图', width=800, height=300)
    text = chart.mark_text(align='center', baseline='bottom', dy=-5, color='black').encode(text=alt.Text('跨度:Q', format='.0f'))
    st.altair_chart(chart + text, use_container_width=True)

def render_chart_ac_trend(df, config):
    st.subheader("红球 AC 值分析")
    if 'AC' not in df.columns: 
        st.info("暂无 AC 值数据")
        return
    
    # Calculate dynamic Y-axis range based on actual data
    ac_values = df['AC'].dropna()
    if ac_values.empty:
        st.info("暂无有效 AC 值数据")
        return
    
    min_ac = int(ac_values.min())
    max_ac = int(ac_values.max())
    
    # Add padding (10% of range, minimum 1)
    padding = max(1, int((max_ac - min_ac) * 0.1))
    y_min = max(0, min_ac - padding)
    y_max = max_ac + padding
    
    chart = alt.Chart(df).mark_line().encode(
        x=alt.X('期号:O', axis=alt.Axis(labelAngle=-45)),
        y=alt.Y('AC:Q', scale=alt.Scale(domain=[y_min, y_max])),
        color=alt.value('#1E90FF'),
        tooltip=['期号', 'AC']
    ).properties(title='红球 AC 值趋势图', width=800, height=300)
    text = chart.mark_text(align='center', baseline='bottom', dy=-5, color='black').encode(text=alt.Text('AC:Q', format='.0f'))
    st.altair_chart(chart + text, use_container_width=True)

def render_chart_hot_pairs(df, config):
    st.subheader("🔥 热门号码对")
    # 动态获取红球列名
    red_cols = [f"{config['red_col_prefix']}{i}" for i in range(1, config['red_count'] + 1)]
    # Pass only red ball columns for better caching
    freq_df = analyze_top_companion_pairs(df[red_cols], red_cols=red_cols, top_n=10)
    
    if freq_df.empty: 
        st.info("数据量不足，无法分析热门号码对")
        return
        
    bars = alt.Chart(freq_df).mark_bar().encode(
        x=alt.X('号码对:O', title='热门号码对', sort='-y', axis=alt.Axis(labelAngle=-45)),
        y=alt.Y('出现次数:Q', title='出现总次数'),
        color=alt.Color('出现次数:Q', scale=alt.Scale(scheme='reds'), legend=None),
        tooltip=['号码对', '出现次数', alt.Tooltip('百分比:Q', format=".1%")]
    ).properties(title='', width=800, height=300)
    text = bars.mark_text(align='center', baseline='bottom', dy=-5).encode(text=alt.Text('百分比:Q', format=".1%"))
    st.altair_chart(bars + text, use_container_width=True)

def render_chart_hot_triples(df, config):
    st.subheader("🔥 热门号码三元组")
    # 动态获取红球列名
    red_cols = [f"{config['red_col_prefix']}{i}" for i in range(1, config['red_count'] + 1)]
    # Pass only red ball columns for better caching
    freq_df = analyze_top_triples(df[red_cols], red_cols=red_cols, top_n=10)
    
    if freq_df.empty: 
        st.info("数据量不足，无法分析热门三元组")
        return
        
    bars = alt.Chart(freq_df).mark_bar(color='red').encode(
        x=alt.X('号码三元组:O', title='热门号码三元组', sort='-y', axis=alt.Axis(labelAngle=-45)),
        y=alt.Y('出现次数:Q', title='出现总次数'),
        color=alt.Color('出现次数:Q', scale=alt.Scale(scheme='reds'), legend=None),
        tooltip=['号码三元组', '出现次数', alt.Tooltip('百分比:Q', format=".1%")]
    ).properties(title='', width=800, height=300)
    text = bars.mark_text(align='center', baseline='bottom', dy=-5).encode(text=alt.Text('百分比:Q', format=".1%"))
    st.altair_chart(bars + text, use_container_width=True)

# --- UI Layout ---

def render_sidebar(config):
    st.sidebar.title(f"{config['name']}分析选项")
    period = st.sidebar.slider("分析期数", 5, 100, 30, 5)
    if config['code'] == 'ssq':
        st.sidebar.divider()
        st.sidebar.subheader("高级筛选 (仅支持双色球)")
        for k in ['hot_nums_filter', 'cold_nums_filter']:
            if k not in st.session_state: st.session_state[k] = False
        st.session_state.hot_nums_filter = st.sidebar.checkbox("热号筛选", value=st.session_state.hot_nums_filter)
        if st.session_state.hot_nums_filter: st.session_state.hot_nums = st.sidebar.slider("红球热号个数", 0, 6, (1, 2))
        st.session_state.cold_nums_filter = st.sidebar.checkbox("冷号筛选", value=st.session_state.cold_nums_filter)
        if st.session_state.cold_nums_filter: st.session_state.cold_nums = st.sidebar.slider("红球冷号个数", 0, 6, (1, 2))
    st.sidebar.divider()
    st.sidebar.subheader("🤖 AI 助手配置")
    
    brand_models = get_brand_models()
    ai_brand = st.sidebar.selectbox("AI 模型品牌", list(brand_models.keys()), index=0)
    ai_model = st.sidebar.selectbox("具体模型选择", brand_models[ai_brand], index=0)
    
    # Store in session state for render_ai to use
    st.session_state.ai_brand = ai_brand
    st.session_state.ai_model = ai_model
    
    return period

def render_metrics(df, config):
    if df.empty: return
    row = df.iloc[0]
    st.subheader("最新开奖信息")
    
    # Create two columns for Layout: Left (Info/Balls) | Right (Prizes)
    col_left, col_right = st.columns([1, 1], gap="large")
    
    with col_left:
        c1, c2, c3, c4 = st.columns([1, 1.5, 2, 2])
        with c1: st.metric("期号", str(row['期号']))
        
        # Format Date: remove time
        display_date = str(row['开奖日期'])
        if ' ' in display_date: display_date = display_date.split(' ')[0]
        elif 'T' in display_date: display_date = display_date.split('T')[0]
        with c2: st.metric("开奖日期", display_date)
        
        # Add Sales and Pool metrics
        if '本期销售金额' in row and not pd.isna(row['本期销售金额']):
            with c3: st.metric("本期销售金额", f"{float(row['本期销售金额']):,.0f}")
        if '奖池累计金额' in row and not pd.isna(row['奖池累计金额']):
            with c4: st.metric("奖池累计金额", f"{float(row['奖池累计金额']):,.0f}")
        
        balls_html = ""
        for i in range(1, config['red_count']+1):
            col = f"{config['red_col_prefix']}{i}"
            if col in df.columns: balls_html += f'<div class="lottery-ball red-ball">{int(row[col])}</div>'
        if config['has_blue']:
            bcols = []
            if config['blue_count'] == 1:
                col = config['blue_col_name'] if config['blue_col_name'] in df.columns else '蓝球'
                if col in df.columns: bcols.append(col)
                elif '篮球' in df.columns: bcols.append('篮球')
            else:
                base = config['blue_col_name']
                for i in range(1, config['blue_count'] + 1):
                    if f"{base}{i}" in df.columns: bcols.append(f"{base}{i}")
                    elif f"篮球{i}" in df.columns: bcols.append(f"篮球{i}")
            
            for c in bcols:
                balls_html += f'<div class="lottery-ball blue-ball">{int(row[c])}</div>'
        
        st.markdown(balls_html, unsafe_allow_html=True)
        
        if config['code'] == 'ssq' and '奇数' in row:
            # Fix: use safe column lookup
            odd_cnt = row['奇数'] if '奇数' in row else 0
            big_cnt = row['大号'] if '大号' in row else 0
            cap = f"红球分析: 奇偶: {int(odd_cnt)}:{int(config['red_count']-odd_cnt)}, 大小: {int(big_cnt)}:{int(config['red_count']-big_cnt)}"
            st.caption(cap)

    with col_right:
        # Display Prize Information
        prize_cols = [c for c in df.columns if c.endswith('注数') and not c.endswith('追加注数')]
        if prize_cols:
            st.markdown("##### 🏆 中奖详情")
            
            # KL8 specific filtering
            kl8_mode = None
            if config['code'] == 'kl8':
                modes = ["选一", "选二", "选三", "选四", "选五", "选六", "选七", "选八", "选九", "选十"]
                kl8_mode = st.pills("玩法选择", modes, selection_mode="single", default="选十")
            
            # Sort prizes (Standard first, then others)
            # Define standard prize orders for different lotteries
            std_prize_map = {
                'ssq': ["一等奖", "二等奖", "三等奖", "四等奖", "五等奖", "六等奖", "福运奖"],
                'dlt': ["一等奖", "二等奖", "三等奖", "四等奖", "五等奖", "六等奖", "七等奖"],
                'd3': ["单选", "组三", "组六"],
                'pl3': ["直选", "组选三", "组选六"],
                'pl5': ["直选"],
                'qlc': ["一等奖", "二等奖", "三等奖", "四等奖", "五等奖", "六等奖", "七等奖"],
                'xqxc': ["一等奖", "二等奖", "三等奖", "四等奖", "五等奖", "六等奖"]
            }
            
            p_order = std_prize_map.get(config['code'], ["一等奖", "二等奖", "三等奖", "四等奖", "五等奖", "六等奖", "七等奖"])
            
            sorted_prizes = []
            for p in p_order:
                if f"{p}注数" in df.columns: sorted_prizes.append(p)
            for c in prize_cols:
                p = c.replace('注数', '')
                if p not in sorted_prizes: sorted_prizes.append(p)

            p_data = []
            for p in sorted_prizes:
                # Filter for KL8 if mode is selected
                if kl8_mode and not p.startswith(kl8_mode):
                    continue
                    
                num = row.get(f"{p}注数", 0)
                money = row.get(f"{p}奖金", 0)
                
                # Logic: Always show standard jackpots OR selected KL8 mode tiers
                is_standard = p in p_order
                is_kl8_selected = kl8_mode and p.startswith(kl8_mode)
                
                if num > 0 or money > 0 or is_standard or is_kl8_selected:
                    # Use 0 if value is NaN/None
                    num = int(num) if pd.notna(num) else 0
                    money = int(float(money)) if pd.notna(money) else 0
                    
                    item = {"奖项": p, "中奖注数": num, "单注奖金": money}
                    
                    # Handle DLT 追加
                    if f"{p}追加注数" in df.columns:
                        anum = row.get(f"{p}追加注数", 0)
                        amoney = row.get(f"{p}追加奖金", 0)
                        if pd.notna(anum) and pd.notna(amoney):
                            if anum > 0 or amoney > 0 or (is_standard and num > 0):
                                item["追加注数"] = int(anum)
                                item["追加奖金"] = int(float(amoney))
                    p_data.append(item)
            
            if p_data:
                # Use pandas Styler to format numbers with thousands separators (e.g. 1,000,000)
                # Streamlit NumberColumn format does NOT support commas, so Styler is necessary.
                df_display = pd.DataFrame(p_data)
                
                # Ensure numeric types
                vals = ["中奖注数", "单注奖金", "追加注数", "追加奖金"]
                for v in vals:
                    if v in df_display.columns:
                        df_display[v] = pd.to_numeric(df_display[v], errors='coerce').fillna(0)

                # Apply comma format
                st.dataframe(
                    df_display.style.format("{:,.0f}", subset=[c for c in vals if c in df_display.columns]), 
                    hide_index=True, 
                    use_container_width=True
                )
            else:
                st.info("本玩法暂无中奖数据" if kl8_mode else "本期暂无详细中奖数据")

    st.markdown("---")

def render_ai(df, config):
    st.subheader(f"🤖 AI 预测助手 ({st.session_state.get('ai_brand', 'Gemini')})")
    
    brand = st.session_state.get("ai_brand", "Gemini")
    model_name = st.session_state.get("ai_model", "gemini-2.0-flash")
    
    # Try to get API Key from environment first
    env_keys = {
        "Gemini": "GEMINI_API_KEY",
        "NVIDIA": "NV_API_KEY",
        "MiniMax": "MINIMAX_API_KEY",
        "DashScope": "ALIYUNCS_API_KEY"
    }
    env_key_name = env_keys.get(brand)
    default_key = os.getenv(env_key_name, "")
    
    # Retrieve API key from environment; no UI input displayed
    key = default_key
    
    if st.button("开始分析并预测"):
        if not key:
            st.error(f"请在系统环境变量中设置 {env_key_name}")
            return
            
        try:
            # Get period from session state if needed, but here we use the df passed in
            # which is already filtered by the slider 'period'
            data_str = prepare_lottery_data_text(df, config)
            
            # Show Prompt
            with st.expander("查看发送给 AI 的原始指令 (Prompt)"):
                # We rebuild the prompt logic here just for display, or we could expose it from helper
                # To keep it simple, just note what's sent
                st.info(f"正在配置 {brand} / {model_name} 进行分析...")
                st.text_area("数据内容:", data_str, height=200)
            
            with st.status("AI 正在深度分析中...", expanded=True) as status:
                prediction = generate_ai_prediction(brand, model_name, key, data_str, config)
                status.update(label="分析完成！", state="complete", expanded=False)
                
            st.markdown("### 📊 AI 预测建议")
            
            raw_content = prediction
            if "分析结果" in raw_content:
                thinking, result = raw_content.split("分析结果", 1)
                with st.expander("🤔 思考过程"):
                    st.markdown(thinking.strip())
                st.markdown("### 📋 深度分析报告")
                st.markdown(result.strip().replace('\\n', '\n'))
            else:
                # 如果没有分割词，尝试使用 format_ai_response 处理可能的 <think> 块
                thinking, result = format_ai_response(raw_content)
                if thinking:
                    with st.expander("🤔 查看 AI 思考过程"):
                        st.markdown(thinking.strip())
                st.markdown(result.replace('\\n', '\n'))
            
        except Exception as e: 
            st.error(f"分析过程中出现错误: {e}")

def render_ai_analysis(df, config):
    st.subheader(f"📊 {config['name']} AI 预测历史对比")
    
    csv_file = "data/ai_predictions_history.csv"
    if not os.path.exists(csv_file):
        st.info("尚未发现 AI 预测记录。请先在 'AI 预测' 板块生成预测或运行批量脚本。")
        return
        
    try:
        # Load and filter by lottery
        df_hist = pd.read_csv(csv_file)
        df_hist = df_hist[df_hist['lottery'] == config['name']]
        
        if df_hist.empty:
            st.info(f"暂无 {config['name']} 的 AI 预测记录。")
            return
            
        # Standardize period format
        df_hist['target_period'] = df_hist['target_period'].astype(str)
        
        # Period Selector
        periods = sorted(df_hist['target_period'].unique(), reverse=True)
        sel_period = st.selectbox("📅 选择预测期号", periods, key="analysis_period_sel")
        
        # Filter by period
        df_period = df_hist[df_hist['target_period'] == sel_period]
        
        # Latest record logic: keep only the latest prediction for each model
        df_period = df_period.sort_values('timestamp', ascending=False).drop_duplicates('model')
        
        if df_period.empty:
            st.warning("该期号暂无预测数据。")
            return
            
        # 🟢 获取当期实际开奖结果 (用于命中标识)
        draw_row = df[df['期号'].astype(str) == str(sel_period)]
        winning_reds = []
        winning_blues = []
        
        if not draw_row.empty:
            row_draw = draw_row.iloc[0]
            for i in range(1, config['red_count'] + 1):
                col = f"{config['red_col_prefix']}{i}"
                if col in row_draw: winning_reds.append(int(row_draw[col]))
            
            if config['has_blue']:
                if '蓝球' in row_draw: winning_blues.append(int(row_draw['蓝球']))
                elif '篮球' in row_draw: winning_blues.append(int(row_draw['篮球']))
            
            # 显示实际开奖号码
            st.markdown(f"#### 📅 {sel_period}期 实际开奖结果")
            res_html = ""
            for r in winning_reds: res_html += f'<div class="lottery-ball red-ball">{r}</div>'
            for b in winning_blues: res_html += f'<div class="lottery-ball blue-ball">{b}</div>'
            st.markdown(res_html, unsafe_allow_html=True)
        else:
            st.markdown(f"#### 📅 {sel_period}期 开奖状态: `⏳ 等待开奖`")

        st.divider()

        # Display Comparative View
        models = df_period['model'].unique()
        cols = st.columns(len(models))
        
        for i, model in enumerate(models):
            with cols[i]:
                row = df_period[df_period['model'] == model].iloc[0]
                st.markdown(f"#### 🤖 {model}")
                st.caption(f"🕒 {row['timestamp']}")
                
                try:
                    recs = json.loads(row['recommendations'])
                    
                    # 1. Dan Codes
                    if recs.get('dan'):
                        st.markdown(f"**📍 核心胆码**")
                        dan_html = ""
                        for d in recs['dan']:
                            is_hit = "hit-ball" if int(d) in winning_reds else ""
                            dan_html += f'<span class="lottery-ball red-ball {is_hit}" style="width:30px; height:30px; line-height:30px; font-size:0.8em;">{d}</span>'
                        st.markdown(dan_html, unsafe_allow_html=True)
                    
                    # 2. Recommendations
                    if recs.get('groups'):
                        st.markdown("**💡 推荐组合**")
                        for idx, g in enumerate(recs['groups']):
                            with st.expander(f"方案 {idx+1}", expanded=(idx==0)):
                                # Red balls
                                reds_html = "🔴"
                                for r in g.get('reds', []):
                                    is_hit = "background-color:gold; color:black; font-weight:bold; padding:2px 4px; border-radius:3px;" if int(r) in winning_reds else ""
                                    reds_html += f' <code style="{is_hit}">{r:02d}</code>'
                                st.markdown(reds_html, unsafe_allow_html=True)
                                
                                # Blue balls if any
                                if g.get('blues'):
                                    blues_html = "🔵"
                                    for b in g['blues']:
                                        is_hit = "background-color:gold; color:black; font-weight:bold; padding:2px 4px; border-radius:3px;" if int(b) in winning_blues else ""
                                        blues_html += f' <code style="{is_hit}">{b:02d}</code>'
                                    st.markdown(blues_html, unsafe_allow_html=True)
                                    
                                # Reason
                                if g.get('reason'):
                                    st.caption(f"分析: {g['reason']}")
                    
                    # 3. KL8 Special
                    if recs.get('kl8_numbers'):
                        st.markdown("**🔢 快乐8 选二十**")
                        nums = sorted(recs['kl8_numbers'])
                        st.write(", ".join([f"{n:02d}" for n in nums]))
                        
                    # 4. Raw Analysis Summary
                    with st.expander("📄 查看完整分析报告"):
                        raw_content = row.get('raw_response', "")
                        if "分析结果" in raw_content:
                            thinking, result = raw_content.split("分析结果", 1)
                            with st.expander("🤔 思考过程"):
                                st.markdown(thinking.strip())
                            st.markdown("### 📋 深度分析报告")
                            st.markdown(result.strip().replace('\\n', '\n'))
                        else:
                            # 如果没有分割词，尝试使用 format_ai_response 处理可能的 <think> 块
                            thinking, result = format_ai_response(raw_content)
                            if thinking:
                                with st.expander("🤔 查看 AI 思考过程"):
                                    st.markdown(thinking.strip())
                            st.markdown(result.replace('\\n', '\n'))
                        
                except Exception as e:
                    st.error(f"解析预测数据失败: {e}")
                    
        st.divider()
        st.caption("注：对比各模型的预测逻辑与命中规律，有助于提高选号参考价值。")

    except Exception as e:
        st.error(f"加载分析数据失败: {e}")

def render_backtest_results(df_full, conf):
    st.markdown(f"### 📋 {conf['name']} 历史回测详情分析")
    
    # 1. Load Data
    csv_file = f"data/{conf['code']}_backtest.csv"
    if not os.path.exists(csv_file):
        st.warning(f"⚠️ 暂无回测数据 ({csv_file} 不存在)")
        return
        
    try:
        df_back = pd.read_csv(csv_file)
        if 'Target_Period' in df_back.columns:
            df_back['Target_Period'] = pd.to_numeric(df_back['Target_Period'], errors='coerce').fillna(0).astype(int).astype(str)
    except Exception as e:
        st.error(f"读取回测数据失败: {e}")
        return
        
    if df_back.empty:
        st.warning("⚠️ 回测数据为空")
        return
        
    # 2. Selectors
    run_times = sorted(df_back['Run_Time'].unique(), reverse=True)
    sel_run_time = st.selectbox("1. 选择回测执行时间", run_times)
    df_run = df_back[df_back['Run_Time'] == sel_run_time].sort_values("Target_Period", ascending=False)
    periods = df_run['Target_Period'].unique()
    
    def format_period_label(p):
        p_match = df_full[df_full['期号'].astype(str) == str(p)]
        if p_match.empty: return f"✨ 期号 {p} (预测下一期)"
        return f"🔙 期号 {p} (历史回测)"
    
    sel_period = st.selectbox("2. 选择回测目标期号", periods, format_func=format_period_label)

    # 3. Build Run Summary Table
    st.markdown("#### 📊 本次运行汇总 (命中率概览)")
    summary_data = []
    methods = ['A', 'B', 'C', 'D']
    metrics = conf.get('eval_metrics', {"top_n_1": 6, "top_n_2": 10, "green_threshold": 3, "red_threshold": 4})
    n1, n2 = metrics['top_n_1'], metrics['top_n_2']
    total_draw = conf['red_count']
    
    red_cols = [c for c in df_full.columns if conf['red_col_prefix'] in c]
    
    for _, r_idx in df_run.iterrows():
        p = r_idx['Target_Period']
        p_str = str(p)
        p_match = df_full[df_full['期号'].astype(str) == p_str]
        
        if p_match.empty:
            summary_data.append({
                "回测期号": f"{p} ✨",
                "综合推荐": "-", "模型 A (统计)": "-", "模型 B (RF)": "-", "模型 C (XGB)": "-", "模型 D (LSTM)": "-"
            })
            continue
            
        a_red = set(p_match.iloc[0][red_cols].values.astype(int).tolist())
        p_hits = {"回测期号": p_str}
        
        r_start, r_end = conf['red_range']
        num_list = list(range(r_start, r_end + 1))
        m_probs = {m: [] for m in methods}
        for n in num_list:
            for m in methods:
                col = f"Prob_{m}_{n:02d}"
                val = r_idx[col] if col in r_idx else 0.0
                try:
                    # Handle potential string/mixed types from CSV
                    m_probs[m].append(float(val))
                except (ValueError, TypeError):
                    m_probs[m].append(0.0)
        
        m_scores = {}
        for m in methods:
            probs = np.array(m_probs[m])
            p_min, p_max = probs.min(), probs.max()
            m_scores[m] = (probs - p_min) / (p_max - p_min) if p_max > p_min else np.zeros_like(probs)
        
        ens_scores = np.mean([m_scores[m] for m in methods], axis=0)
        
        def get_hit_str(scores, actual):
            top_indices = np.argsort(scores)[::-1]
            t1 = (np.array(num_list)[top_indices[:n1]]).tolist()
            t2 = (np.array(num_list)[top_indices[:n2]]).tolist()
            h1 = len(set(t1) & actual)
            h2 = len(set(t2) & actual)
            return f"{h1}/{total_draw} ({n1}) | {h2}/{total_draw} ({n2})"

        p_hits["综合推荐"] = get_hit_str(ens_scores, a_red)
        for m in methods:
            name_map = {'A': '模型 A (统计)', 'B': '模型 B (RF)', 'C': '模型 C (XGB)', 'D': '模型 D (LSTM)'}
            p_hits[name_map[m]] = get_hit_str(m_probs[m], a_red)
        summary_data.append(p_hits)

    if summary_data:
        html = f"""
        <style>
            .summary-table {{ width:100%; border-collapse: collapse; margin-bottom: 20px; }}
            .summary-table th {{ padding: 8px; background-color: #f0f2f6; border: 1px solid #ddd; text-align: center; }}
            .summary-table td {{ padding: 8px; border: 1px solid #ddd; text-align: center; }}
            .hit-req {{ color: #28a745; font-weight: bold; }}
            .hit-exc {{ color: #d73a49; font-weight: bold; }}
        </style>
        <table class="summary-table">
        <thead><tr>
        """
        cols = ["回测期号", "综合推荐", "模型 A (统计)", "模型 B (RF)", "模型 C (XGB)", "模型 D (LSTM)"]
        for col in cols: html += f"<th>{col}</th>"
        html += "</tr></thead><tbody>"
        
        for row in summary_data:
            html += "<tr>"
            for col in cols:
                val = row.get(col, "-")
                style = ""
                if "/" in val and "(" in val:
                    try:
                        h = int(val.split("/")[0])
                        if h >= metrics['red_threshold']: style = "class='hit-exc'"
                        elif h >= metrics['green_threshold']: style = "class='hit-req'"
                    except: pass
                html += f"<td {style}>{val}</td>"
            html += "</tr>"
        html += "</tbody></table>"
        st.markdown(html, unsafe_allow_html=True)
    
    st.divider()

    # 4. Details for Selected Period
    st.markdown(f"#### 🔍 期号 {sel_period} 详细回测对比")
    
    # actual_red/blue logic
    actual_red = []
    actual_blue = []
    try:
        p_match = df_full[df_full['期号'].astype(str) == str(sel_period)]
        if not p_match.empty:
            draw_row = p_match.iloc[0]
            red_cols = [c for c in df_full.columns if conf['red_col_prefix'] in c]
            actual_red = sorted(draw_row[red_cols].values.astype(int).tolist())
            if conf['has_blue'] and conf['blue_col_name']:
                actual_blue = [int(draw_row[conf['blue_col_name']])]
    except Exception as e:
        pass

    # Display Actual Draw & Main Metrics
    col_l, col_r = st.columns([1, 1])
    with col_l:
        if actual_red:
            st.markdown(f"**实际开奖:** " + 
                        " ".join([f"<span class='lottery-ball red-ball'>{r}</span>" for r in actual_red]) + 
                        (f" <span class='lottery-ball blue-ball'>{actual_blue[0]}</span>" if actual_blue else ""), 
                        unsafe_allow_html=True)
        else:
            st.info("暂无实际开奖数据 (未来预测)")
            
    # Process Score Data for Selected Period
    row = df_run[df_run['Target_Period'] == sel_period].iloc[0]
    data = []
    r_start, r_end = conf['red_range']
    num_list = list(range(r_start, r_end + 1))
    
    for n in num_list:
        item = {'Number': n}
        for m in methods:
            prob_col = f"Prob_{m}_{n:02d}"
            item[f'Prob_{m}'] = row[prob_col] if prob_col in row else 0.0
        data.append(item)
    df_metrics = pd.DataFrame(data)
    for m in methods:
        col_p = f'Prob_{m}'
        df_metrics[col_p] = pd.to_numeric(df_metrics[col_p], errors='coerce').fillna(0.0)
        p_min, p_max = df_metrics[col_p].min(), df_metrics[col_p].max()
        df_metrics[f'Score_{m}'] = (df_metrics[col_p] - p_min) / (p_max - p_min) if p_max > p_min else 0.0
            
    df_metrics['Ensemble_Score'] = df_metrics[[f'Score_{m}' for m in methods]].mean(axis=1)
    
    # Calculate current period hits for display in col_r
    h1 = len(set(df_metrics.sort_values('Ensemble_Score', ascending=False).head(n1)['Number']) & set(actual_red)) if actual_red else 0
    h2 = len(set(df_metrics.sort_values('Ensemble_Score', ascending=False).head(n2)['Number']) & set(actual_red)) if actual_red else 0
    
    with col_r:
        s1, s2 = st.columns(2)
        s1.metric(f"Ensemble Top {n1}", f"{h1}/{total_draw}")
        s2.metric(f"Ensemble Top {n2}", f"{h2}/{total_draw}")
    
    st.divider()
    
    # 6. Detailed Table Comparison
    st.subheader(f"🔢 模型详细对比 (Top {n2 + 5})")
    
    # Pre-calculate data and hit metrics for the table
    top_n = n2 + 5
    col_data = {
        'Ensemble': df_metrics.sort_values('Ensemble_Score', ascending=False).reset_index(drop=True)
    }
    for m in methods:
        col_data[m] = df_metrics.sort_values(f'Prob_{m}', ascending=False).reset_index(drop=True)

    # Calculate Header Metrics
    header_metrics = {}
    for key, data_df in col_data.items():
        if actual_red:
            hh1 = len(set(data_df.head(n1)['Number']) & set(actual_red))
            hh2 = len(set(data_df.head(n2)['Number']) & set(actual_red))
            header_metrics[key] = f"<br><small style='color:#28a745; font-weight:normal'>Hits: {hh1}/{total_draw} ({n1}) | {hh2}/{total_draw} ({n2})</small>"
        else:
            header_metrics[key] = ""

    # Build HTML table
    table_html = f"""
    <style>
        .backtest-table {{ width:100%; border-collapse: collapse; font-family: sans-serif; table-layout: fixed; border: 1px solid #dee2e6; }}
        .backtest-table th {{ padding: 12px; text-align: center; background-color: #f8f9fa; border: 1px solid #dee2e6; color: #495057; font-weight: 600; }}
        .backtest-table td {{ padding: 10px; text-align: center; border: 1px solid #dee2e6; vertical-align: middle; }}
        .backtest-table tr:hover {{ background-color: #f1f3f5; }}
        .hit-score {{ color: #d10000; font-weight: bold; background-color: #fff0f0 !important; }}
        .num-label {{ font-size: 1.1em; display: block; margin-bottom: 2px; }}
        .stat-label {{ color: #888; font-size: 0.85em; }}
    </style>
    <table class="backtest-table">
    <thead>
    <tr>
        <th style="width: 60px;">排名</th>
        <th>🏆 综合推荐{header_metrics['Ensemble']}</th>
        <th>模型 A (统计){header_metrics['A']}</th>
        <th>模型 B (RF){header_metrics['B']}</th>
        <th>模型 C (XGB){header_metrics['C']}</th>
        <th>模型 D (LSTM){header_metrics['D']}</th>
    </tr>
    </thead>
    <tbody>
    """
    
    for i in range(min(top_n, len(df_metrics))):
        table_html += "<tr>"
        table_html += f"<td style='color: #6c757d; font-weight: 500;'>{i+1}</td>"
        
        # Ensemble cell
        r_ens = col_data['Ensemble'].iloc[i]
        num_ens = int(r_ens['Number'])
        is_hit_ens = num_ens in actual_red
        class_ens = "class='hit-score'" if is_hit_ens else ""
        table_html += f"<td {class_ens}><span class='num-label'>{num_ens:02d}</span><span class='stat-label'>({r_ens['Ensemble_Score']:.3f})</span></td>"
        
        # Model cells
        for m in methods:
            r_m = col_data[m].iloc[i]
            num_m = int(r_m['Number'])
            is_hit_m = num_m in actual_red
            class_m = "class='hit-score'" if is_hit_m else ""
            table_html += f"<td {class_m}><span class='num-label'>{num_m:02d}</span><span class='stat-label'>{r_m[f'Prob_{m}']:.1%}<br>({r_m[f'Score_{m}']:.2f})</span></td>"
            
        table_html += "</tr>"
        
    table_html += "</tbody></table>"
    st.markdown(table_html, unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

def render_morphological_analysis(df_full, conf):
    """
    形态学优选分析：从回测概率中选取高分球进行组合评分
    """
    st.subheader(f"🎯 {conf['name']} 形态学智能优选")
    
    # 1. 加载回测数据
    csv_file = f"data/{conf['code']}_backtest.csv"
    if not os.path.exists(csv_file):
        st.warning(f"⚠️ 暂无回测数据，请先运行预测脚本。")
        return
        
    try:
        df_back = pd.read_csv(csv_file)
        if df_back.empty: return
        # 核心修复：排除掉重复写入的表头行数据
        df_back = df_back[df_back['Target_Period'] != 'Target_Period']
    except Exception as e:
        st.error(f"读取数据失败: {e}")
        return

    # 2. 预测记录选择 (时间 & 期号)
    run_times = sorted(df_back['Run_Time'].unique(), reverse=True)
    sel_run_time = st.selectbox("📅 1. 选择预测执行批次 (按时间)", run_times, key="morph_run_time")
    
    df_run = df_back[df_back['Run_Time'] == sel_run_time]
    periods = sorted(df_run['Target_Period'].unique(), reverse=True)
    sel_period = st.selectbox("🔢 2. 选择目标预测期号", periods, key="morph_period")
    
    row_data = df_run[df_run['Target_Period'] == sel_period].iloc[0]

    # 3. 参数与模型选择
    col1, col2, col3 = st.columns(3)
    with col1:
        model_options = {'Prob_A': '模型 A (统计)', 'Prob_B': '模型 B (RF)', 'Prob_C': '模型 C (XGB)', 'Prob_D': '模型 D (LSTM)'}
        sel_model = st.selectbox("选择评分基准模型", list(model_options.keys()), format_func=lambda x: model_options[x], index=1)
    with col2:
        top_n_balls = st.slider("候选红球池数量", 10, 20, 15)
    with col3:
        num_rec = st.slider("推荐组合数量", 1, 10, 5)

    # 4. 提取与展示元数据
    st.info(f"📋 **当前方案详情**: 目标期号 `{sel_period}` | 源模型 `{model_options[sel_model]}` | 预测生成于 `{sel_run_time}`")

    # 4.1 提取实际开奖 (如果已开奖)
    actual_red = []
    p_match = df_full[df_full['期号'].astype(str) == str(sel_period)]
    if not p_match.empty:
        r_cols = [f"{conf['red_col_prefix']}{i}" for i in range(1, conf['red_count'] + 1)]
        actual_red = sorted(p_match.iloc[0][r_cols].values.astype(int).tolist())
        st.write(f"✅ **该期实际开奖**: " + " ".join([f"<span style='color:#ff4b4b; font-weight:bold;'>{r:02d}</span>" for r in actual_red]), unsafe_allow_html=True)
    else:
        st.write("✨ **该期尚未开奖 (未来预测)**")

    # 提取高分红球
    r_start, r_end = conf['red_range']
    prob_cols = [f"{sel_model}_{i:02d}" for i in range(r_start, r_end + 1)]
    prob_cols = [c for c in prob_cols if c in row_data.index]
    
    if not prob_cols:
        st.error("所选模型的概率数据缺失")
        return

    probs = pd.to_numeric(row_data[prob_cols]).values
    top_indices = np.argsort(probs)[-top_n_balls:][::-1]
    top_balls = sorted([int(prob_cols[i].split('_')[-1]) for i in top_indices])
    
    # 高亮命中
    ball_html = []
    hit_count = 0
    for b in top_balls:
        if b in actual_red:
            ball_html.append(f"<span style='color:#28a745; font-weight:bold; border-bottom:2px solid #28a745;'>{b:02d}</span>")
            hit_count += 1
        else:
            ball_html.append(f"<span>{b:02d}</span>")
    
    st.write(f"💡 **锁定 Top {top_n_balls} 核心候选球 (命中: {hit_count})**: " + " ".join(ball_html), unsafe_allow_html=True)

    # 5. 获取上一期号码 (对比开奖)
    if df_full.empty: return
    
    # 智能查找上一期：
    # 1. 尝试找到当前期号的开奖日期
    target_row = df_full[df_full['期号'].astype(str) == str(sel_period)]
    if not target_row.empty:
        target_date = target_row['开奖日期'].iloc[0]
        # 找到日期早于 target_date 的第一条记录
        history_row = df_full[df_full['开奖日期'] < target_date].head(1)
    else:
        # 如果当前期还没开奖，直接取最新的开奖记录
        history_row = df_full.head(1)
        
    if history_row.empty: history_row = df_full.iloc[0] # 兜底
    
    last_period_nums = sorted([
        int(history_row[f"{conf['red_col_prefix']}{i}"].iloc[0] if isinstance(history_row, pd.DataFrame) else history_row[f"{conf['red_col_prefix']}{i}"]) 
        for i in range(1, conf['red_count'] + 1)
    ])
    
    # 获取上一期的期号用于显示
    last_issue = history_row['期号'].iloc[0] if not history_row.empty else "未知"
    st.write(f"📏 **形态参考基准**: 以 `{last_issue}` 期号码 `{', '.join([f'{x:02d}' for x in last_period_nums])}` 为参照")

    # 6. 生成组合并打分
    if st.button("🚀 开始形态学漏斗筛选"):
        with st.status("正在进行数千种组合的物理分布扫描...", expanded=True) as status:
            all_combos = list(itertools.combinations(top_balls, 6))
            scored_data = []
            
            for combo in all_combos:
                c_list = list(combo)
                score = calculate_morphology_score(c_list, last_period_nums, conf['name'])
                scored_data.append((c_list, score))
            
            # 排序
            scored_data.sort(key=lambda x: x[1], reverse=True)
            
            # 统计分数分布
            score_counts = Counter([d[1] for d in scored_data])
            total_c = len(scored_data)
            dist_str = " | ".join([f"**{s}分**: {count}组({count/total_c:.1%})" for s, count in sorted(score_counts.items(), reverse=True)])
            
            status.update(label=f"扫描完成！从 {total_c} 组中筛选出最优方案。", state="complete")
            st.markdown(f"📊 **组合得分分布**: {dist_str}")

        # 7. 展示结果
        st.success(f"已为您筛选出得分最高的 {num_rec} 组精品组合：")
        for i in range(min(num_rec, len(scored_data))):
            combo, score = scored_data[i]
            with st.expander(f"🏆 {sel_period}期 推荐组合 {i+1} (得分: {score}/100) - {', '.join([f'{x:02d}' for x in combo])}"):
                report = get_morphology_report(combo, last_period_nums, conf['name'])
                st.code(report, language="text")
                st.write(f"**投注建议:** `{', '.join([f'{x:02d}' for x in combo])}`")

def main():
    st.set_page_config(page_title="彩票分析工具", layout="wide")
    st.markdown("""
    <style>
    .lottery-ball { display: inline-block; width: 40px; height: 40px; border-radius: 50%; text-align: center; line-height: 40px; margin: 5px; font-weight: bold; color: white; }
    .red-ball { background-color: #ff5b5b; }
    .blue-ball { background-color: #5b9fff; }
    .hit-ball { border: 3px solid #ffd700 !important; box-shadow: 0 0 10px #ffd700; transform: scale(1.1); }
    .ai-report-content { font-size: 1.05em; line-height: 1.7; color: #1a1a1a; padding: 15px; background: #fff; border-radius: 8px; border: 1px solid #eee; }
    .ai-thinking-process { font-size: 0.9em; line-height: 1.5; color: #555; background: #fdfdfd; padding: 12px; border-left: 4px solid #ddd; margin-bottom: 20px; }
    </style>
    """, unsafe_allow_html=True)
    sel = st.sidebar.selectbox("选择彩票类型", list(LOTTERY_CONFIG.keys()))
    conf = LOTTERY_CONFIG[sel]; conf['name'] = sel
    p = render_sidebar(conf)
    df_full = load_full_data(sel)
    df = get_filtered_data(df_full, limit=p)
    if df.empty: st.error(f"No data for {sel}"); return
    
    st.title(f"📊 {sel} 数据分析")
    t1, ta, t2, t3, t4, t5 = st.tabs(["📈 趋势分析", "📊 预测分析", "🤖 AI 预测", "🎯 形态优选", "📋 历史数据", "🔙 回测结果"])
    with t1:
        render_metrics(df, conf)
        charts = conf.get("supported_charts", ["red_freq"])
        c_map = {
            "red_freq": render_chart_red_freq, "blue_freq": render_chart_blue_freq,
            "odd_even_ratio": render_chart_odd_even_ratio, "odd_even_trend": render_chart_odd_even_trend,
            "consecutive_dist": render_chart_consecutive_dist, "consecutive_trend": render_chart_consecutive_trend,
            "jump_dist": render_chart_jump_dist, "jump_trend": render_chart_jump_trend,
            "tail_dist": render_chart_tail_dist, "tail_trend": render_chart_tail_trend,
            "size_ratio": render_chart_size_ratio, "size_trend": render_chart_size_trend,
            "zone_dist": render_chart_zone_dist, "zone_trend": render_chart_zone_trend,
            "repeat_dist": render_chart_repeat_dist, "repeat_trend": render_chart_repeat_trend,
            "neighbor_dist": render_chart_neighbor_dist, "neighbor_trend": render_chart_neighbor_trend,
            "isolated_dist": render_chart_isolated_dist, "isolated_trend": render_chart_isolated_trend,
            "sum_trend": render_chart_sum_trend, "span_trend": render_chart_span_trend,
            "ac_trend": render_chart_ac_trend, "hot_pairs": render_chart_hot_pairs, "hot_triples": render_chart_hot_triples
        }
        cols = None
        for i, ck in enumerate(charts):
            if i % 2 == 0: cols = st.columns(2)
            with cols[i % 2]:
                if ck in c_map: c_map[ck](df, conf); st.divider()
    with ta: render_ai_analysis(df, conf)
    with t2: render_ai(df, conf)
    with t3: render_morphological_analysis(df_full, conf)
    with t4: st.dataframe(df, use_container_width=True)
    with t5: render_backtest_results(df_full, conf)

if __name__ == "__main__": main()