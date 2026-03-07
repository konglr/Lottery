import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from collections import Counter
from funcs.functions import analyze_top_companion_pairs, analyze_top_triples

def render_chart_red_freq(df, config):
    st.subheader("红球冷热分析")
    min_v, max_v = config['red_range']
    red_cols = [f"{config['red_col_prefix']}{i}" for i in range(1, config['red_count']+1)]
    if not all(c in df.columns for c in red_cols): return
    calc_range = range(min_v, max_v + 1)
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
    
    # Robustly identify blue ball columns from the dataframe
    blue_cols = []
    # 1. Check direct name from config and its variants
    conf_name = config.get('blue_col_name', '蓝球')
    for p in [conf_name, '蓝球', '篮球']:
        if p in df.columns and p not in blue_cols:
            blue_cols.append(p)
    
    # 2. Check for numbered variants (e.g., 蓝球1, 蓝球2)
    for i in range(1, config.get('blue_count', 1) + 1):
        for p in [conf_name, '蓝球', '篮球']:
            cname = f"{p}{i}"
            if cname in df.columns and cname not in blue_cols:
                blue_cols.append(cname)
                break
                
    # 3. Fallback: search all columns containing '蓝球' or '篮球' if still empty
    if not blue_cols:
        blue_cols = [c for c in df.columns if ('蓝球' in c or '篮球' in c) and '奖金' not in c and '注数' not in c]

    if not blue_cols:
        st.info("数据源中未找到蓝球相关字段")
        return

    # Optimized calculation: flatten and convert to int
    all_vals = df[blue_cols].values.flatten()
    # Filter out NaNs and convert to int
    valid_vals = [int(float(v)) for v in all_vals if pd.notna(v)]
    
    if not valid_vals:
        st.info("蓝球数据为空")
        return
        
    counts = Counter(valid_vals)
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
    possible_ratios = []
    ratio_counts = {}
    for i in range(red_count + 1):
        ratio_str = f"{i}:{red_count - i}"
        possible_ratios.append(ratio_str)
        ratio_counts[ratio_str] = 0
    for val in df['奇数']:
        if pd.isna(val): continue
        odds = int(val)
        evens = red_count - odds
        if 0 <= odds <= red_count:
            ratio = f"{odds}:{evens}"
            ratio_counts[ratio] = ratio_counts.get(ratio, 0) + 1
    df_plot = pd.DataFrame([{"奇偶比例": k, "出现次数": v} for k, v in ratio_counts.items()])
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
    potential_cols = [f"{CN_KEYS[k]}连" for k in range(2, min(int(red_count) + 1, 21))]
    return [c for c in potential_cols if c in df.columns]

def render_chart_consecutive_dist(df, config):
    st.subheader("连号分布分析")
    cols = _get_consecutive_cols(df, config['red_count'])
    if not cols: 
        st.info("暂无连号统计数据")
        return
    counts = {c: df[c].sum() for c in cols}
    max_idx = -1
    for i, c in enumerate(cols):
        if counts[c] > 0: max_idx = i
    if max_idx == -1:
        st.info("本期所选数据范围内未发现连号组合")
        return
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
    text = base.mark_text(align='center', baseline='bottom', dx=15).encode(
        x='期号:O', y='连号类型:N', text='连号组数:Q', color=alt.value('black')
    )
    st.altair_chart((points + text).properties(title='红球连号趋势'), use_container_width=True)

def _get_jump_cols(df, red_count):
    CN_KEYS = ["", "", "二", "三", "四", "五", "六", "七", "八", "九", "十", 
               "十一", "十二", "十三", "十四", "十五", "十六", "十七", "十八", "十九", "二十"]
    potential_cols = [f"{CN_KEYS[k]}跳" for k in range(2, min(int(red_count) + 1, 21))]
    return [c for c in potential_cols if c in df.columns]

def render_chart_jump_dist(df, config):
    st.subheader("跳号分布分析")
    cols = _get_jump_cols(df, config['red_count'])
    if not cols:
        st.info("暂无跳号统计数据")
        return
    counts = {c: df[c].sum() for c in cols}
    max_idx = -1
    for i, c in enumerate(cols):
        if counts[c] > 0: max_idx = i
    if max_idx == -1:
        st.info("所选数据范围内未发现跳号组合")
        return
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
    present_cols = [c for c in cols if c in df_trend['跳号类型'].unique()]
    base = alt.Chart(df_trend).properties(width=800, height=300)
    points = base.mark_circle(size=60).encode(
        x=alt.X('期号:O', title='期号', axis=alt.Axis(labelAngle=-45)),
        y=alt.Y('跳号类型:N', title='跳号类型', sort=present_cols),
        color=alt.Color('跳号类型:N', title='跳号类型', scale=alt.Scale(domain=present_cols), legend=None),
        tooltip=['期号', '跳号类型', '跳号组数']
    )
    text = base.mark_text(align='center', baseline='bottom', dx=15).encode(
        x='期号:O', y='跳号类型:N', text='跳号组数:Q', color=alt.value('black')
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
    text = chart.mark_text(align='center', baseline='bottom', dy=-5).encode(text=alt.Text('百分比:Q', format='.1%'))
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
        color=alt.Color('同尾类型:N', scale=alt.Scale(domain=list(tail_map.values()))),
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
    if 'AC' not in df.columns: return
    ac_values = df['AC'].dropna()
    if ac_values.empty: return
    min_ac, max_ac = int(ac_values.min()), int(ac_values.max())
    padding = max(1, int((max_ac - min_ac) * 0.1))
    chart = alt.Chart(df).mark_line().encode(
        x=alt.X('期号:O', axis=alt.Axis(labelAngle=-45)),
        y=alt.Y('AC:Q', scale=alt.Scale(domain=[max(0, min_ac - padding), max_ac + padding])),
        color=alt.value('#1E90FF'),
        tooltip=['期号', 'AC']
    ).properties(title='红球 AC 值趋势图', width=800, height=300)
    text = chart.mark_text(align='center', baseline='bottom', dy=-5, color='black').encode(text=alt.Text('AC:Q', format='.0f'))
    st.altair_chart(chart + text, use_container_width=True)

def render_chart_hot_pairs(df, config):
    st.subheader("🔥 热门号码对")
    red_cols = [f"{config['red_col_prefix']}{i}" for i in range(1, config['red_count'] + 1)]
    freq_df = analyze_top_companion_pairs(df[red_cols], red_cols=red_cols, top_n=10)
    if freq_df.empty: return
    bars = alt.Chart(freq_df).mark_bar().encode(
        x=alt.X('号码对:O', title='热门号码对', sort='-y', axis=alt.Axis(labelAngle=-45)),
        y=alt.Y('出现次数:Q', title='出现总次数'),
        color=alt.Color('出现次数:Q', scale=alt.Scale(scheme='reds'), legend=None),
        tooltip=['号码对', '出现次数', alt.Tooltip('百分比:Q', format=".1%")]
    ).properties(width=800, height=300)
    text = bars.mark_text(align='center', baseline='bottom', dy=-5).encode(text=alt.Text('百分比:Q', format=".1%"))
    st.altair_chart(bars + text, use_container_width=True)

def render_chart_hot_triples(df, config):
    st.subheader("🔥 热门号码三元组")
    red_cols = [f"{config['red_col_prefix']}{i}" for i in range(1, config['red_count'] + 1)]
    freq_df = analyze_top_triples(df[red_cols], red_cols=red_cols, top_n=10)
    if freq_df.empty: return
    bars = alt.Chart(freq_df).mark_bar(color='red').encode(
        x=alt.X('号码三元组:O', title='热门号码三元组', sort='-y', axis=alt.Axis(labelAngle=-45)),
        y=alt.Y('出现次数:Q', title='出现总次数'),
        color=alt.Color('出现次数:Q', scale=alt.Scale(scheme='reds'), legend=None),
        tooltip=['号码三元组', '出现次数', alt.Tooltip('百分比:Q', format=".1%")]
    ).properties(width=800, height=300)
    text = bars.mark_text(align='center', baseline='bottom', dy=-5).encode(text=alt.Text('百分比:Q', format=".1%"))
    st.altair_chart(bars + text, use_container_width=True)
