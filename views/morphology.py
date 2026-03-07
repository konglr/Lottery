import streamlit as st
import pandas as pd
import numpy as np
import itertools
import os
from collections import Counter
from funcs.ball_filter import calculate_morphology_score, get_morphology_report

def render_morphological_analysis(df_full, conf):
    st.subheader(f"🎯 {conf['name']} 形态学智能优选")
    csv_file = f"data/{conf['code']}_backtest.csv"
    if not os.path.exists(csv_file):
        st.warning(f"⚠️ 暂无回测数据，请先运行预测脚本。")
        return
    try:
        df_back = pd.read_csv(csv_file)
        if df_back.empty: return
        df_back = df_back[df_back['Target_Period'] != 'Target_Period']
    except pd.errors.ParserError:
        st.error(f"⚠️ 数据文件格式冲突: {csv_file}")
        st.warning("检测到 CSV 表头与数据列数不一致（可能因为模型增减导致）。")
        st.info("请在后台手动删除该文件，然后重新运行预测脚本以生成新文件。")
        return
    except Exception as e:
        st.error(f"读取数据失败: {e}")
        return

    run_times = sorted(df_back['Run_Time'].unique(), reverse=True)
    sel_run_time = st.selectbox("📅 1. 选择预测执行批次 (按时间)", run_times, key="morph_run_time")
    df_run = df_back[df_back['Run_Time'] == sel_run_time]
    periods = sorted(df_run['Target_Period'].unique(), reverse=True)
    sel_period = st.selectbox("🔢 2. 选择目标预测期号", periods, key="morph_period")
    row_data = df_run[df_run['Target_Period'] == sel_period].iloc[0]

    col1, col2, col3 = st.columns(3)
    with col1:
        model_options = {'Prob_A': '模型 A (统计)', 'Prob_B': '模型 B (RF)', 'Prob_C': '模型 C (XGB)', 'Prob_D': '模型 D (LSTM)', 'Prob_E': '模型 E (LGBM)', 'Prob_F': '模型 F (CatBoost)', 'Prob_G': '模型 G (HMM)', 'Prob_H': '模型 H (EVT)', 'Prob_I': '模型 I (GA)', 'Prob_J': '模型 J (Poisson)'}
        sel_model = st.selectbox("选择评分基准模型", list(model_options.keys()), format_func=lambda x: model_options[x], index=1)
    with col2: top_n_balls = st.slider("候选红球池数量", 10, 20, 15)
    with col3: num_rec = st.slider("推荐组合数量", 1, 10, 5)

    st.info(f"📋 **当前方案详情**: 目标期号 `{sel_period}` | 源模型 `{model_options[sel_model]}` | 预测生成于 `{sel_run_time}`")

    actual_red = []
    p_match = df_full[df_full['期号'].astype(str) == str(sel_period)]
    if not p_match.empty:
        r_cols = [f"{conf['red_col_prefix']}{i}" for i in range(1, conf['red_count'] + 1)]
        actual_red = sorted(p_match.iloc[0][r_cols].values.astype(int).tolist())
        st.write(f"✅ **该期实际开奖**: " + " ".join([f"<span style='color:#ff4b4b; font-weight:bold;'>{r:02d}</span>" for r in actual_red]), unsafe_allow_html=True)
    else: st.write("✨ **该期尚未开奖 (未来预测)**")

    r_start, r_end = conf['red_range']
    # Handle both new R-prefix and old simple prefix
    prefix_map = {'Prob_A': 'A', 'Prob_B': 'B', 'Prob_C': 'C', 'Prob_D': 'D', 'Prob_E': 'E', 'Prob_F': 'F', 'Prob_G': 'G', 'Prob_H': 'H', 'Prob_I': 'I', 'Prob_J': 'J'}
    mid = prefix_map[sel_model]
    prob_cols = [f"Prob_{mid}_R{i:02d}" for i in range(r_start, r_end + 1)]
    if not all(c in row_data.index for c in prob_cols):
        prob_cols = [f"Prob_{mid}_{i:02d}" for i in range(r_start, r_end + 1)]
    
    if not any(c in row_data.index for c in prob_cols):
        st.error("所选模型的概率数据缺失")
        return

    probs = pd.to_numeric(row_data[prob_cols]).values
    top_indices = np.argsort(probs)[-top_n_balls:][::-1]
    top_balls = sorted([int(prob_cols[i].split('_')[-1].replace('R','')) for i in top_indices])
    
    ball_html = []
    hit_count = 0
    for b in top_balls:
        if b in actual_red:
            ball_html.append(f"<span style='color:#28a745; font-weight:bold; border-bottom:2px solid #28a745;'>{b:02d}</span>")
            hit_count += 1
        else: ball_html.append(f"<span>{b:02d}</span>")
    st.write(f"💡 **锁定 Top {top_n_balls} 核心候选球 (命中: {hit_count})**: " + " ".join(ball_html), unsafe_allow_html=True)

    if df_full.empty: return
    target_row = df_full[df_full['期号'].astype(str) == str(sel_period)]
    if not target_row.empty:
        target_date = target_row['开奖日期'].iloc[0]
        history_row = df_full[df_full['开奖日期'] < target_date].head(1)
    else: history_row = df_full.head(1)
    if history_row.empty: history_row = df_full.iloc[0]
    
    last_period_nums = sorted([int(history_row[f"{conf['red_col_prefix']}{i}"].iloc[0] if isinstance(history_row, pd.DataFrame) else history_row[f"{conf['red_col_prefix']}{i}"]) for i in range(1, conf['red_count'] + 1)])
    last_issue = history_row['期号'].iloc[0] if not history_row.empty else "未知"
    st.write(f"📏 **形态参考基准**: 以 `{last_issue}` 期号码 `{', '.join([f'{x:02d}' for x in last_period_nums])}` 为参照")

    if st.button("🚀 开始形态学漏斗筛选"):
        with st.status("正在进行数千种组合的物理分布扫描...", expanded=True) as status:
            all_combos = list(itertools.combinations(top_balls, conf['red_count']))
            scored_data = []
            for combo in all_combos:
                c_list = list(combo)
                score = calculate_morphology_score(c_list, last_period_nums, conf['name'])
                scored_data.append((c_list, score))
            scored_data.sort(key=lambda x: x[1], reverse=True)
            score_counts = Counter([d[1] for d in scored_data])
            total_c = len(scored_data)
            dist_str = " | ".join([f"**{s}分**: {count}组({count/total_c:.1%})" for s, count in sorted(score_counts.items(), reverse=True)])
            status.update(label=f"扫描完成！从 {total_c} 组中筛选出最优方案。", state="complete")
            st.markdown(f"📊 **组合得分分布**: {dist_str}")

        st.success(f"已为您筛选出得分最高的 {num_rec} 组精品组合：")
        for i in range(min(num_rec, len(scored_data))):
            combo, score = scored_data[i]
            with st.expander(f"🏆 {sel_period}期 推荐组合 {i+1} (得分: {score}/100) - {', '.join([f'{x:02d}' for x in combo])}"):
                report = get_morphology_report(combo, last_period_nums, conf['name'])
                st.code(report, language="text")
                st.write(f"**投注建议:** `{', '.join([f'{x:02d}' for x in combo])}`")
