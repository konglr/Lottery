import streamlit as st
import pandas as pd
import logging
from config import LOTTERY_CONFIG
from funcs.ai_helper import load_renviron
from views.components import load_full_data, render_sidebar, render_metrics
from views.trend_analysis import (
    render_chart_red_freq, render_chart_blue_freq, render_chart_odd_even_ratio,
    render_chart_odd_even_trend, render_chart_consecutive_dist, render_chart_consecutive_trend,
    render_chart_jump_dist, render_chart_jump_trend, render_chart_tail_dist,
    render_chart_tail_trend, render_chart_size_ratio, render_chart_size_trend,
    render_chart_zone_dist, render_chart_zone_trend, render_chart_repeat_dist,
    render_chart_repeat_trend, render_chart_neighbor_dist, render_chart_neighbor_trend,
    render_chart_isolated_dist, render_chart_isolated_trend, render_chart_sum_trend,
    render_chart_span_trend, render_chart_ac_trend, render_chart_hot_pairs, render_chart_hot_triples
)
from views.ai_assistant import render_ai, render_ai_analysis
from views.morphology import render_morphological_analysis
from views.backtest import render_backtest_results

# Load environment variables
load_renviron()

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(), logging.FileHandler('my_log_file.log')]
)

def main():
    st.set_page_config(page_title="彩票大数据分析系统", layout="wide", page_icon="📊")
    
    # CSS Styles
    st.markdown("""
    <style>
    .lottery-ball { display: inline-block; width: 40px; height: 40px; border-radius: 50%; text-align: center; line-height: 40px; margin: 5px; font-weight: bold; color: white; }
    .red-ball { background-color: #ff5b5b; }
    .blue-ball { background-color: #5b9fff; }
    .hit-ball { border: 3px solid #ffd700 !important; box-shadow: 0 0 10px #ffd700; transform: scale(1.1); }
    </style>
    """, unsafe_allow_html=True)

    # 1. Lottery Selection & Config
    sel_lottery = st.sidebar.selectbox("选择彩票类型", list(LOTTERY_CONFIG.keys()))
    conf = LOTTERY_CONFIG[sel_lottery]
    conf['name'] = sel_lottery
    
    # 2. Sidebar Analysis Options
    p_limit = render_sidebar(conf)
    
    # 3. Data Loading
    df_full = load_full_data(sel_lottery)
    if df_full.empty:
        st.error(f"无法加载 {sel_lottery} 的历史数据，请检查 CSV 文件。")
        return
    df = df_full.head(p_limit)
    
    st.title(f"📊 {sel_lottery} 大数据分析预测系统")
    
    # 4. Tab Navigation
    tabs = st.tabs(["📈 趋势分析", "📊 预测历史", "🤖 AI 实时预测", "🎯 形态优选", "📋 原始数据", "🔙 回测结果"])
    
    with tabs[0]:
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
                if ck in c_map: 
                    c_map[ck](df, conf)
                    st.divider()

    with tabs[1]: render_ai_analysis(df_full, conf)
    with tabs[2]: render_ai(df, conf)
    with tabs[3]: render_morphological_analysis(df_full, conf)
    with tabs[4]: st.dataframe(df_full, use_container_width=True)
    with tabs[5]: render_backtest_results(df_full, conf)

if __name__ == "__main__":
    main()
