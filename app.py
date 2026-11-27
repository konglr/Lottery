import sys
import importlib
import os
import streamlit as st
import pandas as pd
import logging
from config import LOTTERY_CONFIG
from funcs.ai_helper import load_renviron

# Ensure submodules are refreshed from disk if cached in Streamlit Cloud's running worker
try:
    import views.components
    if 'views.components' in sys.modules:
        importlib.reload(views.components)
    from views.components import load_full_data, render_sidebar, render_metrics, render_download_button
except (ImportError, AttributeError):
    try:
        if 'views.components' in sys.modules:
            importlib.reload(sys.modules['views.components'])
        from views.components import load_full_data, render_sidebar, render_metrics, render_download_button
    except (ImportError, AttributeError):
        from views.components import load_full_data, render_sidebar, render_metrics
        def render_download_button(lottery_name, key="btn_download", label=None, use_container_width=False):
            conf = LOTTERY_CONFIG.get(lottery_name, {})
            fp = conf.get("data_file")
            if fp and os.path.exists(fp):
                try:
                    with open(fp, "rb") as f:
                        csv_data = f.read()
                    return st.download_button(
                        label=label or f"📥 下载{lottery_name}历史数据 (CSV)",
                        data=csv_data,
                        file_name=f"{lottery_name}_历史开奖数据.csv",
                        mime="text/csv",
                        key=key,
                        use_container_width=use_container_width
                    )
                except Exception:
                    pass
            st.button("暂无数据", disabled=True, key=key, use_container_width=use_container_width)
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
from views.seo_helper import inject_seo_meta, render_share_and_stats, render_seo_guide_and_faq


# Load environment variables
load_renviron()

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(), logging.FileHandler('my_log_file.log')]
)

def main():
    st.set_page_config(
        page_title="双色球/大乐透/快乐8开奖走势图与AI预测 - 彩票大数据系统",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
            'About': '''
            ### 🎰 彩票大数据分析与AI智能预测系统
            
            基于官方真实开奖数据与大语言模型（Gemini / 通义千问 / MiniMax）的综合分析平台。
            
            - **支持彩种**：双色球 (SSQ)、超级大乐透 (DLT)、快乐8 (KL8)、福彩3D、排列三/五、七乐彩、七星彩
            - **技术指标**：20+项冷热频次、连号跳号、奇偶形态、和值跨度图表分析
            - **AI 智能预测**：形态学规则约束与历史全自动回测验证
            
            *免责声明：本系统仅供数据建模与研究学习，不构成任何购彩建议。*
            '''
        }
    )
    
    # CSS Styles
    st.markdown("""
    <style>
    .lottery-ball { display: inline-block; width: 40px; height: 40px; border-radius: 50%; text-align: center; line-height: 40px; margin: 5px; font-weight: bold; color: white; }
    .red-ball { background-color: #ff5b5b; }
    .blue-ball { background-color: #5b9fff; }
    .hit-ball { border: 3px solid #ffd700 !important; box-shadow: 0 0 10px #ffd700; transform: scale(1.1); }
    </style>
    """, unsafe_allow_html=True)

    # 1. Lottery Selection & URL Query Parameters Sync (Deep-linking for SEO)
    lottery_list = list(LOTTERY_CONFIG.keys())
    code_to_name = {c_conf["code"]: name for name, c_conf in LOTTERY_CONFIG.items()}
    
    default_index = 0
    requested_lottery = None
    try:
        if hasattr(st, "query_params") and "lottery" in st.query_params:
            requested_lottery = st.query_params["lottery"]
        elif hasattr(st, "experimental_get_query_params"):
            qp = st.experimental_get_query_params()
            if "lottery" in qp and qp["lottery"]:
                requested_lottery = qp["lottery"][0]
    except Exception:
        pass

    if requested_lottery:
        req_clean = str(requested_lottery).lower().strip()
        if req_clean in code_to_name:
            target_name = code_to_name[req_clean]
            if target_name in lottery_list:
                default_index = lottery_list.index(target_name)
        elif requested_lottery in lottery_list:
            default_index = lottery_list.index(requested_lottery)

    sel_lottery = st.sidebar.selectbox("选择彩票类型", lottery_list, index=default_index)
    conf = LOTTERY_CONFIG[sel_lottery]
    conf['name'] = sel_lottery

    # Sidebar quick download button
    render_download_button(
        sel_lottery,
        key="btn_download_sidebar",
        label=f"📥 下载【{sel_lottery}】历史数据 (CSV)",
        use_container_width=True
    )

    # Sync query parameter in browser address bar
    try:
        if hasattr(st, "query_params"):
            if st.query_params.get("lottery") != conf["code"]:
                st.query_params["lottery"] = conf["code"]
        elif hasattr(st, "experimental_set_query_params"):
            st.experimental_set_query_params(lottery=conf["code"])
    except Exception:
        pass

    # Inject SEO Dynamic Meta (Title, Description, Keywords, OG, Schema.org)
    inject_seo_meta(sel_lottery, conf)
    
    # 2. Sidebar Analysis Options
    p_limit = render_sidebar(conf)
    
    # 3. Data Loading
    df_full = load_full_data(sel_lottery)
    if df_full.empty:
        st.error(f"无法加载 {sel_lottery} 的历史数据，请检查 CSV 文件。")
        return
    df = df_full.head(p_limit)
    
    st.title(f"📊 {sel_lottery} 大数据分析预测系统")
    render_share_and_stats(df, conf)
    
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

    # 5. Semantic Guide & FAQ Section (Rich keywords for search engine indexing & user education)
    st.divider()
    render_seo_guide_and_faq(conf)

if __name__ == "__main__":
    main()
