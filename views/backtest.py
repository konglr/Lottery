import streamlit as st
import pandas as pd
import numpy as np
import os

def render_backtest_results(df_full, conf):
    st.markdown(f"### 📋 {conf['name']} 历史回测详情分析")
    csv_file = f"data/{conf['code']}_backtest.csv"
    if not os.path.exists(csv_file):
        st.warning(f"⚠️ 暂无回测数据 ({csv_file} 不存在)")
        return
    try:
        df_back = pd.read_csv(csv_file)
        if 'Target_Period' in df_back.columns:
            df_back['Target_Period'] = df_back['Target_Period'].astype(str).str.replace(r'\.0$', '', regex=True)
    except Exception as e:
        st.error(f"读取回测数据失败: {e}")
        return
    if df_back.empty:
        st.warning("⚠️ 回测数据为空")
        return

    separate_pool = conf.get('separate_pool', False)
    actual_periods_set = set(df_full['期号'].astype(str).tolist())

    def normalize_period(p_str):
        if p_str in actual_periods_set: return p_str
        if len(p_str) == 7 and p_str.startswith('20'):
            short = p_str[2:]
            if short in actual_periods_set: return short
        if len(p_str) == 5:
            long = '20' + p_str
            if long in actual_periods_set: return long
        return p_str
        
    run_times = sorted(df_back['Run_Time'].unique(), reverse=True)
    sel_run_time = st.selectbox("1. 选择回测执行时间", run_times)
    df_run = df_back[df_back['Run_Time'] == sel_run_time].sort_values("Target_Period", ascending=False)
    periods = df_run['Target_Period'].unique()
    
    def format_period_label(p):
        p_match = df_full[df_full['期号'].astype(str) == normalize_period(str(p))]
        if p_match.empty: return f"✨ 期号 {p} (预测下一期)"
        return f"🔙 期号 {p} (历史回测)"
    
    sel_period = st.selectbox("2. 选择回测目标期号", periods, format_func=format_period_label)

    st.markdown("#### 📊 本次运行汇总 (命中率概览)")
    summary_data = []
    methods = ['A', 'B', 'C', 'D']
    metrics = conf.get('eval_metrics', {"top_n_1": 6, "top_n_2": 10})
    n1, n2 = metrics['top_n_1'], metrics['top_n_2']
    
    red_cols = [f"{conf['red_col_prefix']}{i}" for i in range(1, conf['red_count'] + 1)]
    blue_cols = []
    if separate_pool:
        prefix = conf.get('blue_col_name', '蓝球')
        for i in range(1, conf.get('blue_count', 1) + 1):
            for p in [prefix, '蓝球', '篮球']:
                c = f"{p}{i}" if conf.get('blue_count', 1) > 1 else p
                if c in df_full.columns:
                    blue_cols.append(c)
                    break
    elif conf.get('has_blue'):
        col = conf.get('blue_col_name', '蓝球')
        for p in [col, '蓝球', '篮球']:
            if p in df_full.columns:
                red_cols.append(p)
                break

    for _, r_idx in df_run.iterrows():
        p = r_idx['Target_Period']
        p_str = normalize_period(str(p))
        p_match = df_full[df_full['期号'].astype(str) == p_str]
        if p_match.empty:
            summary_data.append({"回测期号": f"{p} ✨", "综合推荐": "-", "模型 A": "-", "模型 B": "-", "模型 C": "-", "模型 D": "-"})
            continue
        a_red = set(p_match.iloc[0][red_cols].values.astype(int).tolist())
        a_blue = set(p_match.iloc[0][blue_cols].values.astype(int).tolist()) if separate_pool else set()
        r_start, r_end = conf['red_range']
        b_start, b_end = conf.get('blue_range', (0, 0))
        r_list = list(range(r_start, r_end + 1))
        b_list = list(range(b_start, b_end + 1)) if separate_pool else []
        m_scores = {}
        for m in methods:
            r_p = []
            for n in r_list:
                col = f"Prob_{m}_R{n:02d}" if separate_pool else f"Prob_{m}_{n:02d}"
                try:
                    val = r_idx[col] if col in r_idx else 0.0
                    pv = float(val) if isinstance(val, (int, float)) or (isinstance(val, str) and val.replace('.', '').replace('-', '').replace('e', '').isdigit()) else 0.0
                except (ValueError, TypeError, KeyError):
                    pv = 0.0
                r_p.append(pv)
            b_p = []
            if separate_pool:
                for n in b_list:
                    col = f"Prob_{m}_B{n:02d}"
                    try:
                        val = r_idx[col] if col in r_idx else 0.0
                        pv = float(val) if isinstance(val, (int, float)) or (isinstance(val, str) and val.replace('.', '').replace('-', '').replace('e', '').isdigit()) else 0.0
                    except (ValueError, TypeError, KeyError):
                        pv = 0.0
                    b_p.append(pv)
            r_arr, b_arr = np.array(r_p), np.array(b_p)
            combined = np.concatenate([r_arr, b_arr])
            c_min, c_max = combined.min(), combined.max()
            m_scores[m] = (combined - c_min) / (c_max - c_min) if c_max > c_min else np.zeros_like(combined)
        ens_combined = np.mean([m_scores[m] for m in methods], axis=0)
        def get_hit_str(combined_scores, actual_r, actual_b):
            r_scores = combined_scores[:len(r_list)]
            b_scores = combined_scores[len(r_list):] if separate_pool else []
            r_top_n1 = [r_list[i] for i in r_scores.argsort()[::-1][:n1]]
            r_top_n2 = [r_list[i] for i in r_scores.argsort()[::-1][:n2]]
            hr1, hr2 = len(set(r_top_n1) & actual_r), len(set(r_top_n2) & actual_r)
            if separate_pool:
                b_top = [b_list[i] for i in b_scores.argsort()[::-1][:len(actual_b)]]
                hb = len(set(b_top) & actual_b)
                return f"R:{hr1}/{len(actual_r)}|B:{hb}/{len(actual_b)}"
            return f"{hr1}/{len(actual_r)}({n1})|{hr2}/{len(actual_r)}({n2})"
        p_row = {"回测期号": p_str, "综合推荐": get_hit_str(ens_combined, a_red, a_blue)}
        for m in methods: p_row[f"模型 {m}"] = get_hit_str(m_scores[m], a_red, a_blue)
        summary_data.append(p_row)
    if summary_data: st.table(pd.DataFrame(summary_data))
    st.divider()

    st.markdown(f"#### 🔍 期号 {sel_period} 详细回测对比")
    p_str = normalize_period(str(sel_period))
    p_match = df_full[df_full['期号'].astype(str) == p_str]
    actual_red = set(p_match.iloc[0][red_cols].values.astype(int).tolist()) if not p_match.empty else set()
    actual_blue = set(p_match.iloc[0][blue_cols].values.astype(int).tolist()) if not p_match.empty and separate_pool else set()
    if not p_match.empty:
        res_html = "**实际开奖:** "
        for r in sorted(list(actual_red)): res_html += f'<span class="lottery-ball red-ball">{r}</span>'
        for b in sorted(list(actual_blue)): res_html += f'<span class="lottery-ball blue-ball">{b}</span>'
        st.markdown(res_html, unsafe_allow_html=True)
    else: st.info("尚未开奖 (未来预测)")

    st.subheader(f"🔴 红球排名对比 (Top {n2+2})")
    row_idx = df_run[df_run['Target_Period'] == sel_period].iloc[0]
    r_start, r_end = conf['red_range']
    r_list = list(range(r_start, r_end + 1))
    rank_data = []
    for n in r_list:
        item = {'Number': n}
        scores = []
        for m in methods:
            col = f"Prob_{m}_R{n:02d}" if separate_pool else f"Prob_{m}_{n:02d}"
            try:
                val = row_idx[col] if col in row_idx else 0.0
                p = float(val) if isinstance(val, (int, float)) or (isinstance(val, str) and val.replace('.', '').replace('-', '').replace('e', '').isdigit()) else 0.0
            except (ValueError, TypeError, KeyError):
                p = 0.0
            item[f'P_{m}'] = p
            scores.append(p)
        item['Ens'] = np.mean(scores)
        rank_data.append(item)
    df_rank = pd.DataFrame(rank_data)
    cols = st.columns(len(methods) + 1)
    disp_names = ["综合推荐"] + [f"模型 {m}" for m in methods]
    sort_cols = ['Ens'] + [f'P_{m}' for m in methods]
    for i, (name, s_col) in enumerate(zip(disp_names, sort_cols)):
        with cols[i]:
            st.markdown(f"**{name}**")
            top_df = df_rank.sort_values(s_col, ascending=False).head(n2+2)
            for idx, r in top_df.iterrows():
                num = int(r['Number']); is_hit = "✅" if num in actual_red else ""
                st.write(f"{is_hit} `{num:02d}` ({r[s_col]:.1%})")

    if separate_pool:
        st.subheader(f"🔵 蓝球排名对比")
        b_start, b_end = conf['blue_range']
        b_list = list(range(b_start, b_end + 1))
        b_rank_data = []
        for n in b_list:
            item = {'Number': n}; scores = []
            for m in methods:
                col = f"Prob_{m}_B{n:02d}"
                try:
                    val = row_idx[col] if col in row_idx else 0.0
                    p = float(val) if isinstance(val, (int, float)) or (isinstance(val, str) and val.replace('.', '').replace('-', '').replace('e', '').isdigit()) else 0.0
                except (ValueError, TypeError, KeyError):
                    p = 0.0
                item[f'P_{m}'] = p; scores.append(p)
            item['Ens'] = np.mean(scores); b_rank_data.append(item)
        df_b_rank = pd.DataFrame(b_rank_data)
        b_cols = st.columns(len(methods) + 1)
        for i, (name, s_col) in enumerate(zip(disp_names, sort_cols)):
            with b_cols[i]:
                top_b = df_b_rank.sort_values(s_col, ascending=False).head(max(2, len(actual_blue)))
                for idx, r in top_b.iterrows():
                    num = int(r['Number']); is_hit = "🎯" if num in actual_blue else ""
                    st.write(f"{is_hit} `{num:02d}` ({r[s_col]:.1%})")
