import streamlit as st
import pandas as pd
import os
import logging
from config import LOTTERY_CONFIG

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
            df['期号'] = df['期号'].astype(str).str.replace(r'\.0$', '', regex=True)
            if '开奖日期' in df.columns:
                df['开奖日期'] = pd.to_datetime(df['开奖日期'], errors='coerce')
                df = df.sort_values(['开奖日期', '期号'], ascending=[False, False])
            else:
                df = df.sort_values('期号', ascending=False)
        return df.head(100)
    except Exception as e:
        logging.error(f"Error loading {lottery_name}: {e}")
        return pd.DataFrame()

def render_sidebar(config):
    st.sidebar.title(f"{config['name']}分析选项")
    period = st.sidebar.slider("分析期数", 5, 100, 30, 5)
    
    from funcs.ai_helper import get_brand_models
    st.sidebar.divider()
    st.sidebar.subheader("🤖 AI 助手配置")
    brand_models = get_brand_models()
    ai_brand = st.sidebar.selectbox("AI 模型品牌", list(brand_models.keys()), index=0)
    ai_model = st.sidebar.selectbox("具体模型选择", brand_models[ai_brand], index=0)
    
    st.session_state.ai_brand = ai_brand
    st.session_state.ai_model = ai_model
    return period

def render_metrics(df, config):
    if df.empty: return
    row = df.iloc[0]
    st.subheader("最新开奖信息")
    col_left, col_right = st.columns([1, 1], gap="large")
    
    with col_left:
        c1, c2, c3, c4 = st.columns([1, 1.5, 2, 2])
        with c1: st.metric("期号", str(row['期号']))
        display_date = str(row['开奖日期'])
        if ' ' in display_date: display_date = display_date.split(' ')[0]
        elif 'T' in display_date: display_date = display_date.split('T')[0]
        with c2: st.metric("开奖日期", display_date)
        if '本期销售金额' in row and not pd.isna(row['本期销售金额']):
            with c3: st.metric("本期销售金额", f"{float(row['本期销售金额']):,.0f}")
        if '奖池累计金额' in row and not pd.isna(row['奖池累计金额']):
            with c4: st.metric("奖池累计金额", f"{float(row['奖池累计金额']):,.0f}")
        
        balls_html = ""
        for i in range(1, config['red_count']+1):
            col = f"{config['red_col_prefix']}{i}"
            if col in df.columns: balls_html += f'<div class="lottery-ball red-ball">{int(row[col])}</div>'
        
        bcols = []
        if config['has_blue']:
            prefix = config.get('blue_col_name', '蓝球')
            if config['blue_count'] == 1:
                found = False
                for p in [prefix, '蓝球', '篮球']:
                    if p in df.columns:
                        bcols.append(p)
                        found = True
                        break
            else:
                for i in range(1, config['blue_count'] + 1):
                    found = False
                    for p in [prefix, '蓝球', '篮球']:
                        cname = f"{p}{i}"
                        if cname in df.columns:
                            bcols.append(cname)
                            found = True
                            break
            
            for c in bcols:
                balls_html += f'<div class="lottery-ball blue-ball">{int(row[c])}</div>'
        st.markdown(balls_html, unsafe_allow_html=True)

    with col_right:
        prize_cols = [c for c in df.columns if c.endswith('注数') and not c.endswith('追加注数')]
        if prize_cols:
            st.markdown("##### 🏆 中奖详情")
            kl8_mode = None
            if config['code'] == 'kl8':
                modes = ["选一", "选二", "选三", "选四", "选五", "选六", "选七", "选八", "选九", "选十"]
                kl8_mode = st.pills("玩法选择", modes, selection_mode="single", default="选十")
            
            std_prize_map = {
                'ssq': ["一等奖", "二等奖", "三等奖", "四等奖", "五等奖", "六等奖"],
                'dlt': ["一等奖", "二等奖", "三等奖", "四等奖", "五等奖", "六等奖", "七等奖"],
                'd3': ["单选", "组三", "组六"], 'pl3': ["直选", "组选三", "组选六"], 'pl5': ["直选"],
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
                if kl8_mode and not p.startswith(kl8_mode): continue
                num = row.get(f"{p}注数", 0)
                money = row.get(f"{p}奖金", 0)
                if num > 0 or money > 0 or p in p_order:
                    item = {"奖项": p, "中奖注数": int(num) if pd.notna(num) else 0, "单注奖金": int(float(money)) if pd.notna(money) else 0}
                    if f"{p}追加注数" in df.columns:
                        item["追加注数"] = int(row.get(f"{p}追加注数", 0)) if pd.notna(row.get(f"{p}追加注数")) else 0
                        item["追加奖金"] = int(float(row.get(f"{p}追加奖金", 0))) if pd.notna(row.get(f"{p}追加奖金")) else 0
                    p_data.append(item)
            if p_data:
                df_display = pd.DataFrame(p_data)
                st.dataframe(df_display.style.format("{:,.0f}", subset=[c for c in ["中奖注数", "单注奖金", "追加注数", "追加奖金"] if c in df_display.columns]), hide_index=True, use_container_width=True)
    st.markdown("---")
