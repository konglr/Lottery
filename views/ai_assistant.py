import streamlit as st
import pandas as pd
import json
import os
from funcs.ai_helper import (
    prepare_lottery_data_text,
    generate_ai_prediction,
    format_ai_response
)

def render_ai(df, config):
    st.subheader(f"🤖 AI 预测助手 ({st.session_state.get('ai_brand', 'Gemini')})")
    brand = st.session_state.get("ai_brand", "Gemini")
    model_name = st.session_state.get("ai_model", "gemini-2.0-flash")
    
    env_keys = {
        "Gemini": "GEMINI_API_KEY", "NVIDIA": "NV_API_KEY",
        "MiniMax": "MINIMAX_API_KEY", "DashScope": "ALIYUNCS_API_KEY"
    }
    env_key_name = env_keys.get(brand)
    key = os.getenv(env_key_name, "")
    
    if st.button("开始分析并预测"):
        if not key:
            st.error(f"请在系统环境变量中设置 {env_key_name}")
            return
        try:
            data_str = prepare_lottery_data_text(df, config)
            with st.expander("查看发送给 AI 的原始指令 (Prompt)"):
                st.info(f"正在配置 {brand} / {model_name} 进行分析...")
                st.text_area("数据内容:", data_str, height=200)
            
            with st.status("AI 正在深度分析中...", expanded=True) as status:
                prediction = generate_ai_prediction(brand, model_name, key, data_str, config)
                status.update(label="分析完成！", state="complete", expanded=False)
                
            st.markdown("### 📊 AI 预测建议")
            raw_content = prediction
            if "分析结果" in raw_content:
                thinking, result = raw_content.split("分析结果", 1)
                with st.expander("🤔 思考过程"): st.markdown(thinking.strip())
                st.markdown("### 📋 深度分析报告")
                st.markdown(result.strip().replace('\\n', '\n'))
            else:
                thinking, result = format_ai_response(raw_content)
                if thinking:
                    with st.expander("🤔 查看 AI 思考过程"): st.markdown(thinking.strip())
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
        df_hist = pd.read_csv(csv_file)
        df_hist = df_hist[df_hist['lottery'] == config['name']]
        if df_hist.empty:
            st.info(f"暂无 {config['name']} 的 AI 预测记录。")
            return
        df_hist['target_period'] = df_hist['target_period'].astype(str)
        periods = sorted(df_hist['target_period'].unique(), reverse=True)
        sel_period = st.selectbox("📅 选择预测期号", periods, key="analysis_period_sel")
        df_period = df_hist[df_hist['target_period'] == sel_period]
        df_period = df_period.sort_values('timestamp', ascending=False).drop_duplicates('model')
        
        draw_row = df[df['期号'].astype(str) == str(sel_period)]
        winning_reds, winning_blues = [], []
        if not draw_row.empty:
            row_draw = draw_row.iloc[0]
            for i in range(1, config['red_count'] + 1):
                col = f"{config['red_col_prefix']}{i}"
                if col in row_draw: winning_reds.append(int(row_draw[col]))
            if config['has_blue']:
                prefix = config.get('blue_col_name', '蓝球')
                if config['blue_count'] == 1:
                    for p in [prefix, '蓝球', '篮球']:
                        if p in row_draw:
                            winning_blues.append(int(row_draw[p]))
                            break
                else:
                    for i in range(1, config['blue_count'] + 1):
                        for p in [prefix, '蓝球', '篮球']:
                            cname = f"{p}{i}"
                            if cname in row_draw:
                                winning_blues.append(int(row_draw[cname]))
                                break
            st.markdown(f"#### 📅 {sel_period}期 实际开奖结果")
            res_html = ""
            for r in winning_reds: res_html += f'<div class="lottery-ball red-ball">{r}</div>'
            for b in winning_blues: res_html += f'<div class="lottery-ball blue-ball">{b}</div>'
            st.markdown(res_html, unsafe_allow_html=True)
        else:
            st.markdown(f"#### 📅 {sel_period}期 开奖状态: `⏳ 等待开奖`")
        st.divider()

        models = df_period['model'].unique()
        cols = st.columns(len(models))
        for i, model in enumerate(models):
            with cols[i]:
                row = df_period[df_period['model'] == model].iloc[0]
                st.markdown(f"#### 🤖 {model}")
                st.caption(f"🕒 {row['timestamp']}")
                try:
                    recs = json.loads(row['recommendations'])
                    if recs.get('dan'):
                        st.markdown(f"**📍 核心胆码**")
                        dan_html = ""
                        for d in recs['dan']:
                            is_hit = "hit-ball" if int(d) in winning_reds else ""
                            dan_html += f'<span class="lottery-ball red-ball {is_hit}" style="width:30px; height:30px; line-height:30px; font-size:0.8em;">{d}</span>'
                        st.markdown(dan_html, unsafe_allow_html=True)
                    if recs.get('groups'):
                        st.markdown("**💡 推荐组合**")
                        for idx, g in enumerate(recs['groups']):
                            with st.expander(f"方案 {idx+1}", expanded=(idx==0)):
                                reds_html = "🔴"
                                for r in g.get('reds', []):
                                    is_hit = "background-color:gold; color:black; font-weight:bold; padding:2px 4px; border-radius:3px;" if int(r) in winning_reds else ""
                                    reds_html += f' <code style="{is_hit}">{r:02d}</code>'
                                st.markdown(reds_html, unsafe_allow_html=True)
                                if g.get('blues'):
                                    blues_html = "🔵"
                                    for b in g['blues']:
                                        is_hit = "background-color:gold; color:black; font-weight:bold; padding:2px 4px; border-radius:3px;" if int(b) in winning_blues else ""
                                        blues_html += f' <code style="{is_hit}">{b:02d}</code>'
                                    st.markdown(blues_html, unsafe_allow_html=True)
                                if g.get('reason'): st.caption(f"分析: {g['reason']}")
                    if recs.get('kl8_numbers'):
                        st.markdown("**🔢 快乐8 选二十**")
                        st.write(", ".join([f"{n:02d}" for n in sorted(recs['kl8_numbers'])]))
                    with st.expander("📄 查看完整分析报告"):
                        thinking, result = format_ai_response(row.get('raw_response', ""))
                        if thinking: st.markdown(f"**思考过程:**\n{thinking}")
                        st.markdown(result.replace('\\n', '\n'))
                except Exception as e: st.error(f"解析预测数据失败: {e}")
        st.divider()
    except Exception as e: st.error(f"加载分析数据失败: {e}")
