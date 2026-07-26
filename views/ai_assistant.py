import streamlit as st
import pandas as pd
import json
import os
import re
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
        "Gemini": "GEMINI_API_KEY", "DeepSeek": "DEEPSEEK_API_KEY", "NVIDIA": "NV_API_KEY",
        "MiniMax": "MINIMAX_API_KEY", "DashScope": "ALIYUNCS_API_KEY"
    }
    env_key_name = env_keys.get(brand)
    key = os.getenv(env_key_name, "") if env_key_name else ""
    
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

def read_predictions_history_csv(csv_file):
    try:
        return pd.read_csv(csv_file)
    except Exception as e:
        st.warning("由于历史数据文件格式有些许异常，已自动启用兼容模式加载数据。")
        
        def split_recommendations(group4):
            brace_count = 0
            in_str = False
            escape = False
            i = 0
            n = len(group4)
            while i < n and group4[i] not in ('{', '['):
                i += 1
            
            while i < n:
                c = group4[i]
                if escape:
                    escape = False
                    i += 1
                    continue
                if c == '\\':
                    escape = True
                    i += 1
                    continue
                if c == '"':
                    in_str = not in_str
                    i += 1
                    continue
                if not in_str:
                    if c == '{':
                        brace_count += 1
                    elif c == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            end_idx = i
                            break
                i += 1
            else:
                return None, None
            
            next_idx = end_idx + 1
            while next_idx < n and group4[next_idx] != ',':
                next_idx += 1
            
            recs_full = group4[:next_idx]
            rest = group4[next_idx+1:]
            return recs_full, rest

        def split_rest(rest):
            rest = rest.strip()
            if rest.endswith(','):
                return rest[:-1], ""
            
            keywords = [',Lucky', ',用户', ',后验', ',源=', ',回测', ',"Lucky', ',"用户', ',"源=']
            best_idx = -1
            for kw in keywords:
                idx = rest.rfind(kw)
                if idx > best_idx:
                    best_idx = idx
                    
            if best_idx != -1:
                raw_resp = rest[:best_idx]
                remark = rest[best_idx+1:]
                remark = remark.strip()
                if remark.startswith('"') and remark.endswith('"'):
                    remark = remark[1:-1]
                return raw_resp, remark
            
            last_comma = rest.rfind(',')
            if last_comma != -1:
                last_part = rest[last_comma+1:].strip()
                if len(last_part) < 50:
                    return rest[:last_comma], last_part
                
            return rest, ""

        def clean_recommendations(recs_str):
            recs_str = recs_str.strip()
            if recs_str.startswith('"') and recs_str.endswith('"'):
                recs_str = recs_str[1:-1]
            
            recs_str = recs_str.replace('\\"\\"', '\\"')
            
            res = []
            i = 0
            n = len(recs_str)
            while i < n:
                c = recs_str[i]
                if c == '\\' and i + 2 < n and recs_str[i+1] == '"' and recs_str[i+2] == '"':
                    res.append('\\')
                    res.append('"')
                    i += 3
                elif c == '\\' and i + 1 < n and recs_str[i+1] == '"':
                    if i + 2 < n and recs_str[i+2] == '"':
                        res.append('\\')
                        res.append('"')
                        i += 3
                    else:
                        res.append('"')
                        i += 2
                elif c == '"' and i + 1 < n and recs_str[i+1] == '"':
                    res.append('"')
                    i += 2
                else:
                    res.append(c)
                    i += 1
                    
            recs_clean = "".join(res)
            recs_clean = re.sub(r'np\.int64\((\d+)\)', r'\1', recs_clean)
            return recs_clean

        cleaned_rows = []
        with open(csv_file, 'r', encoding='utf-8-sig') as f:
            f.readline()  # Skip header
            for line in f:
                line_str = line.strip()
                if not line_str:
                    continue
                try:
                    parts = line_str.split(',', 2)
                    timestamp = parts[0]
                    lottery = parts[1]
                    remaining = parts[2]
                    
                    m = re.match(r'^(.+?),(\d+),(\d+),(.*)$', remaining)
                    if not m:
                        continue
                    model = m.group(1).strip()
                    target_period = m.group(2)
                    input_periods = m.group(3)
                    rest_of_line = m.group(4)
                    
                    if model.startswith('"') and model.endswith('"'):
                        model = model[1:-1].replace('""', '"')
                        
                    recs_raw, final_rest = split_recommendations(rest_of_line)
                    if not recs_raw:
                        continue
                        
                    raw_response, remark = split_rest(final_rest)
                    
                    if raw_response.startswith('"') and raw_response.endswith('"'):
                        raw_response = raw_response[1:-1].replace('""', '"')
                        
                    recs_clean = clean_recommendations(recs_raw)
                    json_obj = json.loads(recs_clean)
                    
                    cleaned_rows.append({
                        "timestamp": timestamp,
                        "lottery": lottery,
                        "model": model,
                        "target_period": int(target_period),
                        "input_periods": int(input_periods),
                        "recommendations": json.dumps(json_obj, ensure_ascii=False),
                        "raw_response": raw_response,
                        "_备注_source": remark
                    })
                except Exception:
                    continue
        return pd.DataFrame(cleaned_rows)

def render_ai_analysis(df, config):
    st.subheader(f"📊 {config['name']} AI 预测历史对比")
    csv_file = "data/ai_predictions_history.csv"
    if not os.path.exists(csv_file):
        st.info("尚未发现 AI 预测记录。请先在 'AI 预测' 板块生成预测或运行批量脚本。")
        return
    try:
        df_hist = read_predictions_history_csv(csv_file)
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
