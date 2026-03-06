import os
import pandas as pd

def load_renviron(path=".Renviron"):
    """
    Load environment variables from .Renviron file.
    """
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    k, v = line.split('=', 1)
                    os.environ[k.strip()] = v.strip().strip('"\'')

def get_brand_models():
    """
    Returns the dictionary of supported brands and their models.
    """
    return {
        "Gemini": [
            "models/gemini-3.1-pro-preview", 
            "models/gemini-3-pro-preview", 
            "models/gemini-3-flash-preview", 
            "models/gemini-2.5-pro", 
            "models/gemini-2.5-flash", 
            "models/gemini-2.5-flash-lite",
            "models/gemini-3.1-flash-lite-preview"
        ],
        "NVIDIA": [
            "z-ai/glm4.7", 
            "meta/llama-3.1-70b-instruct",
            "meta/llama-3.1-405b-instruct",
            "deepseek-ai/deepseek-r1",
            "deepseek-ai/deepseek-v3",
            "microsoft/phi-3-medium-128k-instruct"
        ],
        "MiniMax": ["MiniMax-M2.5", "MiniMax-M2.1"],
        "DashScope": [
            "qwen3.5-plus", "qwen3-max-2026-01-23", "qwen3-coder-next", "qwen3-coder-plus",
            "MiniMax-M2.5", "glm-5", "glm-4.7", "kimi-k2.5"
        ]
    }

def prepare_lottery_data_text(df, config):
    """
    Converts lottery history dataframe to a text string format for AI prompt.
    Only includes draw number, red balls, and blue balls.
    """
    if df.empty:
        return ""
    
    data_str = ""
    for _, r in df.iterrows():
        # Get Red Balls
        red_cols = [f"{config['red_col_prefix']}{i}" for i in range(1, config['red_count']+1)]
        reds = [int(r[col]) for col in red_cols if col in r and pd.notna(r[col])]
        
        # Get Blue Balls
        blues = []
        if config['has_blue']:
            if config['blue_count'] == 1:
                bc = config['blue_col_name'] if config['blue_col_name'] in r else '蓝球'
                if bc in r and pd.notna(r[bc]):
                    blues.append(int(r[bc]))
                elif '篮球' in r and pd.notna(r['篮球']):
                    blues.append(int(r['篮球']))
            else:
                base = config['blue_col_name']
                for i in range(1, config['blue_count'] + 1):
                    if f"{base}{i}" in r and pd.notna(r[f"{base}{i}"]): 
                        blues.append(int(r[f"{base}{i}"]))
                    elif f"篮球{i}" in r and pd.notna(r[f"篮球{i}"]): 
                        blues.append(int(r[f"篮球{i}"]))
        
        data_str += f"期号: {r['期号']}, 红球: {reds}" + (f", 蓝球: {blues}" if blues else "") + "\n"
    return data_str

def get_prediction_prompt(history_text, config):
    """
    Constructs the AI prompt based on history data and lottery configuration.
    """
    is_kl8 = config['code'] == 'kl8'
    
    # Construct Prompt
    prompt = f"你是一位专业的彩票数据分析专家。请根据以下最新的 {len(history_text.strip().splitlines())} 期 {config['name']} 开奖历史数据进行深度分析：\n\n"
    prompt += history_text
    prompt += "\n**要求：**\n"
    prompt += "1. 简要分析近期号码的冷热趋势、奇偶比例以及是否有明显的连号或跳号规律。\n"
    
    if is_kl8:
        prompt += "2. 结合分析结果，为下一期给出 20 个推荐的投注号码。\n"
    else:
        prompt += "2. 结合分析结果，为下一期给出 5 组推荐的投注号码。(考虑组号形态学的概率分布（如： 每组号码的连号（号码相连如12-13.25-26-27），跳号（号码之间相隔一个号码，如11-13/30-32—），重号（是与最近一期的开奖号码相同的数字），形态学的分布)\n"
        
    prompt += "3. 给出你的理由：详细说明选择这些号码的依据（如遗漏值、和值、跨度形态等）。\n"
    prompt += f"4. **胆码推荐**：请从你选出的{ '号码' if is_kl8 else '红球' }中，选出 1-3 个最看好的作为“胆码”。\n"
    
    prompt += "\n**输出格式要求 (必须严格遵守，以便程序解析)**：\n"
    if is_kl8:
        prompt += "【预测结果开始】\n"
        prompt += "- 胆码: [01, 02, 03, 04, 05]\n"
        prompt += "- 推荐号码: [01, 02, ..., 20]\n"
        prompt += "【预测结果结束】\n"
    else:
        red_example = ", ".join([f"{i:02d}" for i in range(1, config['red_count'] + 1)])
        blue_example = ", ".join([f"{i:02d}" for i in range(1, config['blue_count'] + 1)])
        prompt += "【预测结果开始】\n"
        prompt += "- 胆码: [01, 02]\n"
        prompt += f"- 第1组: [红球: {red_example}]" + (f" + [蓝球: {blue_example}]" if config['has_blue'] else "") + " | 分析: [理由内容]\n"
        prompt += f"- 第2组: [红球: {red_example}]" + (f" + [蓝球: {blue_example}]" if config['has_blue'] else "") + " | 分析: [理由内容]\n"
        prompt += "... (以此类推完成 5 组)\n"
        if config['code'] == 'qlc':
            prompt += "注：七乐彩的基本号和特别号共用 1-30 的号码池，且互不重复。这意味着你实际上是在 30 个号码中选取 8 个不重复的号码（7个红球+1个蓝球）。请在分析时将这 8 个位置作为一个整体进行考量，以提高全局命中概率。\n"
        prompt += "【预测结果结束】\n"
    prompt += "\n总结请严格遵守：先输出详细的分析/思考过程，然后输出关键词：分析结果，最后再给出【预测结果开始】和【预测结果结束】之间的固定格式内容。语言为中文。"
    return prompt

def generate_ai_prediction(brand, model, api_key, history_text, config):
    """
    Unified entry point for AI predictions across different brands.
    """
    prompt = get_prediction_prompt(history_text, config)

    try:
        if brand == "Gemini":
            from google import genai
            client = genai.Client(api_key=api_key)
            response = client.models.generate_content(
                model=model,
                contents=prompt
            )
            return response.text
            
        else:
            from openai import OpenAI
            # OpenAI compatible APIs
            base_url = ""
            if brand == "NVIDIA":
                base_url = "https://integrate.api.nvidia.com/v1"
            elif brand == "MiniMax":
                base_url = "https://api.minimax.chat/v1"
            elif brand == "DashScope":
                base_url = "https://coding.dashscope.aliyuncs.com/v1"
            
            client = OpenAI(api_key=api_key, base_url=base_url)
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.choices[0].message.content

    except Exception as e:
        raise Exception(f"AI 接口请求失败 ({brand}/{model}): {str(e)}")

def simple_chat(brand, model, api_key, message):
    """
    Generic chat function for testing API connectivity.
    """
    try:
        if brand == "Gemini":
            from google import genai
            client = genai.Client(api_key=api_key)
            response = client.models.generate_content(
                model=model,
                contents=message
            )
            return response.text
        else:
            from openai import OpenAI
            base_url = ""
            if brand == "NVIDIA":
                base_url = "https://integrate.api.nvidia.com/v1"
            elif brand == "MiniMax":
                base_url = "https://api.minimax.chat/v1"
            elif brand == "DashScope":
                base_url = "https://coding.dashscope.aliyuncs.com/v1"
            
            client = OpenAI(api_key=api_key, base_url=base_url)
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": message}]
            )
            return response.choices[0].message.content
    except Exception as e:
        return f"ERROR: {str(e)}"

import re
import json

def parse_ai_recommendations(text, config):
    """
    Parse the AI's Markdown response to extract structured recommendation data.
    """
    is_kl8 = config['code'] == 'kl8'
    results = {
        "dan": [], # 胆码
        "groups": [], # List of {reds: [], blues: [], reason: ""}
        "kl8_numbers": [] # Only for KL8
    }

    # Step 1: Strip thinking blocks to avoid catching placeholders in the "plan"
    clean_text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE).strip()

    # Step 2: Extract the summary block if present
    summary_match = re.search(r"【预测结果开始】(.*?)【预测结果结束】", clean_text, re.DOTALL)
    source_text = summary_match.group(1) if summary_match else clean_text

    # Extract Dan (胆码) - Flexible regex for different formats
    # Support: 胆码: [1, 2, 3] or 胆码: 1, 2, 3 or 胆码：1、2、3
    dan_match = re.search(r"胆码\s*[:：]\s*\[?([\d\s,，、\|-]+)\]?", source_text)
    if dan_match:
        results["dan"] = [int(n) for n in re.findall(r"\d+", dan_match.group(1))]

    if is_kl8:
        # Match 推荐号码: [01, 02, ..., 20]
        match = re.search(r"推荐号码\s*[:：]\s*\[?([\d\s,，、\|-]+)\]?", source_text)
        if match:
            nums = re.findall(r"\d+", match.group(1))
            results["kl8_numbers"] = [int(n) for n in nums]
    else:
        # Match 第X组: [红球: ...] + [蓝球: ...] | 分析: [...]
        group_lines = re.findall(r"第\s*[一二三四五12345]\s*组\s*[:：].*?(?=\n|$)", source_text)
        
        for line in group_lines:
            reds = []
            blues = []
            reason = ""
            
            # Extract Reds - support both "[红球: ...]" and "红球: [...]"
            red_match = re.search(r"(?:红球|号码)\s*[:：]\s*\[?([\d\s,，、\|-]+)\]?", line)
            if red_match:
                reds = [int(n) for n in re.findall(r"\d+", red_match.group(1))]
            
            # Extract Blues
            blue_match = re.search(r"蓝球\s*[:：]\s*\[?([\d\s,，、\|-]+)\]?", line)
            if blue_match:
                blues = [int(n) for n in re.findall(r"\d+", blue_match.group(1))]
                
            # Extract Reason - everything after '分析:' or '分析：'
            reason_match = re.search(r"分析\s*[:：]\s*(.*?)$", line)
            if reason_match:
                reason = reason_match.group(1).strip("[] \t|")
            
            if reds:
                results["groups"].append({
                    "reds": reds,
                    "blues": blues,
                    "reason": reason
                })
    
    return results

def format_ai_response(text):
    """
    Detects <think> blocks in the AI response and returns a tuple (thinking, result).
    """
    thinking = ""
    result = text
    
    # Match <think>...</think> block (case-insensitive, dotall)
    think_match = re.search(r"<think>(.*?)</think>", text, re.DOTALL | re.IGNORECASE)
    if think_match:
        thinking = think_match.group(1).strip()
        # Remove the think block from the final result
        result = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE).strip()
    
    return thinking, result
