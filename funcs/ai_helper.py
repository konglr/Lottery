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
            "models/gemini-2.0-flash"
        ],
        "NVIDIA": [
            "z-ai/glm4.7", 
            "meta/llama-3.1-70b-instruct",
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

def generate_ai_prediction(brand, model, api_key, history_text, config):
    """
    Unified entry point for AI predictions across different brands.
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
        prompt += "2. 结合分析结果，为下一期给出 5 组推荐的投注号码。\n"
        
    prompt += "3. 详细给出你选择这些号码的理由（如：考虑了遗漏值、和值范围、或是特定组合的重复性）。\n"
    prompt += "\n**输出格式：** 请使用清晰的 Markdown 格式输出，语言为中文。"

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
