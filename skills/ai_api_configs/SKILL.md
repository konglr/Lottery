---
name: ai_api_configs
description: 本项目使用的 AI API 配置、技术指标规范及项目架构参考手册。
---

# AI API 配置与技术规范

本文件汇总了项目中使用的所有 AI API 配置、技术指标规范及项目架构。所有 Agent 在协助开发时应参考此文件中的规范。

---

## 1. 阿里云 DashScope API (Qwen)

* **API Key 环境变量**: `ALIYUNCS_API_KEY` (从 `.Renviron` 读取)
* **Base URLs**:
    * OpenAI 兼容协议：`https://coding.dashscope.aliyuncs.com/v1`
    * Anthropic 兼容协议：`https://coding.dashscope.aliyuncs.com/apps/anthropic`
* **已验证模型**:
    * **Qwen系列**: `qwen3.5-plus`, `qwen3-max-2026-01-23`, `qwen3-coder-next`, `qwen3-coder-plus`, `qwen-max`, `qwen-plus`
    * **第三方托管**: `MiniMax-M2.5`, `glm-5`, `glm-4.7`, `kimi-k2.5`, `glm-4`, `kimi-k2-5`
* **R 示例**:
    ```r
    apiKey <- Sys.getenv("ALIYUNCS_API_KEY")
    # 使用 httr2 调用 OpenAI 兼容接口
    ```

---
---

## 2. Gemini API (✅ 已迁移至 google-genai)

* **SDK 要求**: 必须使用最新的 `google-genai` package (不再支持 `google-generativeai`)。
* **API Key 环境变量**: `GEMINI_API_KEY` (从 `.Renviron` 读取)
* **已验证模型**:
    * **Gemini 3系列**: `models/gemini-3.1-pro-preview`, `models/gemini-3-pro-preview`, `models/gemini-3-flash-preview`
    * **Gemini 2.5系列**: `models/gemini-2.5-pro`, `models/gemini-2.5-flash`, `models/gemini-2.5-flash-lite`
    * **Gemini 2.0系列**: `models/gemini-2.0-flash`
* **Python 调用示例** (最新 Client 架构):
    ```python
    from google import genai
    import os
    
    # 手动加载 .Renviron (详见下文 4. 规范)
    api_key = os.getenv("GEMINI_API_KEY")
    client = genai.Client(api_key=api_key)
    
    response = client.models.generate_content(
        model='models/gemini-3-flash-preview',
        contents="Hello"
    )
    print(response.text)
    ```

---

## 3. MiniMax API (✅ 已验证)

* **状态**: ✅ 2026-02-26 测试通过
* **API Key 环境变量**: `MINIMAX_API_KEY` (从 `.Renviron` 读取)
* **Base URL**: `https://api.minimax.chat/v1`
* **兼容协议**: OpenAI 兼容 (`/v1/chat/completions`)
* **推荐模型**: `MiniMax-M2.5`, `MiniMax-M2.1`
* **Python 调用示例** (原生实现，无需 dotenv):
    ```python
    import os
    from openai import OpenAI

    # 手动加载 .Renviron
    def load_renviron(path=".Renviron"):
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        k, v = line.split('=', 1)
                        os.environ[k.strip()] = v.strip().strip('"\'')
    load_renviron()

    api_key = os.getenv("MINIMAX_API_KEY")
    client = OpenAI(api_key=api_key, base_url="https://api.minimax.chat/v1")
    response = client.chat.completions.create(model="MiniMax-M2.5", messages=[{"role": "user", "content": "Hello"}])
    ```

---

## 3. NVIDIA NIM API (新增)

* **API Key 环境变量**: `NV_API_KEY` (注意：不是 `NVIDIA_API_KEY`)
* **Base URL**: `https://integrate.api.nvidia.com/v1`
* **兼容协议**: OpenAI 兼容 (`/v1/chat/completions`)
* **支持模型**:
    * `z-ai/glm4.7` (GLM-4.7)
    * `minimaxai/minimax-m2.1` (MiniMax M2.1, NVIDIA 托管版)
    * `meta/llama-3.1-405b-instruct` (Llama 3.1)
* **Python 调用示例**:
    ```python
    import os
    from openai import OpenAI

    api_key = os.getenv("NV_API_KEY")
    client = OpenAI(api_key=api_key, base_url="https://integrate.api.nvidia.com/v1")
    
    # 测试 GLM-4.7
    response = client.chat.completions.create(
        model="z-ai/glm4.7", 
        messages=[{"role": "user", "content": "Hello"}]
    )
    ```

---

## 4. 环境变量加载规范

项目推荐使用 `.Renviron` 文件管理敏感信息。在 Python 环境中，应优先尝试从项目根目录通过 `load_renviron` 函数或 `os.getenv` 直接读取。
