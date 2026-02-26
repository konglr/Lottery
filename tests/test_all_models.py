import os
import sys
import time
from datetime import datetime

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from funcs.ai_helper import load_renviron, get_brand_models, simple_chat

def test_all_models_connectivity():
    # Load environment variables
    load_renviron()
    
    brand_models = get_brand_models()
    
    api_key_env_map = {
        "Gemini": "GEMINI_API_KEY",
        "NVIDIA": "NV_API_KEY",
        "MiniMax": "MINIMAX_API_KEY",
        "DashScope": "ALIYUNCS_API_KEY"
    }

    print("\n" + "="*100)
    print(f"{'Brand':<12} | {'Model':<40} | {'Status':<8} | {'Latency':<8} | {'Response Preview'}")
    print("-" * 100)

    for brand, models in brand_models.items():
        api_key = os.getenv(api_key_env_map.get(brand))
        
        if not api_key:
            for model in models:
                print(f"{brand:<12} | {model:<40} | {'SKIP':<8} | {'N/A':<8} | No API Key found in .Renviron")
            continue

        for model in models:
            start_time = time.time()
            # Simple Hello test
            response = simple_chat(brand, model, api_key, "Hello, please reply with 'Hello' and your model name.")
            latency = time.time() - start_time
            
            if response.startswith("ERROR:"):
                status = "❌ FAIL"
                preview = response[:60].replace('\n', ' ') + "..."
            else:
                status = "✅ OK"
                preview = response[:60].replace('\n', ' ') + "..."
            
            print(f"{brand:<12} | {model:<40} | {status:<8} | {latency:>7.2f}s | {preview}")
            
            # Small sleep to avoid rate limits
            time.sleep(0.5)

    print("="*100)
    print(f"\nConnectivity Test Finished at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    test_all_models_connectivity()
