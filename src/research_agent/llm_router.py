import os
import requests
import streamlit as st

# Ensure your config file is in the right folder, or change this to: from config import load_config
from utils.config import load_config 

load_config()

# @st.cache_data(show_spinner=False) # ⚠️ Keep commented out during testing to avoid caching hallucinations!
def analyze_text_with_fallback(prompt: str) -> str:
    # 1. ATTEMPT CLOUD (OpenRouter Waterfall)
    openrouter_key = os.getenv("OPENROUTER_API_KEY")
    if openrouter_key:
        
        # Ordered strictly by Finance capability, then by reasoning/context power
        model_list = [
            "arcee-ai/trinity-large-preview:free",     # 🏆 #1 Choice: Finance (#15) - 400B MoE
            "nvidia/nemotron-3-nano-30b-a3b:free",     # 🥈 #2 Choice: Finance (#50)
            "stepfun/step-3.5-flash:free",             # Fast MoE Reasoning Fallback
            "qwen/qwen3-vl-30b-a3b-thinking:free",     # Strong STEM/Math Thinking Fallback
            "liquid/lfm2.5-1.2b-thinking:free",        # Agentic Thinking Fallback
            "arcee-ai/trinity-mini:free",              # Efficient Mini MoE
            "nvidia/nemotron-nano-12b-2-vl:free",      # Fast Transformer-Mamba Hybrid
            "liquid/lfm2.5-1.2b-instruct:free"         # Final ultra-lightweight fallback
        ]
        
        for model in model_list:
            try:
                response = requests.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {openrouter_key}", 
                        "HTTP-Referer": "http://localhost:8501", 
                        "X-Title": "IntelliCredit"
                    },
                    json={
                        "model": model, 
                        "messages": [{"role": "user", "content": prompt}]
                    },
                    timeout=20 # Slightly higher timeout for thinking/reasoning models
                )
                if response.status_code == 200:
                    return response.json()['choices'][0]['message']['content']
                else:
                    print(f"⚠️ [ROUTER] {model} failed with status {response.status_code}. Trying next...")
            except Exception as e:
                print(f"⚠️ [ROUTER] {model} timed out or errored: {e}. Trying next...")
                continue

    # 2. FALLBACK TO LOCAL (Ollama Qwen 2.5)
    try:
        print("🚨 [ROUTER] All Cloud APIs failed or unreachable. Routing to local model...")
        local_response = requests.post(
            "http://localhost:11434/api/chat", # ✅ Fixed: Changed from /api/generate
            json={
                "model": "qwen2.5:7b", 
                "messages": [{"role": "user", "content": prompt}],
                "stream": False
            },
            timeout=60
        )
        if local_response.status_code == 200:
            return local_response.json()['message']['content']
    except Exception as e:
        return f"❌ All systems failed. Local engine unreachable. Error: {e}"
