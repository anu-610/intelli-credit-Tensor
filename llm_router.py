import os
import requests
import time
import streamlit as st
from utils.config import load_config

# Ensure keys are loaded
load_config()

@st.cache_data(show_spinner=False)
def analyze_text_with_fallback(prompt: str) -> str:
    gemini_key = os.getenv("GEMINI_API_KEY")
    openrouter_key = os.getenv("OPENROUTER_API_KEY")
    
    time.sleep(1.5) # Anti-spam delay to protect your API limits

    # --- 1. Primary Attempt: Google Gemini 2.0 Flash ---
    try:
        if not gemini_key:
            raise ValueError("No Gemini key found")
            
        gemini_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={gemini_key}"
        headers = {'Content-Type': 'application/json'}
        data = {"contents": [{"parts": [{"text": prompt}]}]}
        
        response = requests.post(gemini_url, headers=headers, json=data)
        response.raise_for_status()
        return response.json()['candidates'][0]['content']['parts'][0]['text']

    except Exception as gemini_error:
        print(f"⚠️ Gemini hit a limit. Trying OpenRouter Waterfall...")
        
        # --- 2. Fallback Attempt: OpenRouter Waterfall ---
        try:
            if not openrouter_key:
                raise ValueError("No OpenRouter key found")
                
            openrouter_url = "https://openrouter.ai/api/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {openrouter_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "http://localhost:8501", # Required by OpenRouter
                "X-Title": "IntelliCredit"              # Required by OpenRouter
            }
            
            # The best free models available right now
            free_models = [
                "arcee-ai/trinity-large-preview:free", 
                "stepfun/step-3.5-flash:free",         
                "liquid/lfm-2.5-1.2b-thinking:free"   
            ]
            
            last_error = ""
            
            for model_id in free_models:
                data = {
                    "model": model_id,
                    "messages": [{"role": "user", "content": prompt}]
                }
                response = requests.post(openrouter_url, headers=headers, json=data)
                
                if response.status_code == 200:
                    print(f"✅ Success! OpenRouter model {model_id} generated the response.")
                    return response.json()['choices'][0]['message']['content']
                else:
                    last_error = f"{response.status_code} - {response.text}"
                    print(f"⚠️ OpenRouter model {model_id} failed. Trying next...")
            
            raise ValueError(f"All OpenRouter models failed. Last error: {last_error}")
            
        except Exception as openrouter_error:
            print(f"⚠️ OpenRouter completely failed. Waking up LOCAL AI...")
            
            # --- 3. Ultimate Fallback: Local Ollama (using your gemma3:4b model) ---
            try:
                local_url = "http://localhost:11434/api/generate"
                local_data = {
                    "model": "gemma3:4b", # Matched to the model in your screenshot!
                    "prompt": prompt,
                    "stream": False
                }
                
                local_response = requests.post(local_url, json=local_data)
                local_response.raise_for_status()
                print("✅ Success! Local Gemma3 model generated the response.")
                return local_response.json()['response']
                
            except Exception as local_error:
                return f"❌ Error: All Cloud APIs failed AND Local AI failed. Is Ollama running? Details: {local_error}"
