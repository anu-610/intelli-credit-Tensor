import json
import re
import traceback
import os
import requests

# Import the multi-agent router which already handles the OpenRouter -> Local fallback
from src.research_agent.llm_router import analyze_text_with_fallback


def hunt_missing_financials(company_name: str, metric: str) -> float:
    """
    Sub-agent that dynamically searches the web if critical financial data is missing.
    Prevents the model from making up numbers when the PDF or CSV is incomplete.
    """
    if not company_name or company_name.strip() == "":
        return 0.0
        
    print(f"\n🕵️‍♂️ [DATA HUNTER AGENT] '{metric}' is missing! Launching targeted web search for {company_name}...")
    
    tavily_key = os.getenv("TAVILY_API_KEY")
    if not tavily_key:
        print("⚠️ [DATA HUNTER] Tavily API Key missing in .env. Cannot hunt for data.")
        return 0.0
        
    try:
        url = "https://api.tavily.com/search"
        data = {
            "api_key": tavily_key, 
            "query": f"What is the exact latest {metric} in INR for {company_name}? Provide the total numeric value.", 
            "search_depth": "basic", 
            "include_answer": True
        }
        response = requests.post(url, json=data)
        answer = response.json().get("answer", "")
        
        if not answer:
            print("⚠️ [DATA HUNTER] Web search returned no answer.")
            return 0.0
            
        print(f"🌐 [DATA HUNTER] Found web snippet: {answer[:150]}...")
        
        # Use LLM to extract and CONVERT the number from the search snippet
        prompt = f"""
        Extract the {metric} in INR from this text. Return ONLY a plain integer representing the absolute total amount in single Rupees. 
        
        STRICT CONVERSION RULES:
        - 'Crore' or 'Cr' -> multiply by 1,00,00,000 (e.g., 50 Cr = 500000000)
        - 'Lakh' or 'L' -> multiply by 1,00,000 (e.g., 10 Lakh = 1000000)
        - 'Billion' or 'B' -> multiply by 1,000,000,000
        - 'Trillion' or 'T' -> multiply by 1,000,000,000,000
        - 'Million' or 'M' -> multiply by 1,000,000
        
        Text: {answer}
        """
        val_str = analyze_text_with_fallback(prompt)
        
        # Strip all characters except digits to ensure a clean float conversion
        clean_val = re.sub(r'[^\d]', '', val_str.strip())
        extracted_number = float(clean_val) if clean_val else 0.0
        
        if extracted_number > 0:
            print(f"🎯 [DATA HUNTER] Successfully extracted and converted {metric}: ₹{extracted_number:,}")
        else:
            print(f"⚠️ [DATA HUNTER] Could not extract a valid number.")
            
        return extracted_number
        
    except Exception as e:
        print(f"⚠️ [DATA HUNTER] Search failed: {e}")
        return 0.0

def calculate_risk_score(qualitative_notes: str, extracted_insights: str = "", requested_amount: float = 0, company_name: str = "", ceo_name: str = "") -> dict:
    """
    Main Risk Appraisal Engine. Combines LLM reasoning with deterministic math overrides.
    Accepts ceo_name to enforce the Identity Isolation Protocol.
    """
    
    prompt = f"""
### SYSTEM ROLE: THE ANTIMATTER DATA-EXTRACTION & RISK SURVEILLANCE ENGINE
You are a cold, deterministic Risk Appraisal Engine. Your primary function is to convert raw data into a mathematically sound credit decision. You must distinguish between "Hard Data Mismatches" and "Reputational News."

#### PART I: THE AMNESIA & IDENTITY ISOLATION PROTOCOL
You possess zero external knowledge. Your world begins and ends with the text provided in the INPUT DATA block.
CRITICAL IDENTITY RULE: The borrowing entity is '{company_name}'. The Key Promoter/CEO is '{ceo_name if ceo_name else "Not Provided"}'. 
You MUST strictly isolate risks to this specific individual and their exact corporate entity. Disregard any news, bankruptcies, or penalties involving individuals or companies with similar names but different leadership (e.g., if appraising Mukesh Ambani's company, you MUST ignore news about Anil Ambani). Conflating separate legal entities is a critical analytical failure.

#### PART II: THE FATAL FRAUD & DATA-INTEGRITY KILL SWITCH
You MUST scan BOTH the '### Structured Data Synthesis' AND '### AI Forensic Multi-File Analysis' sections.
You are REQUIRED to trigger this kill switch if you see ANY ONE of the following triggers:
- TRIGGER A: The phrase "Major discrepancy detected!" is present.
- TRIGGER B: The text explicitly mentions "Circular Trading".
- TRIGGER C: The text explicitly mentions "Loan Shopping".
- TRIGGER D: The text explicitly mentions "Inflated expenses".
2. Mandatory Consequence: If ANY of these triggers are met, the Credit Score MUST be 0 to 3, and the Approved Limit MUST be 0%. This is a FATAL FRAUD FLAG. You are overriding the system to REJECT the loan.
3. Source Isolation: DO NOT trigger this Part II for web intelligence, general "allegations," or reputational news. It only applies to the structured CSV data sections.
#### PART III: THE RIGOROUS SKEPTICISM PROTOCOL
Assign a 4 (Caution) by default. A 9-10 requires explicit proof of excellence and no risks.
- Trend Analysis: If revenue grows but profit is flat, deduct 1 point for "Margin Compression."
- Macro Headwinds: If text mentions volatility or regulatory shifts, deduct 1 point.

#### PART IV: EXACT QUOTE VERIFICATION
Every risk factor must be accompanied by an exact, word-for-word quote from the EXTRACTED_INSIGHTS.

#### PART V: THE FIVE Cs OF CREDIT
Character, Capacity, Capital, Collateral, and Conditions. Evaluate based ONLY on the provided text and mention identity verification of {ceo_name}.

#### PART VI: NEWS, ALLEGATIONS & REPUTATIONAL NOISE (WEIGHTED)
1. Allegations matching the CEO '{ceo_name}': Deduct -1 to -2 points.
2. Regulatory Notices/Investigations matching the CEO: Deduct -2 to -3 points.
3. Proven Conviction/Asset Seizure matching the CEO: Deduct -5 points.
4. Logic: If the company has strong revenue and bank flows, historical news should NOT result in a score below 5.

#### PART VII: STRUCTURAL INTEGRITY
Respond ONLY with a valid raw JSON object. Use double braces for the JSON template.

### INPUT DATA
- BORROWING ENTITY: {company_name}
- KEY PROMOTER / CEO: {ceo_name if ceo_name else "Not Provided"}
- REQUESTED_LOAN_AMOUNT: {requested_amount} INR
- QUALITATIVE_NOTES: {qualitative_notes}
- EXTRACTED_INSIGHTS: {extracted_insights}

### REQUIRED JSON OUTPUT FORMAT
{{
    "verification_check_1_data_mismatch": "(Answer Yes/No: Does the Structured Data Synthesis section specifically state 'Major discrepancy detected!'?)",
    "verification_check_2_news_allegations": "(Summarize any reputational risks. Verify the names match the CEO '{ceo_name}' before including them.)",
    "credit_score": (integer 0-10),
    "approved_limit_percentage": (integer 0-100),
    "estimated_ebitda_inr": (integer, MANDATORY: Convert 'Crore' to 10000000 and 'Lakh' to 100000. E.g. '25,000 Crore' -> 250000000000. Output 0 if missing.),
    "estimated_net_worth_inr": (integer, MANDATORY: Convert 'Crore' to 10000000, 'Lakh' to 100000, and 'Trillion' to 1000000000000. Output 0 if missing.),
    "recommended_interest_rate": (float),
    "decision_logic": "(Explain based ONLY on your verification checks. Distinguish between hard data mismatches and news allegations. Confirm you isolated results to {ceo_name}.)"
}}
    """

    print("\n" + "="*60)
    print("🚀 [DEBUG] STARTING ML SCORER")

    try:
        raw_output = analyze_text_with_fallback(prompt)
        
        # Clean the output to ensure we only parse the JSON block
        cleaned_output = re.sub(r'```(?:json)?', '', raw_output).strip()
        json_match = re.search(r'\{.*\}', cleaned_output, re.DOTALL)
        
        if json_match:
            decision = json.loads(json_match.group(0))
            
            # Fix decimal interest rate bug
            if decision.get("recommended_interest_rate", 0) < 1.0:
                decision["recommended_interest_rate"] *= 100
            
            # Extract AI findings
            percentage = decision.get("approved_limit_percentage", 10)
            base_ai_limit = int(requested_amount * (percentage / 100.0))
            
            ebitda = float(decision.get("estimated_ebitda_inr", 0))
            net_worth = float(decision.get("estimated_net_worth_inr", 0))
            
            # -------------------------------------------------------------------
            # THE AUTONOMOUS AGENTIC FALLBACK
            # -------------------------------------------------------------------
            if ebitda == 0:
                ebitda = hunt_missing_financials(company_name, "EBITDA")
                decision["estimated_ebitda_inr"] = ebitda
                
            if net_worth == 0:
                net_worth = hunt_missing_financials(company_name, "Net Worth")
                decision["estimated_net_worth_inr"] = net_worth
            
            # -------------------------------------------------------------------
            # PYTHON SECURE MATH EXECUTION & FULL TRANSPARENCY UI
            # -------------------------------------------------------------------
            final_limit = base_ai_limit
            
            req_str = f"₹{int(requested_amount):,}"
            base_limit_str = f"₹{int(base_ai_limit):,}"
            ebitda_str = f"₹{int(ebitda):,}" if ebitda > 0 else "N/A"
            ebitda_cap_str = f"₹{int(ebitda * 3):,}" if ebitda > 0 else "N/A"
            nw_str = f"₹{int(net_worth):,}" if net_worth > 0 else "N/A"
            nw_cap_str = f"₹{int(net_worth * 0.6):,}" if net_worth > 0 else "N/A"
            
            ai_logic = decision.get("decision_logic", "")
            
            if ebitda > 0 or net_worth > 0:
                caps = []
                if ebitda > 0: caps.append(ebitda * 3)
                if net_worth > 0: caps.append(net_worth * 0.6)
                
                financial_cap = min(caps)
                
                if base_ai_limit > financial_cap:
                    final_limit = int(financial_cap)
                    final_limit_str = f"₹{int(final_limit):,}"
                    
                    math_audit_html = (
                        f"<br><br><span style='color:#ff4d4f'><b>⚠️ STRICT SYSTEM OVERRIDE (AI LIMIT REJECTED):</b></span><br>"
                        f"&nbsp;&nbsp;[1] AI Proposed Approval: {percentage}% of {req_str} = <b>{base_limit_str}</b><br>"
                        f"&nbsp;&nbsp;[2] EBITDA Bank Cap (3x): 3 * {ebitda_str} = <b>{ebitda_cap_str}</b><br>"
                        f"&nbsp;&nbsp;[3] Net Worth Bank Cap (60%): 0.6 * {nw_str} = <b>{nw_cap_str}</b><br>"
                        f"&nbsp;&nbsp;[4] <b>Final Limit Enforced</b> (Lowest of above) = <span style='color:#ff4d4f'><b>{final_limit_str}</b></span>"
                    )
                else:
                    final_limit_str = f"₹{int(final_limit):,}"
                    math_audit_html = (
                        f"<br><br><span style='color:#00d084'><b>✅ SYSTEM MATH VERIFICATION (PASSED):</b></span><br>"
                        f"&nbsp;&nbsp;[1] AI Proposed Approval: {percentage}% of {req_str} = <b>{base_limit_str}</b><br>"
                        f"&nbsp;&nbsp;[2] EBITDA Bank Cap (3x): 3 * {ebitda_str} = <b>{ebitda_cap_str}</b><br>"
                        f"&nbsp;&nbsp;[3] Net Worth Bank Cap (60%): 0.6 * {nw_str} = <b>{nw_cap_str}</b><br>"
                        f"&nbsp;&nbsp;[4] AI Limit is within safe regulatory caps. <b>Approved: {final_limit_str}</b>"
                    )
            else:
                final_limit_str = f"₹{int(final_limit):,}"
                math_audit_html = (
                        f"<br><br><span style='color:#f5a623'><b>⚠️ SYSTEM MATH VERIFICATION (WARNING):</b></span><br>"
                        f"&nbsp;&nbsp;Could not locate exact EBITDA or Net Worth data to enforce hard caps.<br>"
                        f"&nbsp;&nbsp;Proceeding with raw AI recommendation of <b>{final_limit_str}</b>."
                    )
            
            decision["decision_logic"] = f"<b>🤖 AI REASONING:</b> {ai_logic} {math_audit_html}"
            decision["recommended_limit_inr"] = int(final_limit)
            decision["engine_source"] = f"🧠 OpenRouter/Local Hybrid"
            
            return decision
        
        raise ValueError("Could not find a valid JSON object in the LLM response.")

    except Exception as e:
        print(f"❌ [DEBUG] ALL LLMs FAILED! Error: {e}")
        return {
            "credit_score": 3,
            "recommended_limit_inr": int(requested_amount * 0.1) if requested_amount else 5000000,
            "recommended_interest_rate": 15.0,
            "decision_logic": f"⚙️ System Fallback Triggered. AI Error: {str(e)[:80]}...",
            "engine_source": f"⚙️ Quantitative Fallback"
        }
