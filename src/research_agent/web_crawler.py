import os
import requests

def execute_tavily_search(query: str, api_key: str) -> dict:
    """Helper function to execute a single deep search."""
    url = "https://api.tavily.com/search"
    headers = {"Content-Type": "application/json"}
    
    data = {
        "api_key": api_key,
        "query": query,
        "search_depth": "advanced", 
        "include_answer": True, 
        "max_results": 5            
    }
    
    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        return {"error": str(e)}

def crawl_company_news(company_name: str, ceo_name: str = "") -> str:
    """
    Performs a thorough, multi-faceted deep web crawl for a corporate entity.
    Injects the CEO name to prevent hallucination and name-collisions.
    """
    if not company_name:
        return "No company name provided for research."

    tavily_key = os.getenv("TAVILY_API_KEY")
    if not tavily_key:
        return "⚠️ Web Research Skipped: TAVILY_API_KEY not found in .env file."

    # INJECT CEO NAME to force Tavily to find the exact specific company
    ceo_context = f" (led by {ceo_name})" if ceo_name and ceo_name.strip() else ""
    target_entity = f"{company_name}{ceo_context}"

    queries = [
        ("📊 Financial Statistics & Market Intelligence", 
         f"Provide EXACT NUMBERS in INR and percentages for {target_entity}. I need the specific revenue, net profit, profit margin, and net worth figures for the last 5 historical years (e.g., FY21, FY22, FY23, FY24). Return quantitative data only."),
         
        ("⚖️ Legal & Regulatory Risks", 
         f"{target_entity} NCLT insolvency litigation, RBI SEBI penalties, major fraud, or default court disputes"),
         
        ("👤 Corporate Governance & Promoters", 
         f"{target_entity} promoters track record, management changes, corporate governance issues, whistleblower"),
         
        ("📰 Recent News & Controversies", 
         f"Latest negative news, controversies, scams, financial distress, or red flags regarding {target_entity}"),
         
        ("🌍 Sector & Macro-Economic Outlook", 
         f"What is the current economic outlook, major risks, headwinds, and regulatory challenges for the primary industry/sector that {target_entity} operates in?")
    ]

    report = f"### Deep Web Intelligence Report: {company_name}{ceo_context}\n\n"
    seen_urls = set()
    unique_sources = []

    for section_title, query in queries:
        result = execute_tavily_search(query, tavily_key)
        
        if "error" in result:
            report += f"**{section_title}:**\n⚠️ Error fetching data: {result['error']}\n\n"
            continue
            
        answer = result.get("answer", "No AI summary generated for this topic.")
        
        if "Controversies" in section_title:
            report += f"**{section_title}:**\n🚨 CRITICAL NEGATIVE NEWS REVIEW: {answer}\n\n"
        else:
            report += f"**{section_title}:**\n{answer}\n\n"
        
        for r in result.get("results", []):
            url = r.get("url")
            if url and url not in seen_urls:
                seen_urls.add(url)
                unique_sources.append((r.get("title", "Source"), url))
    
    report += "**Top Verified Sources:**\n"
    for title, url in unique_sources:
        report += f"- [{title}]({url})\n"

    return report
