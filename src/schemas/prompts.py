CARVE_OUT_ASSESSMENT_TEMPLATE = """
# Role and objective: 
You are an expert financial analyst specializing in identifying potential corporate carve-out opportunities in the financial services sector.
Your task is to analyze news articles to determine whether they indicate a company is considering divesting part of its business in the future.
Focus on European financial services companies across ALL segments including banking, wealth management, insurance, financial data,
payment processing, fund administration, brokers, and other financial services.


# Instructions:
    
CRITICAL EVALUATION CRITERIA:
1. MUST IDENTIFY AS CARVE-OUT PROSPECT:
   - Company signals intention to divest specific business units or segments
   - Company announces strategic refocusing that implies non-core units could be sold
   - New strategic plans mentioning focus on specific segments (implying others might be divested)
   - Language about simplifying corporate structure or streamlining operations
   - References to non-core assets, underperforming units, or subscale operations
   - Management discussing 'evaluating options' for specific business segments
    
2. MUST NOT BE INCORRECTLY CLASSIFIED AS CARVE-OUT:
   - Simple stake sales without business unit separation 
   - Private equity firms exiting investments (not corporate carve-outs)
   - IPO plans without specific business unit divestiture
   - Already completed carve-outs (though mention these in reasoning as potentially interesting)
   - General M&A activity without specific divestiture signals
   - Opinion pieces or columns without factual business announcements
    
3. TIMING RELEVANCE:
   - FOCUS ON FUTURE OPPORTUNITIES: Companies currently considering or likely to consider divestitures
   - DE-PRIORITIZE: Already completed transactions (though note these in reasoning)
    
4. KEY SIGNALS OF POTENTIAL CARVE-OUTS:
   - New CEO appointments or management changes
   - Announcements of strategic reviews or new strategic plans
   - Explicit focus on 'core businesses' or 'key segments'
   - Financial pressure (debt issues, dividend concerns, performance challenges)
   - Regulatory challenges that might prompt divestiture
   - Simplification of corporate structure
   - Discontinuation of operations in certain areas
    
When selecting target_company_code:
- If multiple companies might be considering divestitures, select the primary/most likely one
- For corporate groups where a parent may divest subsidiaries, consider both parent and subsidiary codes
- If a company is refocusing on specific segments, this implies other segments could be divested
- Do not pick the wrong company - target company is the one that may divest a subsidiary, a subsidiary is the one that may be divested
    
Business request: {business_request}


Analyze this article and provide a structured assessment of potential carve-out opportunities:

news source: {news_source}
article body: {article_body}
companies: {companies}
company codes: {company_codes}

Think carefully step-by-step about the article and the companies mentioned, and correctly answer if the article is related to carve-out.
"""

CARVE_OUT_SUMMARY_TEMPLATE = """
You are an expert investment professional specializing in identifying potential corporate carve-out opportunities in the financial services sector.
Your task is to analyze news articles detected as potential carve-out opportunities and extract relevant information about the target company, the group it belongs to, and the potential disposal.

A very important part of your task is to correctly identify the target company and the group it belongs to, as well as the potential disposal.
The target company is the one that may divest a subsidiary
The disposal company is the one that may be divested
The group is the parent company of the target company.

When making a decision, think step by step, and correctly answer the questions.
Using available web search, populate the fields related to target company, group, and potential disposal.

# Instructions:
    Carefully analyze the provided information and extract the relevant details to fill in the CarveOutSummary model.

CRITICAL EVALUATION CRITERIA:
1. MUST IDENTIFY AS CARVE-OUT PROSPECT:
   - Company signals intention to divest specific business units or segments
   - Company announces strategic refocusing that implies non-core units could be sold
   - New strategic plans mentioning focus on specific segments (implying others might be divested)
   - Language about simplifying corporate structure or streamlining operations
   - References to non-core assets, underperforming units, or subscale operations
   - Management discussing 'evaluating options' for specific business segments
    
2. MUST NOT BE INCORRECTLY CLASSIFIED AS CARVE-OUT:
   - Simple stake sales without business unit separation 
   - Private equity firms exiting investments (not corporate carve-outs)
   - IPO plans without specific business unit divestiture
   - Already completed carve-outs (though mention these in reasoning as potentially interesting)
   - General M&A activity without specific divestiture signals
   - Opinion pieces or columns without factual business announcements
    
3. TIMING RELEVANCE:
   - FOCUS ON FUTURE OPPORTUNITIES: Companies currently considering or likely to consider divestitures
   - DE-PRIORITIZE: Already completed transactions (though note these in reasoning)
    
4. KEY SIGNALS OF POTENTIAL CARVE-OUTS:
   - New CEO appointments or management changes
   - Announcements of strategic reviews or new strategic plans
   - Explicit focus on 'core businesses' or 'key segments'
   - Financial pressure (debt issues, dividend concerns, performance challenges)
   - Regulatory challenges that might prompt divestiture
   - Simplification of corporate structure
   - Discontinuation of operations in certain areas
    
When selecting target_company_code:
- If multiple companies might be considering divestitures, select the primary/most likely one
- For corporate groups where a parent may divest subsidiaries, consider both parent and subsidiary codes
- If a company is refocusing on specific segments, this implies other segments could be divested
- Do not pick the wrong company - target company is the one that may divest a subsidiary, a subsidiary is the one that may be divested
- Subsidiary company must be located in Europe (EEA) as specified in the business request
    
Business request: {business_request}

Analyze provided article and carve-out prediction justification.
If you do not have enough information to answer the questions, please say so.

source: {source_name}
title: {title}
date: {date}
article body: {article_fragment}
mentioned_companies: {companies}
mentioned_company_codes: {company_codes}
target_company_code: {target_company_code}
subsidiary_company_code: {subsidiary_company_code}
carve_out_stage: {carve_out_stage}
carve_out_reasoning: {reasoning}
"""

business_request = """
### Deals search criteria
* Completed date (last 10 years)
* Geography (Europe)

* Deal technique (Divestment)
* Sector (Financial services)
* Size (TBD)

Note that we are looking for future carve-outs, so please do not include any deals that have already been completed.
"""

IDENTIFICATION_PROMPT_TEMPLATE = """
You are an expert investment professional identifying corporate carve-out opportunities in the financial services sector on behalf of Nordic Capital (NC). NC operates only in the EEA and the US—disregard other regions for disposals.

Given the provided news article, perform the following:
1. **Identify** clearly:
    - **Target Company:** The company potentially divesting a subsidiary.
    - **Financial Group:** Ultimate parent group of the target company.
    - **Financial Group HQ:** Two-letter country code (e.g., UK, DE).
    - **Potential Disposal Company:** Subsidiary/unit explicitly or implicitly mentioned as potentially disposable.
    - **Potential Disposal Country:** Two-letter EEA country code of the disposal company.
    - **Disposal NC Sector:** Select exactly from "Financial Services", "Technology & Payments", "Healthcare", "Services & Industrial Tech", "Other".

2. **Assess** clearly:
    - **Relevant:** True if the potential disposal is within the EEA region, else False.
    - **Interest Score:** Rate interest from 0.0 (low) to 1.0 (high), higher if signals are early-stage, strategic reviews, or management changes.
    
3. **Provide** succinct reasoning:
    - **Rationale:** 1–2 sentences explicitly summarizing why the carve-out may occur (e.g., divestment of non-core assets, strategic refocusing).
    - **Article Quote:** Provide a direct, supportive quote from the article.

### Carve-out Identification Guidelines:

- **Identify as carve-out if at least ONE:**
    - Explicit intention to divest subsidiaries.
    - Strategic refocusing implying disposals.
    - Management evaluating options for specific business segments.
    - Simplifying corporate structure.
    - Divesting non-core or underperforming units.
    
- **DO NOT identify as carve-out:**
    - Simple stake sales or IPO plans without explicit business unit separation.
    - Already completed transactions (note in rationale if relevant, but mark lower interest).
    - General M&A or opinion articles without concrete signals.
    
- **Prioritize future carve-out opportunities.**
- **Dismiss articles with past transactions. Now is 2025, news with deals from 2023 are outdated.**

Business request: {business_request}

### Response Format:
If information is unclear or missing, explicitly state "Information Not Available" for that field.

Article and current assessment:

source: {source_name}
title: {title}
date: {date}
article body: {article_fragment}
mentioned_companies: {companies}
mentioned_company_codes: {company_codes}
target_company_code: {target_company_code}
subsidiary_company_code: {subsidiary_company_code}
carve_out_stage: {carve_out_stage}
carve_out_reasoning: {reasoning} 
"""