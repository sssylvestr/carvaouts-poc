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
    - **Potential Disposal:** Potential subdivision to be disposed of (e.g., UK business, Insurance Arm, IP business, etc.) - specific parts of the business being considered for divestiture.
    - **Potential Disposal Company:** Name of the specific subsidiary or business unit considered for disposal. **CRITICAL: Output "Unknown" if Disposal company is not OBVIOUS from the text of the article.**

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

Business request: {business_request}

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


IDENTIFICATION_SEARCH_PROMPT_TEMPLATE = """
You are an expert investment researcher assisting Nordic Capital (NC) in validating and enriching information about potential carve-out opportunities.

BACKGROUND CONTEXT:
The information below was extracted from a news article that mentions a company expressing intention or considering the disposal of a business unit/subsidiary/asset. This represents a potential carve-out opportunity where the target company may divest the mentioned asset or subsidiary.

The article discusses {target_company} (part of the {financial_group} financial group) potentially disposing of {potential_disposal_company}, which operates in the {potential_disposal} sector/business area.

INFORMATION SOURCES AVAILABLE TO YOU:
1. **WEB SEARCH**: Use web search to find authoritative information about companies, their headquarters, industry classifications, and business activities
2. **ARTICLE CONTENT**: Analyze the provided article text for explicit mentions of locations, business types, company descriptions, and operational details
3. **COMPANY CONTEXT**: Use the provided company names and business descriptions as starting points for your research

Your task is to combine information from ALL available sources to provide the most accurate and complete picture.

Given context from article analysis:
- Target Company: {target_company} (the company mentioned as potentially divesting)
- Financial Group: {financial_group} (the parent/ultimate owner of the target company)
- Potential Disposal: {potential_disposal} (the business/asset being considered for disposal)
- Potential Disposal Company: {potential_disposal_company} (the specific entity/subsidiary being divested)

Article source: {news_source}
Article content: {article_body}

RESEARCH METHODOLOGY:
1. **Start with the article**: Look for explicit mentions of headquarters, locations, business types, and industry classifications
2. **Supplement with web search**: Use web search to verify article information and fill in missing details about:
   - Company headquarters and locations
   - Parent company industry classifications
   - Business unit industry categories
   - Subsidiary operational focus areas
3. **Cross-validate**: Ensure information from web search is consistent with article context
4. **Prioritize authoritative sources**: Use official company websites, regulatory filings, and reputable business databases

RESEARCH OBJECTIVES:
Using BOTH web search AND article analysis, determine:

- **Financial Group HQ**: Two-letter country code of the financial group's headquarters
  (Check: article mentions, company websites, business registries)

- **Group Industry**: Sector of the financial parent group (e.g. "Financial Services", "Technology & Payments", "Healthcare", "Services & Industrial Tech")
  (Check: article descriptions, company profiles, industry classifications)

- **Group Vertical**: Specific subsector within the chosen industry (e.g., Banking, Insurance, Asset Management, Digital Payments, Fintech, Software, Pharmaceuticals, Medical Devices, Healthcare Services)
  (Check: detailed business descriptions, company activities, operational focus)

- **Potential Disposal Country**: Two-letter country code where the potential disposal company is based
  (Check: article mentions, subsidiary locations, operational geographies)

- **Potential Disposal Industry**: Sector of the disposal company (e.g. "Financial Services", "Technology & Payments", "Healthcare", "Services & Industrial Tech")
  (Check: article business descriptions, subsidiary classifications, operational activities)

- **Potential Disposal Vertical**: Specific subsector within the disposal company's industry (e.g., Banking, Insurance, Asset Management, Digital Payments, Fintech, Software, Pharmaceuticals, Medical Devices, Healthcare Services)
  (Check: detailed operational descriptions, specific business activities, service offerings)

QUALITY STANDARDS:
- **Use multiple sources**: Don't rely on a single source; cross-reference information
- **Prefer explicit mentions**: Direct statements in articles or official sources are more reliable than inferences
- **Be conservative**: If information conflicts between sources or is ambiguous, return "Unknown"
- **Document your reasoning**: Base your conclusions on verifiable facts from your sources

IMPORTANT: Only provide information you can confidently verify through your research combining web search results AND article analysis. If uncertain about any field after checking all available sources, return "Unknown" for that specific item.
"""