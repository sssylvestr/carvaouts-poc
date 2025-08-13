from typing import Literal
from pydantic import BaseModel, Field


class CarveOutAssessment(BaseModel):
    is_co: bool = Field(description="Is the article about a future corporate carve-out?")
    target_company_code: str = Field(description="Code of the company that may be open to divest/carve-out its subsidiary")
    subsidiary_company_code: str = Field(description="Code of the subsidiary that may be open to divest/carve-out")
    is_relevant: bool = Field(description="Is the article relevant to the conditions of business request?")
    co_stage: Literal["Early", "Late"] = Field(description="Stage of the carve-out opportunity")
    short_reasoning: str = Field(description="Justification for the answers provided; not more than 2 sentences")


class CarveOutIdentificationSummary(BaseModel):
    target_company: str = Field(description="Name of the company potentially divesting a subsidiary")
    financial_group: str = Field(description="Ultimate parent group of the target company")

    potential_disposal: str = Field(description="Potential subdivision to be disposed of (e.g., UK business, Insurance Arm, IP business, etc.) - specific parts of the business being considered for divestiture")
    potential_disposal_company: str = Field(description="Name of the specific subsidiary or business unit considered for disposal")

    relevant: bool = Field(description="True if the potential disposal is within the EEA region, else False")
    interest_score: float = Field(description="Rate interest from 0.0 (low) to 1.0 (high), higher if signals are early-stage, strategic reviews, or management changes")
    rationale: str = Field(description="1–2 sentences explicitly summarizing why the carve-out may occur (e.g., divestment of non-core assets, strategic refocusing)")
    article_quote: str = Field(description="A direct, supportive quote from the article")

class SearchCarveOutIdentificationSummary(BaseModel):
    financial_group_hq: str = Field(description="Headquarters of the ultimate parent financial group; two-letter country code")
    group_industry: str = Field(description="Sector of the financial parent group (e.g. 'Financial Services', 'Technology & Payments', 'Healthcare', 'Services & Industrial Tech')")
    group_vertical: str = Field(description="Specific subsector within the chosen industry (e.g., Banking, Insurance, Asset Management, Digital Payments, Fintech, Software, Pharmaceuticals, Medical Devices, Healthcare Services)")
    potential_disposal_country: str = Field(description="Two-letter EEA country code of the disposal company")  
    potential_disposal_industry: str = Field(description="Sector of the disposal company (e.g. 'Financial Services', 'Technology & Payments', 'Healthcare', 'Services & Industrial Tech')")
    potential_disposal_vertical: str = Field(description="Specific subsector within the disposal company's industry (e.g., Banking, Insurance, Asset Management, Digital Payments, Fintech, Software, Pharmaceuticals, Medical Devices, Healthcare Services)")
