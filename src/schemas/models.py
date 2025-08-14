from typing import Literal, Optional
from pydantic import BaseModel, Field


class CarveOutAssessment(BaseModel):
    is_co: bool = Field(description="Is the article about a future corporate carve-out?")
    target_company_code: str = Field(description="Code of the company that may be open to divest/carve-out its subsidiary")
    subsidiary_company_code: str = Field(description="Code of the subsidiary that may be open to divest/carve-out")
    is_relevant: bool = Field(description="Is the article relevant to the conditions of business request?")
    co_stage: Literal["Early", "Late"] = Field(description="Stage of the carve-out opportunity")
    short_reasoning: str = Field(description="Justification for the answers provided; not more than 2 sentences")


class CarveOutIdentificationSummary(BaseModel):
    target_company: str = Field(description="Name of the company that may consider divesting a subsidiary")
    potential_disposal: str = Field(
        description="Potential subdivision to be disposed of, i.e. UK business, Insurance Arm, IP business, etc - specific parts of the business (if mentioned)",
    )
    potential_disposal_company: Optional[str] = Field(
        description="Name of the specific subsidiary or business unit considered for disposal (if mentioned)",
    )

    relevant: Optional[bool] = Field(
        default=None,
        description="True if the potential disposal is within the EEA; False if outside the EEA; null if unclear.",
    )
    interest_score: Literal[1, 2, 3, 4, 5] = Field(description="Interest level of the carve-out opportunity on a 1–5 scale (1=low, 5=high), based on timing and early-stage indicators")
    rationale: str = Field(description="Brief rationale (1-3 sentences with quotes) for why the carve-out opportunity exists, with included direct quotes from the article")

class SearchCarveOutIdentificationSummary(BaseModel):
    group: Optional[str] = Field(description="Ultimate parent financial group of the target company (if explicitly stated)")
    financial_group_hq: Optional[str] = Field(description="Headquarters of the ultimate parent financial group; two-letter country code")
    group_industry: str = Field(description="Sector of the financial parent group (e.g. 'Financial Services', 'Technology & Payments', 'Healthcare', 'Services & Industrial Tech')")
    group_vertical: str = Field(description="Specific subsector within the chosen industry (e.g., Banking, Insurance, Asset Management, Digital Payments, Fintech, Software, Pharmaceuticals, Medical Devices, Healthcare Services)")
    potential_disposal_company: Optional[str] = Field(description="Name of the specific subsidiary or business unit considered for disposal (if mentioned)")
    potential_disposal_country: Optional[str] = Field(description="Two-letter EEA country code of the disposal company")  
    potential_disposal_industry: str = Field(description="Sector of the disposal company (e.g. 'Financial Services', 'Technology & Payments', 'Healthcare', 'Services & Industrial Tech')")
    potential_disposal_vertical: str = Field(description="Specific subsector within the disposal company's industry (e.g., Banking, Insurance, Asset Management, Digital Payments, Fintech, Software, Pharmaceuticals, Medical Devices, Healthcare Services)")