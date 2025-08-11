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
    target_company: str = Field(description="Name of the company that may consider divesting a subsidiary")
    group: str = Field(description="Ultimate parent financial group of the target company")
    group_hq: str = Field(description="Headquarters of the ultimate parent financial group; two-letter country code")
    group_sector: str = Field(description="Sector of the financial group, i.e. Banking, Insurance, Data, etc.")
    
    potential_disposal: str = Field(description="Potential subdivision to be disposed of, i.e. UK business, Insurance Arm, IP business, etc - specific parts of the business")
    potential_disposal_company: str = Field(description="Name of the specific subsidiary or business unit considered for disposal")
    potential_disposal_country: str = Field(description="EEA country where the potential disposal company is based; two-letter country code")  
    disposal_nc_sector: Literal["Financial Services", "Technology & Payments", "Healthcare","Services & Industrial Tech", "Other"] = Field(description="The specific sector applicable to the disposal company")

    article_quote: str = Field(description="A relevant direct quote from the article supporting the carve-out rationale")    
    relevant: bool = Field(description="Boolean indicator if the article meets regional criteria for potential disposals within the EEA")
    interest_score: float = Field(description="Interest level of carve-out opportunity between 0 (low) to 1 (high), based on strategic timing and early-stage indicators")
    rationale: str = Field(description="Brief rationale (1-2 sentences) for why the carve-out opportunity exists (e.g., strategic refocusing, divestment of non-core assets)")

class SearchCarveOutIdentificationSummary(BaseModel):
    group_hq: str = Field(description="Headquarters of the ultimate parent financial group; two-letter country code")
    group_sector: str = Field(description="Sector of the financial parent group, i.e. Banking, Insurance, Data, etc.")
    potential_disposal_country: str = Field(description="EEA country where the potential disposal company is based; two-letter country code")  
    disposal_nc_sector: Literal["Financial Services", "Technology & Payments", "Healthcare","Services & Industrial Tech", "Other", "Unknown"] = Field(description="The specific NC sector applicable to the disposal company")