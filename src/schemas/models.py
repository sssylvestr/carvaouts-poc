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

    # Optional enrichment fields (populate only if clearly stated in the article or provided in supplementary fields)
    group: str = Field(
        description="Ultimate parent financial group of the target company (if explicitly stated)",
    )
    group_hq: Optional[str] = Field(
        default=None,
        description="Headquarters of the ultimate parent financial group; two-letter country code (if known)",
    )
    vertical: Optional[str] = Field(
        default=None,
        description="Sector of the financial group, i.e. Banking, Insurance, Data, etc. (if stated)",
    )

    potential_disposal: str = Field(
        description="Potential subdivision to be disposed of, i.e. UK business, Insurance Arm, IP business, etc - specific parts of the business (if mentioned)",
    )
    potential_disposal_company: str = Field(
        description="Name of the specific subsidiary or business unit considered for disposal (if mentioned)",
    )
    potential_disposal_country: Optional[str] = Field(
        default=None,
        description="EEA country where the potential disposal company is based; two-letter country code (if known)",
    )
    disposal_nc_sector: Optional[Literal[
        "Financial Services",
        "Technology & Payments",
        "Healthcare",
        "Services & Industrial Tech",
        "Other",
    ]] = Field(
        default=None,
        description="The specific NC sector applicable to the disposal company (if determinable from the article)",
    )
    disposal_nc_vertical: Optional[str] = Field(
        default=None,
        description="The specific NC vertical applicable to the disposal company (if determinable from the article)",
    )

    relevant: Optional[bool] = Field(
        default=None,
        description="Boolean indicator if the article meets regional criteria for potential disposals within the EEA",
    )
    interest_score: Literal[1, 2, 3, 4, 5] = Field(description="Interest level of the carve-out opportunity on a 1–5 scale (1=low, 5=high), based on timing and early-stage indicators")
    rationale: str = Field(description="Brief rationale (1-3 sentences with quotes) for why the carve-out opportunity exists, with included direct quotes from the article")
