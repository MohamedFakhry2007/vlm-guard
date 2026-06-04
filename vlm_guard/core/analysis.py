from typing import Any, Literal

from pydantic import BaseModel, Field


class Analysis(BaseModel):
    model_config = {"extra": "forbid", "validate_assignment": True}

    # Core identification
    label: str = Field(
        description="Claim label — e.g., disease name, therapy, COR level, threshold"
    )
    domain: str = Field(
        default="generic",
        description="Domain namespace for rules: 'generic', 'cardiology', 'ntd_microscopy', ..."
    )
    claim_type: Literal[
        "diagnosis",
        "recommendation",
        "threshold",
        "contraindication",
        "definition",
        "value_statement",
        "refusal",
        "other"
    ] = Field(default="other")

    # Claim text (what the LLM actually said)
    claim_text: str = Field(
        default="",
        description="The exact verbatim claim text from the LLM reasoning step"
    )

    # Confidence / Evidence
    confidence: Literal["High", "Medium", "Low"] = "Medium"
    evidence: str = Field(default="", description="Supporting evidence from the claim or guideline excerpt")
    findings: str = Field(default="", description="Comprehensive description of observations / full reasoning step text")
    recommendation: str = Field(default="", description="Recommended next steps or caveats")

    # Extension point
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Domain-specific extensions: severity, species, LVEF value, drug, COR, ..."
    )

    # Validation state (set by the engine, not by user creation)
    validation_status: Literal["pending", "passed", "blocked", "corrected", "flagged", "unverifiable"] = "pending"
    validation_message: str = ""
