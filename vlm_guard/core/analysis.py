from pydantic import BaseModel, Field
from typing import Any, Literal


class Analysis(BaseModel):
    model_config = {"extra": "forbid", "validate_assignment": True}

    label: str = Field(
        description="Detection/diagnosis label (domain-specific, e.g. disease name, defect type, classification)"
    )
    confidence: Literal["High", "Medium", "Low"] = "Medium"
    evidence: str = Field(default="", description="Specific evidence supporting the label")
    findings: str = Field(default="", description="Comprehensive description of observations")
    recommendation: str = Field(default="", description="Recommended next steps")
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Extension point for domain-specific fields (severity, species, location, etc.)"
    )
