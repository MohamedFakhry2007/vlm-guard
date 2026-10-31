"""Cardiology-specific Analysis helpers for HeartSafe."""

from typing import Any, Literal

from vlm_guard.core.analysis import Analysis


CardiologyClaimType = Literal[
    "diagnosis", "recommendation", "threshold",
    "contraindication", "definition", "value_statement",
    "drug_interaction", "refusal", "other",
]

CardiologyCOR = Literal[
    "Class 1", "Class 2a", "Class 2b", "Class 3: No Benefit", "Class 3: Harm",
]

CardiologyHFType = Literal[
    "HFrEF", "HFmrEF", "HFpEF", "HFimpEF", "At Risk", "Pre-HF", "Advanced HF",
]


def cardiology_claim(
    *,
    label: str,
    claim_text: str,
    claim_type: CardiologyClaimType = "other",
    confidence: str = "Medium",
    condition: str | None = None,
    cor: CardiologyCOR | None = None,
    lvef: int | None = None,
    drug: str | None = None,
    evidence: str = "",
) -> Analysis:
    """Create an Analysis object with cardiology-specific metadata."""
    meta: dict[str, Any] = {}
    if condition:
        meta["condition"] = condition
    if cor:
        meta["cor"] = cor
    if lvef is not None:
        meta["lvef"] = lvef
    if drug:
        meta["drug"] = drug

    return Analysis(
        label=label,
        domain="cardiology",
        claim_type=claim_type,
        claim_text=claim_text,
        confidence=confidence,
        evidence=evidence,
        findings=claim_text,
        metadata=meta,
    )
