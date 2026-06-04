"""Parse LLM structured reasoning output into replicable Analysis objects."""

import json
import re
from typing import Any

from vlm_guard.core.analysis import Analysis


def parse_reasoning_steps(
    llm_output: str,
    domain: str = "generic",
    default_confidence: str = "Medium",
) -> tuple[list[Analysis], str, dict[str, Any] | None]:
    """Parse structured JSON output from LLM into claims + answer.

    Expected input format:
    ```json
    {
      "reasoning_steps": [
        {
          "step": "Patient has HFrEF with LVEF 30%",
          "claim_type": "diagnosis",
          "grounding": "Per guideline section 5.2..."
        },
        {
          "step": "LVEF <=35% threshold for MRA is met",
          "claim_type": "threshold",
          "grounding": "..."
        }
      ],
      "answer": "MRA is recommended (Class 1)..."
    }
    ```

    Returns:
        (claims, answer, raw_dict_or_None)
    """
    clean = re.sub(r"```json|```", "", llm_output).strip()

    try:
        parsed = json.loads(clean)
    except json.JSONDecodeError:
        match = re.search(r"\{[\s\S]*\"reasoning_steps\"[\s\S]*\"answer\"[\s\S]*\}", clean)
        if match:
            try:
                parsed = json.loads(match.group())
            except json.JSONDecodeError:
                return _fallback_parse(llm_output, domain)
        else:
            return _fallback_parse(llm_output, domain)

    if not isinstance(parsed, dict):
        return _fallback_parse(llm_output, domain)

    raw_answer = parsed.get("answer", "")
    raw_steps = parsed.get("reasoning_steps", [])
    if not isinstance(raw_steps, list):
        raw_steps = []

    claims = []
    for i, step in enumerate(raw_steps):
        if isinstance(step, str):
            claims.append(Analysis(
                label=_infer_label(step, domain),
                domain=domain,
                claim_type=_infer_claim_type(step),
                claim_text=step,
                evidence=step,
                findings=step,
                confidence=default_confidence,
                metadata={"step_index": i},
            ))
        elif isinstance(step, dict):
            claim_text = step.get("step", step.get("claim", ""))
            claims.append(Analysis(
                label=step.get("label", _infer_label(claim_text, domain)),
                domain=domain,
                claim_type=step.get("claim_type", _infer_claim_type(claim_text)),
                claim_text=claim_text,
                evidence=step.get("grounding", step.get("evidence", "")),
                findings=claim_text,
                confidence=step.get("confidence", default_confidence),
                metadata={
                    "step_index": i,
                    **(step.get("metadata", {})),
                },
            ))

    return claims, raw_answer, parsed


def _fallback_parse(
    raw_text: str,
    domain: str,
) -> tuple[list[Analysis], str, None]:
    return [
        Analysis(
            label="Unclear",
            domain=domain,
            claim_type="other",
            claim_text=raw_text[:500],
            evidence="Model output could not be parsed as JSON",
            findings=raw_text[:500],
            confidence="Low",
            recommendation="Model response was not structured correctly. Review raw output.",
        )
    ], raw_text, None


def _infer_label(text: str, domain: str) -> str:
    for keyword in [
        "Class 1", "Class I", "Class 2a", "Class IIa",
        "Class 2b", "Class IIb", "Class 3", "Class III",
        "HFrEF", "HFmrEF", "HFpEF", "HFimpEF",
        "ARNi", "ACEi", "ARB", "MRA", "SGLT2i",
    ]:
        if keyword.lower() in text.lower():
            return keyword
    return "Clinical Finding"


def _infer_claim_type(text: str) -> str:
    text_lower = text.lower()
    if any(w in text_lower for w in ["class 1", "class i", "recommended", "reasonable", "may be considered"]):
        return "recommendation"
    if any(w in text_lower for w in ["lvef", "threshold", "\u2264", "\u2265", ">", "<", "%"]):
        return "threshold"
    if any(w in text_lower for w in ["defined as", "refers to", "is defined"]):
        return "definition"
    if any(w in text_lower for w in ["contraindicated", "should not", "avoid"]):
        return "contraindication"
    if "value" in text_lower and "$" in text:
        return "value_statement"
    if any(w in text_lower for w in ["cannot answer", "do not contain", "unable to"]):
        return "refusal"
    return "diagnosis"
