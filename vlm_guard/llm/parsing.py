import json
import re
from typing import Any

from vlm_guard.core.analysis import Analysis


def extract_json(text: str) -> dict[str, Any] | None:
    clean = re.sub(r"```json|```", "", text).strip()
    match = re.search(r"\{[\s\S]*\"label\"[\s\S]*?\}", clean)
    if match:
        return json.loads(match.group())
    match = re.search(r"\{[\s\S]*\"detected_disease\"[\s\S]*?\}", clean)
    if match:
        return json.loads(match.group())
    match = re.search(r"\{[\s\S]*\"findings\"[\s\S]*?\}", clean)
    if match:
        return json.loads(match.group())
    return None


def parse_to_analysis(text: str) -> tuple[Analysis, str]:
    clean = re.sub(r"```json|```", "", text).strip()
    json_match = re.search(r"\{[\s\S]*\"label\"[\s\S]*?\}", clean)

    if not json_match:
        json_match = re.search(r"\{[\s\S]*\"detected_disease\"[\s\S]*?\}", clean)
    if not json_match:
        json_match = re.search(r"\{[\s\S]*\"findings\"[\s\S]*?\}", clean)

    if json_match:
        try:
            parsed = json.loads(json_match.group())
            allowed = set(Analysis.model_fields.keys())
            filtered = {k: v for k, v in parsed.items() if k in allowed}
            if "label" not in filtered and "detected_disease" in parsed:
                filtered["label"] = parsed["detected_disease"]
            analysis = Analysis(**filtered)
            return analysis, text
        except (json.JSONDecodeError, ValueError):
            pass

    return Analysis(
        label="Unclear",
        confidence="Low",
        evidence="Model output could not be parsed",
        findings=text[:500],
        recommendation="Model response was not structured correctly. Review raw output.",
    ), text
