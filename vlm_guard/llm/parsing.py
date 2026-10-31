import json
import re
from typing import Any

from vlm_guard.core.analysis import Analysis


# Keys that identify a model output object, in order of preference.
_ANCHOR_KEYS = ("label", "detected_disease", "findings")


def _strip_fences(text: str) -> str:
    return re.sub(r"```json|```", "", text).strip()


def _find_json_object(text: str) -> dict[str, Any] | None:
    """Return the first complete JSON object in ``text`` that has an anchor key.

    Tries ``json.JSONDecoder().raw_decode`` from every ``{`` so nested objects
    are parsed whole instead of being cut at the first ``}``. Anchor keys are
    tried in order of preference, matching only top-level keys.
    """
    decoder = json.JSONDecoder()
    objects: list[dict[str, Any]] = []
    for match in re.finditer(r"\{", text):
        try:
            obj, _ = decoder.raw_decode(text, match.start())
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            objects.append(obj)

    for key in _ANCHOR_KEYS:
        for obj in objects:
            if key in obj:
                return obj
    return None


def extract_json(text: str) -> dict[str, Any] | None:
    return _find_json_object(_strip_fences(text))


def parse_to_analysis(text: str) -> tuple[Analysis, str]:
    parsed = _find_json_object(_strip_fences(text))

    if parsed is not None:
        try:
            allowed = set(Analysis.model_fields.keys())
            filtered = {k: v for k, v in parsed.items() if k in allowed}
            if "label" not in filtered and "detected_disease" in parsed:
                filtered["label"] = parsed["detected_disease"]
            analysis = Analysis(**filtered)
            return analysis, text
        except ValueError:
            pass

    return Analysis(
        label="Unclear",
        confidence="Low",
        evidence="Model output could not be parsed",
        findings=text[:500],
        recommendation="Model response was not structured correctly. Review raw output.",
    ), text
