"""Example HeartSafe rules demonstrating clinical claim validation.

These rules show the pattern for cardiology domain rules.
The complete rule set will be maintained by the HeartSafe team.
"""

import re

from vlm_guard.core.analysis import Analysis
from vlm_guard.core.rule import BaseRule, RuleResult, CrossClaimRule, CrossClaimResult


def _normalize_text(text: str) -> str:
    return text.replace("\u2264", "<=").replace("\u2265", ">=")


class CORLevelRule(BaseRule):
    """Validates Class of Recommendation (COR) levels against known lookup.

    Blocks: "MRA is Class 2a in HFrEF" (should be Class 1 for LVEF <=35%)
    Corrects: fuzzy matches
    """

    name = "heartsafe.cor_level"
    description = "Validates Class of Recommendation for GDMT therapies in HF"

    _COR_TABLE: dict[tuple[str, str], str] = {
        ("MRA", "HFrEF with LVEF <=35%, NYHA II-IV, on BB+RASi"): "Class 1",
        ("ARNi", "HFrEF"): "Class 1",
        ("SGLT2i", "HFpEF"): "Class 2a",
        ("Beta blocker", "HFrEF"): "Class 1",
        ("Hydralazine+ISDN", "African American HFrEF NYHA III-IV"): "Class 1",
        ("Ivabradine", "HFrEF LVEF <=35%, sinus rhythm >=70, on BB"): "Class 2a",
        ("Digoxin", "HFrEF"): "Class 2b",
        ("Tafamidis", "ATTR-CM NYHA I-III"): "Class 1",
    }

    def condition(self, analysis: Analysis, context: dict) -> bool:
        return analysis.domain == "cardiology"

    def action(self, analysis: Analysis, context: dict) -> tuple[Analysis, RuleResult]:
        text = _normalize_text(analysis.claim_text.lower())
        label_lower = analysis.label.lower()

        cor_mentioned = None
        for cor in ["class 1", "class i", "class 2a", "class iia",
                     "class 2b", "class iib", "class 3", "class iii"]:
            if cor in text:
                cor_mentioned = cor
                break

        if not cor_mentioned:
            return analysis, RuleResult(action_taken=False)

        matched = False
        condition_from_meta = _normalize_text(analysis.metadata.get("condition", ""))
        for (therapy, condition), expected_cor in self._COR_TABLE.items():
            if therapy.lower() in text and (
                any(c.strip() in text for c in _normalize_text(condition.lower()).split(","))
                or condition_from_meta == condition
            ):
                matched = True
                expected_lower = expected_cor.lower()
                if cor_mentioned != expected_lower:
                    suggestion = (
                        f"Claim states {cor_mentioned.title()} for {therapy}, "
                        f"but guideline states {expected_cor}. "
                        f"Condition: {condition}"
                    )
                    analysis.validation_status = "blocked"
                    analysis.label = expected_cor
                    analysis.validation_message = suggestion
                    return analysis, RuleResult(
                        action_taken=True,
                        action_type="block",
                        message=suggestion,
                        severity="error",
                        correction_suggestion=suggestion,
                        modified_fields={
                            "validation_status": "blocked",
                            "label": expected_cor,
                        },
                    )
                else:
                    analysis.validation_status = "passed"
                    analysis.label = expected_cor
                    return analysis, RuleResult(
                        action_taken=True,
                        action_type="pass",
                        message=f"COR {expected_cor} confirmed for {therapy}",
                    )

        if not matched:
            analysis.validation_status = "unverifiable"
            return analysis, RuleResult(
                action_taken=True,
                action_type="flag",
                message=f"COR claim for '{label_lower}' not in lookup table \u2014 unverifiable",
                severity="info",
            )

        return analysis, RuleResult()


class LVEFThresholdRule(BaseRule):
    """Validates LVEF ranges against guideline definitions.

    Corrects: LVEF classification
    Example: "LVEF 38% is normal" -> corrected to "HFrEF"
    """

    name = "heartsafe.lvef_threshold"
    description = "Validates LVEF ranges and corrects misclassification"

    _EF_RANGES = [
        ("HFrEF", 0, 40),
        ("HFmrEF", 41, 49),
        ("HFpEF", 50, 100),
    ]

    def condition(self, analysis: Analysis, context: dict) -> bool:
        return analysis.domain == "cardiology" and "lvef" in analysis.claim_text.lower()

    def action(self, analysis: Analysis, context: dict) -> tuple[Analysis, RuleResult]:
        text = analysis.claim_text.lower()

        ef_match = re.search(r"lvef\s*(?:is|=|of)?\s*(\d+)\s*%?", text)
        if not ef_match:
            ef_match = re.search(r"(\d+)\s*%\s*(?:lvef|ef|ejection fraction)", text)
        if not ef_match:
            return analysis, RuleResult(action_taken=False)

        ef_value = int(ef_match.group(1))

        correct_label = "Normal"
        for stage_name, lo, hi in self._EF_RANGES:
            if lo <= ef_value <= hi:
                correct_label = stage_name
                break

        current_label = analysis.label
        if current_label != correct_label:
            analysis.label = correct_label
            analysis.validation_status = "corrected"
            analysis.validation_message = (
                f"LVEF {ef_value}% \u2192 {correct_label} "
                f"(was labeled as '{current_label}')"
            )
            return analysis, RuleResult(
                action_taken=True,
                action_type="correct",
                message=analysis.validation_message,
                severity="warning",
                correction_suggestion=f"LVEF {ef_value}% corresponds to {correct_label}, not {current_label}",
                modified_fields={"label": correct_label, "validation_status": "corrected"},
            )

        analysis.validation_status = "passed"
        return analysis, RuleResult(action_taken=True, action_type="pass", message=f"LVEF {ef_value}% confirmed as {correct_label}")


class AnswerConsistencyRule(CrossClaimRule):
    """Ensures the final answer is consistent with all validated claims.

    If all claims say "MRA Class 1" but answer says "MRA is not recommended",
    this rule blocks the answer.
    """

    name = "heartsafe.answer_consistency"
    description = "Cross-validates final answer against all claims"
    order = 2000

    def condition(
        self,
        claims: list[Analysis],
        answer: str,
        context: dict,
    ) -> bool:
        return bool(claims) and bool(answer)

    def action(
        self,
        claims: list[Analysis],
        answer: str,
        context: dict,
    ) -> tuple[list[Analysis], str, CrossClaimResult]:
        answer_lower = answer.lower()

        claim_labels = []
        for c in claims:
            if c.validation_status in ("passed", "corrected", "flagged", "unverifiable"):
                claim_labels.append(c.label)

        contradictions = self._find_contradictions(claim_labels, answer_lower)

        if contradictions:
            corrected_answer = self._apply_corrections(answer, contradictions)
            return claims, corrected_answer, CrossClaimResult(
                action_taken=True,
                action_type="correct" if corrected_answer != answer else "block",
                message=f"Answer contradicted claims. Fixed: {'; '.join(contradictions)}",
                severity="error",
                modified_answer=corrected_answer,
            )

        return claims, answer, CrossClaimResult(action_taken=False)

    def _find_contradictions(
        self,
        claim_labels: list[str],
        answer_lower: str,
    ) -> list[str]:
        contradictions = []
        pos_verbs = ["recommended", "indicated", "beneficial", "improves"]
        neg_verbs = ["not recommended", "contraindicated", "should not", "avoid"]

        has_negation = any(v in answer_lower for v in neg_verbs)
        has_positive = any(v in answer_lower for v in pos_verbs) and not has_negation

        for label in claim_labels:
            if "Class 1" in label or "Class 2a" in label:
                if has_negation:
                    contradictions.append(f"'{label}' in claims contradicts 'not recommended' in answer")
            if "Class 3" in label or "should not" in label.lower():
                if has_positive:
                    contradictions.append(f"'{label}' in claims contradicts 'recommended' in answer")

        return contradictions

    @staticmethod
    def _apply_corrections(
        answer: str,
        contradictions: list[str],
    ) -> str:
        return answer + " [NOTE: Answer may contradict validated claims \u2014 review required]"
