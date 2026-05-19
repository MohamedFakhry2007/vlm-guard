"""Minimal example: defining a custom rule with VLM-Guard."""

from vlm_guard import GuardrailEngine, BaseRule, RuleResult, Analysis


class PlantRule(BaseRule):
    """Rejects impossible plant morphology combinations."""

    name = "plant.sun_vs_shade"
    description = "Sun-loving plants should not show shade-adapted morphology"

    def condition(self, analysis, context):
        text = (analysis.findings + " " + analysis.evidence).lower()
        return "sun-loving" in text and "large thin leaves" in text

    def action(self, analysis, context):
        analysis.label = "Unclear"
        analysis.confidence = "Low"
        analysis.recommendation = (
            "Sun-loving plants typically have small, thick leaves. "
            "Large thin leaves suggest shade adaptation. "
            "Please review identification."
        )
        return analysis, RuleResult(
            action_taken=True,
            action_type="flag",
            message="Sun-loving plant with shade morphology → flagged",
            modified_fields={"label": "Unclear", "confidence": "Low"},
        )


def main():
    engine = GuardrailEngine()
    engine.register(PlantRule())

    result = Analysis(
        label="Quercus rubra",
        confidence="High",
        evidence="Large thin leaves observed",
        findings="Sun-loving oak species with large thin shade-adapted leaves",
        recommendation="None",
    )

    final, audit = engine.apply_with_audit(result)
    print(f"Before: {result.label} (confidence: {result.confidence})")
    print(f"After:  {final.label} (confidence: {final.confidence})")
    print(f"Audit:  {audit.summary()}")


if __name__ == "__main__":
    main()
