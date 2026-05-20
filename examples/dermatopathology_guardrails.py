"""Dermatopathology example: catching biologically impossible VLM outputs.

Demonstrates two custom guardrail rules for skin pathology that catch
common hallucination patterns in multimodal LLM outputs:
  1. Leishmania amastigotes reported in stratum corneum (dead keratin)
  2. Malassezia (superficial yeast) reported in deep dermis/subcutis
"""

from vlm_guard import GuardrailEngine, BaseRule, RuleResult, Analysis


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _text(analysis: Analysis) -> str:
    return (analysis.findings + " " + analysis.evidence).lower()


def _in_deep_tissue(text: str) -> bool:
    return any(p in text for p in [
        "dermis", "deep dermis", "reticular dermis",
        "subcutis", "subcutaneous", "deep tissue",
    ])


def _in_stratum_corneum(text: str) -> bool:
    return any(p in text for p in [
        "stratum corneum", "cornified", "dead keratino",
        "surface keratin", "horny layer",
    ])


# ---------------------------------------------------------------------------
# Rule 1: Leishmania cannot reside in stratum corneum
# ---------------------------------------------------------------------------

class LeishmaniaCorneumContradictionRule(BaseRule):
    name = "derm.leishmania_corneum_contradiction"
    description = (
        "Leishmania amastigotes are obligate intracellular parasites of "
        "macrophages and cannot reside in dead stratum corneum keratinocytes"
    )

    def condition(self, analysis: Analysis, context: dict) -> bool:
        t = _text(analysis)
        meta_type = analysis.metadata.get("type", "").lower()
        meta_org = analysis.metadata.get("organism", "").lower()
        leishmania_label = "leishmania" in analysis.label.lower()
        leishmania_meta = "leishmania" in meta_org
        corneum = _in_stratum_corneum(t) or _in_stratum_corneum(meta_type)
        return corneum and (leishmania_label or leishmania_meta)

    def action(self, analysis: Analysis, context: dict) -> tuple[Analysis, RuleResult]:
        analysis.label = "Suspicious Morphological Analysis"
        analysis.confidence = "Low"
        analysis.recommendation = (
            "Leishmania requires intracellular confirmation in living tissue. "
            "Order a Giemsa or Leishmania IHC stain to rule out "
            "Leishmaniasis before initiating therapy."
        )
        analysis.evidence = (
            f"CONTRADICTION: {self.description}. "
            f"Original finding placed Leishmania in stratum corneum, "
            f"which is biologically impossible. "
            f"Original description: {analysis.evidence}"
        )
        return analysis, RuleResult(
            action_taken=True,
            action_type="correct",
            message=(
                "Corrected impossible structural mapping: "
                "Leishmania cannot reside in dead keratin scale layer"
            ),
            modified_fields={
                "label": "Suspicious Morphological Analysis",
                "confidence": "Low",
            },
        )


# ---------------------------------------------------------------------------
# Rule 2: Malassezia is superficial; deep location is contradictory
# ---------------------------------------------------------------------------

class MalasseziaDepthContradictionRule(BaseRule):
    name = "derm.malassezia_depth_contradiction"
    description = (
        "Malassezia is a commensal yeast of the superficial stratum corneum; "
        "deep dermal or subcutaneous location is biologically contradictory"
    )

    def condition(self, analysis: Analysis, context: dict) -> bool:
        t = _text(analysis)
        malassezia_mentioned = "malassezia" in t or "pityrosporum" in t
        deep_location = _in_deep_tissue(t)
        label_unrelated = "malassezia" in analysis.label.lower()
        return malassezia_mentioned and deep_location and not label_unrelated

    def action(self, analysis: Analysis, context: dict) -> tuple[Analysis, RuleResult]:
        analysis.label = "Unclear"
        analysis.confidence = "Low"
        analysis.recommendation = (
            "Malassezia is a superficial stratum corneum yeast. "
            "Deep dermal/subcutaneous organisms are unlikely to be Malassezia. "
            "Consider deep fungal infection (e.g., dematiaceous fungi, "
            "phaeohyphomycosis) and order PAS or GMS stain with culture."
        )
        analysis.evidence = (
            f"CONTRADICTION: {self.description}. "
            f"Original finding placed yeast forms in deep tissue, "
            f"which is inconsistent with Malassezia biology."
        )
        return analysis, RuleResult(
            action_taken=True,
            action_type="block",
            message=(
                "Malassezia reported in deep dermis - blocked; "
                "superficial yeast cannot cause deep infection"
            ),
            modified_fields={"label": "Unclear", "confidence": "Low"},
        )


# ---------------------------------------------------------------------------
# Demo: run both scenarios
# ---------------------------------------------------------------------------

def _print_divider(title: str):
    print()
    print("=" * 65)
    print(f"  {title}")
    print("=" * 65)


def _print_audit(audit):
    print(f"{'Rule Fired':40s} {'Action':10s} Message")
    print("-" * 65)
    for entry in audit.summary():
        print(f"{entry['rule']:40s} {entry['action']:10s} {entry['message']}")
    print("-" * 65)


def _print_report(label, confidence, recommendation):
    print(f"\nCorrected Diagnosis:  {label}")
    print(f"Adjusted Confidence:  {confidence}")
    print(f"New Clinical Actions:  {recommendation}")


def scenario_leishmania_corneum():
    _print_divider("Scenario 1: Leishmania in Stratum Corneum (Dead Skin)")

    vlm_output = Analysis(
        label="Cutaneous Leishmaniasis",
        confidence="High",
        findings=(
            "Small round structures with distinct kinetoplasts observed "
            "within stratum corneum layer."
        ),
        evidence="Visual assessment of H&E histology section",
        recommendation="Initiate standard antimonial therapy.",
        metadata={
            "type": "Stratum Corneum",
            "organism": "Leishmania amastigotes",
        },
    )

    print(f"\nRaw VLM Output -> {vlm_output.label} (confidence: {vlm_output.confidence})")
    print(f"  Location: {vlm_output.metadata['type']}")
    print(f"  Organism: {vlm_output.metadata['organism']}")

    engine = GuardrailEngine()
    engine.register(LeishmaniaCorneumContradictionRule())
    engine.register(MalasseziaDepthContradictionRule())

    final, audit = engine.apply_with_audit(vlm_output, context={})

    print("\n--- Audit Trail ---")
    _print_audit(audit)

    print("\n--- Re-routed Clinical Report ---")
    _print_report(final.label, final.confidence, final.recommendation)
    print()


def scenario_malassezia_deep():
    _print_divider("Scenario 2: Malassezia in Deep Dermis")

    vlm_output = Analysis(
        label="Dermatitis, unspecified",
        confidence="Medium",
        findings=(
            "Budding yeast forms observed within reticular dermis "
            "with mild perivascular inflammation. "
            "Morphology consistent with Malassezia species."
        ),
        evidence="PAS-stained skin biopsy showing yeast forms",
        recommendation="Consider antifungal therapy.",
        metadata={
            "type": "Deep Dermis",
            "organism": "Malassezia spp.",
        },
    )

    print(f"\nRaw VLM Output -> {vlm_output.label} (confidence: {vlm_output.confidence})")
    print(f"  Findings: {vlm_output.findings[:80]}...")

    engine = GuardrailEngine()
    engine.register(LeishmaniaCorneumContradictionRule())
    engine.register(MalasseziaDepthContradictionRule())

    final, audit = engine.apply_with_audit(vlm_output, context={})

    print("\n--- Audit Trail ---")
    _print_audit(audit)

    print("\n--- Re-routed Clinical Report ---")
    _print_report(final.label, final.confidence, final.recommendation)
    print()


def main():
    scenario_leishmania_corneum()
    scenario_malassezia_deep()

    print("=" * 65)
    print("  Both scenarios complete. Biologically impossible outputs")
    print("  were caught by the guardrail engine before reaching the clinician.")
    print("=" * 65)


if __name__ == "__main__":
    main()
