"""Regression tests for the remaining Phase 0 items.

- HeartSafe no longer ships as a top-level ``plugins`` package.
- VLMGuardPipeline status comes from verification, not the model's confidence.
- NTD rules do not treat negated findings as positive evidence.
"""

import tomllib
from pathlib import Path

from PIL import Image

from vlm_guard import Analysis, BaseRule, GuardrailEngine, RuleResult, VLMGuardPipeline
from vlm_guard.llm.parsing import parse_to_analysis
from vlm_guard.plugins.ntd_microscopy import register_ntd_rules

ROOT = Path(__file__).resolve().parent.parent


# ─── Top-level ``plugins`` package ───────────────────────────

def test_heartsafe_importable_from_vlm_guard_namespace():
    from vlm_guard.plugins.heartsafe import register_heartsafe_rules
    from vlm_guard.plugins.heartsafe.rules import CORLevelRule
    from vlm_guard.plugins.heartsafe.schema import cardiology_claim

    engine = GuardrailEngine()
    register_heartsafe_rules(engine)
    assert any(isinstance(r, CORLevelRule) for r in engine.rules)
    assert cardiology_claim(label="x", claim_text="y").domain == "cardiology"


def test_no_top_level_plugins_package_in_repo_or_build():
    assert not (ROOT / "plugins" / "heartsafe").exists()
    config = tomllib.loads((ROOT / "pyproject.toml").read_text())
    include = config["tool"]["setuptools"]["packages"]["find"]["include"]
    assert all(p.startswith("vlm_guard") for p in include), include


# ─── Pipeline status not derived from self-confidence ────────

class _BlockAllRule(BaseRule):
    name = "test.block_all"
    description = "Blocks everything"

    def condition(self, analysis, context):
        return True

    def action(self, analysis, context):
        analysis.validation_status = "blocked"
        return analysis, RuleResult(action_taken=True, action_type="block", message="no", severity="error")


class _NeverRule(BaseRule):
    name = "test.never"
    description = "Never fires"

    def condition(self, analysis, context):
        return False

    def action(self, analysis, context):
        return analysis, RuleResult()


class _RelabelRule(BaseRule):
    name = "test.relabel"
    description = "Relabels without touching validation_status"

    def condition(self, analysis, context):
        return True

    def action(self, analysis, context):
        analysis.label = "Other"
        return analysis, RuleResult(action_taken=True, action_type="promote", message="relabelled")


def _run(rules, confidence):
    engine = GuardrailEngine()
    engine.register_list(rules)
    pipeline = VLMGuardPipeline(
        model_fn=lambda image, prompt: f'{{"label": "x", "confidence": "{confidence}"}}',
        parser_fn=parse_to_analysis,
        guardrail_engine=engine,
    )
    return pipeline.run(Image.new("RGB", (4, 4)), "prompt")


def test_high_confidence_blocked_output_is_not_ok():
    result = _run([_BlockAllRule()], "High")
    assert result.status == "blocked"


def test_low_confidence_passing_output_is_passed():
    result = _run([_NeverRule()], "Low")
    assert result.status == "passed"


def test_rule_modification_is_reflected_in_status():
    result = _run([_RelabelRule()], "High")
    assert result.status == "corrected"


def test_no_rules_means_unverified():
    result = _run([], "High")
    assert result.status == "pending"


def test_producer_confidence_kept_as_metadata_only():
    result = _run([_NeverRule()], "Low")
    assert result.metadata["producer_confidence"] == "Low"


# ─── NTD negation stopgap ────────────────────────────────────

def _ntd(label, findings, sample_type, confidence="Medium", evidence=""):
    engine = GuardrailEngine()
    register_ntd_rules(engine)
    analysis = Analysis(
        label=label, confidence=confidence, findings=findings, evidence=evidence, recommendation="",
    )
    return engine.apply(analysis, context={"sample_type": sample_type})


def test_negated_flagellates_do_not_promote_trypanosomiasis():
    out = _ntd("Unclear", "No flagellated organisms seen. Debris is extracellular in plasma.", "CSF")
    assert out.label == "Unclear"


def test_affirmed_flagellates_still_promote_trypanosomiasis():
    out = _ntd("Unclear", "Flagellated organisms seen, extracellular in plasma.", "CSF")
    assert out.label == "Trypanosomiasis"


def test_post_negation_does_not_count_as_evidence():
    out = _ntd("Unclear", "Trypomastigotes were not seen; cells are extracellular in plasma.", "CSF")
    assert out.label == "Unclear"


def test_negation_scoped_to_clause():
    out = _ntd("Unclear", "No ring forms. Flagellated organisms extracellular in plasma.", "CSF")
    assert out.label == "Trypanosomiasis"


def test_negated_amastigotes_do_not_promote_leishmaniasis():
    out = _ntd("Unclear", "Macrophages examined; no amastigotes identified.", "Bone Marrow")
    assert out.label == "Unclear"


def test_uncertainty_is_not_negation():
    out = _ntd("Unclear", "Amastigotes cannot be excluded.", "Bone Marrow")
    assert out.label == "Leishmaniasis"


def test_negation_in_findings_does_not_leak_into_evidence():
    out = _ntd(
        "Unclear", "No ring forms", "CSF",
        evidence="Flagellated organisms extracellular in plasma.",
    )
    assert out.label == "Trypanosomiasis"


def test_trailing_absent_is_negation():
    out = _ntd("Unclear", "Flagellated forms absent. Debris extracellular in plasma.", "CSF")
    assert out.label == "Unclear"


def test_no_sheath_still_reads_as_unsheathed():
    from vlm_guard.plugins.ntd_microscopy.rules import _drop_negated, _says_unsheathed
    assert _says_unsheathed(_drop_negated("Microfilaria with no sheath; nuclei to tail tip."))
