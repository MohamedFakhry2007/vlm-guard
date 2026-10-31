"""Regression tests for the Phase 0 bug fixes.

Each test reproduces a verified bug and pins the corrected behaviour.
"""

import subprocess
import sys

from PIL import Image

from vlm_guard import (
    Analysis,
    BaseRule,
    GuardrailEngine,
    RuleResult,
    TextGuardPipeline,
    VLMGuardPipeline,
)
from vlm_guard.llm.parsing import extract_json, parse_to_analysis

from vlm_guard.plugins.heartsafe import register_heartsafe_rules
from vlm_guard.plugins.heartsafe.rules import CORLevelRule
from vlm_guard.plugins.heartsafe.schema import cardiology_claim


# ─── 1. Status overwrite ─────────────────────────────────────

def test_blocked_status_not_overwritten_by_non_acting_rule():
    engine = GuardrailEngine()
    register_heartsafe_rules(engine, include_cross_claim=False)

    claim = cardiology_claim(
        label="MRA",
        claim_text="MRA is Class 2a in HFrEF with LVEF <=35%, NYHA II-IV, on BB+RASi; lvef reduced",
        claim_type="recommendation",
    )
    result, audit = engine.apply_with_audit(claim)

    assert result.label == "Class 1"
    assert any(e.action_type == "block" for e in audit.entries)
    assert result.validation_status == "blocked"


def test_non_pending_status_never_downgraded_to_passed():
    class FlagRule(BaseRule):
        name = "test.flag"
        description = "Flags everything"

        def condition(self, analysis, context):
            return True

        def action(self, analysis, context):
            analysis.validation_status = "flagged"
            return analysis, RuleResult(action_taken=True, action_type="flag", message="flagged")

    class NoopRule(BaseRule):
        name = "test.noop"
        description = "Matches but takes no action"

        def condition(self, analysis, context):
            return True

        def action(self, analysis, context):
            return analysis, RuleResult(action_taken=False)

    class PassRule(BaseRule):
        name = "test.pass"
        description = "Explicitly passes"

        def condition(self, analysis, context):
            return True

        def action(self, analysis, context):
            analysis.validation_status = "passed"
            return analysis, RuleResult(action_taken=True, action_type="pass", message="ok")

    engine = GuardrailEngine()
    engine.register_list([FlagRule(), NoopRule(), PassRule()])
    result = engine.apply(Analysis(label="x", confidence="High"))
    assert result.validation_status == "flagged"


def test_pending_status_still_becomes_passed():
    class NoopRule(BaseRule):
        name = "test.noop"
        description = "Matches but takes no action"

        def condition(self, analysis, context):
            return True

        def action(self, analysis, context):
            return analysis, RuleResult(action_taken=False)

    engine = GuardrailEngine()
    engine.register(NoopRule())
    result = engine.apply(Analysis(label="x", confidence="High"))
    assert result.validation_status == "passed"


# ─── 2. Roman numeral COR parsing ───────────────────────────

def test_cor_class_iia_not_parsed_as_class_i():
    engine = GuardrailEngine()
    engine.register(CORLevelRule())
    claim = cardiology_claim(
        label="Ivabradine",
        claim_text="Ivabradine is Class IIa in HFrEF LVEF <=35%, sinus rhythm >=70, on BB",
        claim_type="recommendation",
    )
    result, audit = engine.apply_with_audit(claim)

    assert result.validation_status == "passed"
    assert result.label == "Class 2a"
    assert not any(e.action_type == "block" for e in audit.entries)


def test_cor_roman_iib_and_iii_normalised():
    engine = GuardrailEngine()
    engine.register(CORLevelRule())

    digoxin = cardiology_claim(label="Digoxin", claim_text="Digoxin is Class IIb in HFrEF")
    assert engine.apply(digoxin).validation_status == "passed"

    wrong = cardiology_claim(label="Digoxin", claim_text="Digoxin is Class III in HFrEF")
    result = engine.apply(wrong)
    assert result.validation_status == "blocked"
    assert result.label == "Class 2b"


def test_cor_class_i_roman_still_matches_class_1():
    engine = GuardrailEngine()
    engine.register(CORLevelRule())
    claim = cardiology_claim(label="ARNi", claim_text="ARNi is Class I in HFrEF")
    result = engine.apply(claim)
    assert result.validation_status == "passed"
    assert result.label == "Class 1"


# ─── 3. Audit lost on retry / shared engine audit ───────────

class _BlockBadRule(BaseRule):
    name = "test.block_bad"
    description = "Blocks claims labelled 'bad'"

    def condition(self, analysis, context):
        return analysis.label == "bad"

    def action(self, analysis, context):
        analysis.validation_status = "blocked"
        return analysis, RuleResult(
            action_taken=True, action_type="block", message="bad claim", severity="error",
        )


def test_retry_keeps_every_attempt_audit():
    outputs = iter(["bad", "good"])

    def model_fn(prompt):
        return next(outputs)

    def parser_fn(raw):
        return [Analysis(label=raw, confidence="High")], raw, None

    engine = GuardrailEngine()
    engine.register(_BlockBadRule())
    pipeline = TextGuardPipeline(model_fn=model_fn, parser_fn=parser_fn, engine=engine, max_retries=1)

    result = pipeline.run("prompt")

    assert result.retry_count == 1
    assert result.status == "passed"
    assert len(result.attempt_audits) == 2
    first, second = result.attempt_audits
    assert [e.action_type for e in first.entries] == ["block"]
    assert second.entries == []
    assert result.audit is second


def test_apply_with_audit_returns_independent_trails():
    engine = GuardrailEngine()
    engine.register(_BlockBadRule())

    _, audit1 = engine.apply_with_audit(Analysis(label="bad", confidence="High"))
    _, audit2 = engine.apply_with_audit(Analysis(label="good", confidence="High"))

    assert audit1 is not audit2
    assert len(audit1.entries) == 1
    assert len(audit2.entries) == 0


def test_apply_to_claims_returns_independent_trails():
    engine = GuardrailEngine()
    engine.register(_BlockBadRule())

    _, _, audit1 = engine.apply_to_claims([Analysis(label="bad", confidence="High")], "a")
    _, _, audit2 = engine.apply_to_claims([Analysis(label="good", confidence="High")], "a")

    assert audit1 is not audit2
    assert len(audit1.entries) == 1
    assert len(audit2.entries) == 0


def test_apply_and_apply_with_audit_consistent():
    engine = GuardrailEngine()
    engine.register(_BlockBadRule())

    engine.apply(Analysis(label="bad", confidence="High"))
    engine.apply(Analysis(label="bad", confidence="High"))
    # apply() is a fresh run each time, like apply_with_audit(): no accumulation.
    assert len(engine.audit.entries) == 1

    plain = engine.apply(Analysis(label="bad", confidence="High"))
    audited, audit = engine.apply_with_audit(Analysis(label="bad", confidence="High"))
    assert plain.model_dump() == audited.model_dump()
    assert len(audit.entries) == 1


# ─── 4. Nested JSON parsing ──────────────────────────────────

NESTED = '{"label":"Malaria","metadata":{"species":"P. falciparum"},"confidence":"High"}'


def test_parse_nested_json_object():
    analysis, _ = parse_to_analysis(NESTED)
    assert analysis.label == "Malaria"
    assert analysis.confidence == "High"
    assert analysis.metadata == {"species": "P. falciparum"}


def test_extract_nested_json_object():
    result = extract_json(f"Here you go:\n```json\n{NESTED}\n```\nThanks")
    assert result == {
        "label": "Malaria",
        "metadata": {"species": "P. falciparum"},
        "confidence": "High",
    }


def test_parse_json_with_braces_in_prose():
    text = 'Note {not json}. Result: {"label": "Malaria", "confidence": "Medium", "metadata": {"a": {"b": 1}}}'
    analysis, _ = parse_to_analysis(text)
    assert analysis.label == "Malaria"
    assert analysis.metadata == {"a": {"b": 1}}


# ─── 5. Caller context mutation ──────────────────────────────

def test_pipeline_does_not_mutate_caller_context():
    seen = {}

    class CaptureRule(BaseRule):
        name = "test.capture"
        description = "Captures context"

        def condition(self, analysis, context):
            seen.update(context)
            return False

        def action(self, analysis, context):
            return analysis, RuleResult()

    engine = GuardrailEngine()
    engine.register(CaptureRule())
    pipeline = VLMGuardPipeline(
        model_fn=lambda image, prompt: '{"label": "x", "confidence": "High"}',
        parser_fn=parse_to_analysis,
        guardrail_engine=engine,
        enhancer_fn=lambda img: img,
    )

    ctx = {"sample_type": "Blood"}
    pipeline.run(Image.new("RGB", (4, 4)), "prompt", context=ctx)

    assert ctx == {"sample_type": "Blood"}
    assert seen.get("image_enhanced") is True


# ─── 6. Core/plugin coupling ─────────────────────────────────

def _run_isolated(code: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)


def test_core_import_does_not_load_ntd_plugin():
    proc = _run_isolated(
        "import sys, vlm_guard\n"
        "assert 'vlm_guard.plugins.ntd_microscopy' not in sys.modules, 'plugin imported by core'\n"
    )
    assert proc.returncode == 0, proc.stderr


def test_ntd_report_imports_without_fpdf():
    proc = _run_isolated(
        "import sys\n"
        "sys.modules['fpdf'] = None  # simulate fpdf not installed\n"
        "import vlm_guard.plugins.ntd_microscopy.report as r\n"
        "assert callable(r.create_ntd_pdf)\n"
    )
    assert proc.returncode == 0, proc.stderr
