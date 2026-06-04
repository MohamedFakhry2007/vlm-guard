"""Integration test for vlm-guard v0.2.0 — HeartSafe scenario."""

from vlm_guard import (
    Analysis, BaseRule, RuleResult,
    CrossClaimRule, CrossClaimResult,
    GuardrailEngine, TextGuardPipeline,
    parse_reasoning_steps, AuditTrail,
)
from vlm_guard.core.rule import ActionType


# ─── 1. Test Analysis model extensions ────────────────────────

def test_analysis_extensions():
    a = Analysis(
        label="MRA is Class 1 in HFrEF",
        domain="cardiology",
        claim_type="recommendation",
        claim_text="MRA is recommended (Class 1) for HFrEF with LVEF ≤35%",
        confidence="High",
        evidence="Per AHA/ACC guideline section 5.3: MRA is Class 1...",
        metadata={"drug": "MRA", "cor": "Class 1"},
    )

    assert a.domain == "cardiology"
    assert a.claim_type == "recommendation"
    assert a.claim_text.startswith("MRA is recommended")
    assert a.validation_status == "pending"

    # Test backward compatibility
    b = Analysis(label="Malaria", confidence="Medium")
    assert b.domain == "generic"
    assert b.claim_type == "other"
    assert b.claim_text == ""
    assert b.validation_status == "pending"


# ─── 2. Test parse_reasoning_steps ──────────────────────────

def test_parse_reasoning_steps():
    llm_output = """```json
{
  "reasoning_steps": [
    {
      "step": "Patient has HFrEF with LVEF 30%",
      "claim_type": "diagnosis",
      "grounding": "Per guideline, HFrEF is LVEF ≤40%"
    },
    {
      "step": "LVEF ≤35% threshold for MRA is met",
      "claim_type": "threshold",
      "grounding": "MRA indicated for LVEF ≤35%"
    },
    {
      "step": "MRA is recommended (Class 1) for this patient",
      "claim_type": "recommendation",
      "grounding": "Section 5.3: MRA Class 1 for HFrEF with LVEF ≤35%"
    }
  ],
  "answer": "MRA is recommended (Class 1) for this patient with HFrEF and LVEF 30%."
}
```"""

    claims, answer, raw = parse_reasoning_steps(llm_output, domain="cardiology")

    assert len(claims) == 3
    assert claims[0].domain == "cardiology"
    assert claims[0].claim_type == "diagnosis"
    assert "HFrEF" in claims[0].claim_text
    assert claims[1].claim_type == "threshold"
    assert claims[2].claim_type == "recommendation"
    assert "MRA is recommended" in answer


# ─── 3. Test COR rule ──────────────────────────────────────

def test_cor_rule_block():
    from plugins.heartsafe.rules import CORLevelRule

    rule = CORLevelRule()
    engine = GuardrailEngine()
    engine.register(rule)

    claim = Analysis(
        label="MRA is Class 2a in HFrEF",
        domain="cardiology",
        claim_type="recommendation",
        claim_text="MRA has a Class 2a recommendation in HFrEF with LVEF ≤35%",
        metadata={"drug": "MRA", "condition": "HFrEF with LVEF ≤35%, NYHA II-IV, on BB+RASi"},
    )

    result = engine.apply(claim)
    assert result.validation_status == "blocked"
    assert "Class 1" in result.label


def test_cor_rule_pass():
    from plugins.heartsafe.rules import CORLevelRule

    rule = CORLevelRule()
    engine = GuardrailEngine()
    engine.register(rule)

    claim = Analysis(
        label="MRA is Class 1 in HFrEF",
        domain="cardiology",
        claim_type="recommendation",
        claim_text="MRA is recommended (Class 1) for HFrEF with LVEF ≤35%",
        metadata={"drug": "MRA"},
    )

    result = engine.apply(claim)
    assert result.validation_status == "passed"


# ─── 4. Test LVEF threshold rule ──────────────────────────

def test_lvef_correction():
    from plugins.heartsafe.rules import LVEFThresholdRule

    rule = LVEFThresholdRule()
    engine = GuardrailEngine()
    engine.register(rule)

    claim = Analysis(
        label="HFpEF",
        domain="cardiology",
        claim_type="diagnosis",
        claim_text="Patient has HFpEF with LVEF 38%",
    )

    result = engine.apply(claim)
    assert result.validation_status == "corrected"
    assert result.label == "HFrEF"


# ─── 5. Test cross-claim validation ───────────────────────

def test_answer_consistency():
    from plugins.heartsafe.rules import AnswerConsistencyRule

    rule = AnswerConsistencyRule()
    engine = GuardrailEngine()
    engine.register_cross_claim(rule)

    claims = [
        Analysis(
            label="MRA Class 1",
            domain="cardiology",
            claim_type="recommendation",
            claim_text="MRA is recommended (Class 1)",
            validation_status="passed",
        ),
    ]

    modified_claims, modified_answer, audit = engine.apply_to_claims(
        claims,
        answer="MRA is not recommended for this patient.",
        context={},
    )

    assert "Class 1" in modified_answer or "NOTE" in modified_answer


# ─── 6. Test full pipeline ────────────────────────────────

def test_text_pipeline():
    from plugins.heartsafe import register_heartsafe_rules

    engine = GuardrailEngine()
    register_heartsafe_rules(engine)

    def mock_llm(prompt: str) -> str:
        return """```json
{
  "reasoning_steps": [
    {
      "step": "Patient has HFrEF with LVEF 30%",
      "claim_type": "diagnosis",
      "grounding": "Per guideline, HFrEF is LVEF ≤40%"
    },
    {
      "step": "MRA is recommended (Class 2a) for this patient",
      "claim_type": "recommendation",
      "grounding": "Section 5.3"
    }
  ],
  "answer": "MRA is not recommended for this patient."
}
```"""

    pipeline = TextGuardPipeline(
        model_fn=mock_llm,
        parser_fn=lambda t: parse_reasoning_steps(t, domain="cardiology"),
        engine=engine,
        max_retries=1,
    )

    result = pipeline.run("What is the MRA recommendation?")

    assert len(result.claims) == 2
    assert result.elapsed_seconds > 0
    assert result.audit is not None


# ─── 7. Test audit serialization ──────────────────────────

def test_audit_to_dict():
    from plugins.heartsafe.rules import CORLevelRule

    rule = CORLevelRule()
    engine = GuardrailEngine()
    engine.register(rule)

    claim = Analysis(
        label="MRA Class 2a",
        domain="cardiology",
        claim_type="recommendation",
        claim_text="MRA is Class 2a in HFrEF",
        metadata={"drug": "MRA"},
    )

    _, audit = engine.apply_with_audit(claim)

    serialized = audit.to_dict()
    assert isinstance(serialized, list)
    if serialized:
        entry = serialized[0]
        assert "rule_name" in entry
        assert "action_type" in entry
        assert "message" in entry
        assert "severity" in entry
