# Changelog

## v0.2.0 (2026-06-04)

### Added
- `Analysis.claim_text` \u2014 verbatim claim text from LLM reasoning step
- `Analysis.domain` \u2014 domain namespace for rule routing (e.g., "cardiology", "generic")
- `Analysis.claim_type` \u2014 typed claim classification (diagnosis, recommendation, threshold, ...)
- `Analysis.validation_status` \u2014 engine-set status after validation (pending/passed/blocked/...)
- `CrossClaimRule` base class \u2014 rules that operate on all claims + answer together
- `CrossClaimResult` dataclass
- `GuardrailEngine.register_cross_claim()` and `apply_to_claims()` \u2014 multi-claim validation
- `TextGuardPipeline` \u2014 text-only pipeline with retry loop and correction prompt builder
- `TextPipelineResult` \u2014 result type for text pipeline
- `parse_reasoning_steps()` \u2014 JSON structured output parser \u2192 list[Analysis]
- `AuditEntry.rule_type`, `.severity`, `.domain`, `.context`, `.claim_index` \u2014 richer auditing
- `AuditTrail.record_cross_claim()`, `.to_dict()` \u2014 serializable audit for UI
- `plugins/heartsafe/` \u2014 example cardiology rules: CORLevelRule, LVEFThresholdRule, AnswerConsistencyRule
- `CHANGELOG.md`

### Changed
- `RuleResult.correction_suggestion` \u2014 LLM-readable feedback for retry loop
- `RuleResult.severity` \u2014 info/warning/error for UI color coding
- `GuardrailEngine.apply()` now sets `Analysis.validation_status` on each claim
- `AuditTrail.record()` accepts optional `context` and `claim_index` params
- `Analysis` Pydantic model: all new fields have defaults for backward compatibility

### Fixed
- N/A

### Deprecated
- None

### Removed
- None

### Security
- N/A
