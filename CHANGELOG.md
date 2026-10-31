# Changelog

## v0.2.2 (2026-10-31)

Remaining Phase 0 items (see `docs/hybrid-verification-architecture.md`, section 2.3 E and H, and the negation note in C).

### Changed (breaking)
- HeartSafe moved from the top-level `plugins.heartsafe` package to `vlm_guard.plugins.heartsafe`. Update imports, for example `from vlm_guard.plugins.heartsafe import register_heartsafe_rules`. The wheel no longer installs a top-level `plugins` package, which could collide with other distributions.
- `PipelineResult.status` from `VLMGuardPipeline` is now derived from verification only: the most severe of the analysis' `validation_status` and the statuses implied by audit actions (`block` gives "blocked", `flag` gives "flagged", `correct`/`promote` give "corrected"). Possible values are "blocked", "flagged", "unverifiable", "corrected", "passed", and "pending" when no rules are registered. The old "ok"/"low_confidence" values, which came from the model's self-reported confidence, are gone. That confidence is kept in `PipelineResult.metadata["producer_confidence"]`.

### Fixed
- NTD microscopy rules no longer treat negated findings as evidence. "No flagellated organisms seen" no longer promotes Trypanosomiasis, and "no amastigotes identified" no longer promotes Leishmaniasis. Negated spans are removed per clause before keyword matching. Uncertainty ("cannot be excluded") is not treated as negation, and "no sheath" is still read as a morphological feature. `NegativeValidationRule` still sees the raw text. This is a stopgap until assertion status comes from a clinical NLP component.

### Added
- `vlm_guard.core.pipeline.derive_status(analysis, audit)`.

## v0.2.1 (2026-10-31)

Phase 0 correctness and decoupling fixes (see `docs/hybrid-verification-architecture.md`, section 2.3).

### Fixed
- `GuardrailEngine`: a claim whose status has left "pending" is never downgraded to "passed". Previously a later rule that matched but took no action overwrote an earlier "blocked" with "passed".
- HeartSafe `CORLevelRule`: roman numerals are normalised to arabic with word boundaries, so "Class IIa" is no longer read as "Class I" (and "Class I" now matches "Class 1").
- `parse_to_analysis` / `extract_json`: JSON objects are scanned with `json.JSONDecoder.raw_decode`, so outputs with nested objects (for example `metadata`) no longer fall back to "Unclear"/"Low".
- `VLMGuardPipeline` no longer writes `image_enhanced` into the caller's context dict.
- `TextGuardPipeline` keeps every attempt's audit trail.

### Changed
- Audit trails are per run. `apply()`, `apply_with_audit()` and `apply_to_claims()` each create a new `AuditTrail`; trails returned earlier are never modified. `apply()` is now equivalent to `apply_with_audit()[0]`. `engine.audit` still exists and points at the latest run's trail.
- `TextPipelineResult.attempt_audits` added: one `AuditTrail` per attempt, in order. `audit` remains the final attempt's trail.
- `import vlm_guard` no longer imports the NTD plugin. Use `from vlm_guard.plugins import ntd_microscopy`.
- `vlm_guard.plugins.ntd_microscopy.report` imports `fpdf` only when a PDF is generated, so the module can be imported without `fpdf` installed.

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
