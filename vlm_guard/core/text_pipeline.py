import time
from dataclasses import dataclass, field
from typing import Any, Callable

from vlm_guard.core.analysis import Analysis
from vlm_guard.core.engine import GuardrailEngine
from vlm_guard.core.audit import AuditTrail


@dataclass
class TextPipelineResult:
    claims: list[Analysis]
    answer: str
    status: str
    elapsed_seconds: float
    audit: AuditTrail
    retry_count: int = 0
    metadata: dict = field(default_factory=dict)


class TextGuardPipeline:
    """Text-only guardrail pipeline for RAG / chat-based LLM outputs.

    Flow:
        1. Call model_fn(prompt) -> raw text
        2. Parse raw text -> claims + answer via parser_fn
        3. Apply GuardrailEngine to claims + answer
        4. If any claim is blocked and retry_fn is set:
             a. Build correction prompt
             b. Call model_fn(correction_prompt)
             c. Re-parse and re-validate (max max_retries times)
        5. Return final result
    """

    def __init__(
        self,
        *,
        model_fn: Callable[[str], str],
        parser_fn: Callable[[str], tuple[list[Analysis], str, dict | None]],
        engine: GuardrailEngine,
        max_retries: int = 1,
    ):
        self.model_fn = model_fn
        self.parser_fn = parser_fn
        self.engine = engine
        self.max_retries = max_retries

    def run(
        self,
        prompt: str,
        context: dict | None = None,
    ) -> TextPipelineResult:
        t0 = time.perf_counter()
        context = context or {}

        raw = self.model_fn(prompt)
        claims, answer, raw_dict = self.parser_fn(raw)

        claims, answer, audit = self.engine.apply_to_claims(claims, answer, context)

        retry_count = 0
        status = self._determine_status(claims, audit)

        if status == "blocked" and self.max_retries > 0:
            correction_prompt = self._build_correction_prompt(
                prompt, claims, audit, original_answer=answer
            )
            raw2 = self.model_fn(correction_prompt)
            claims2, answer2, _ = self.parser_fn(raw2)
            claims2, answer2, audit2 = self.engine.apply_to_claims(claims2, answer2, context)
            claims, answer, audit = claims2, answer2, audit2
            retry_count = 1
            status = self._determine_status(claims, audit)

        t1 = time.perf_counter()

        return TextPipelineResult(
            claims=claims,
            answer=answer,
            status=status,
            elapsed_seconds=t1 - t0,
            audit=audit,
            retry_count=retry_count,
            metadata={"raw_output_length": len(raw)},
        )

    def _determine_status(
        self,
        claims: list[Analysis],
        audit: AuditTrail,
    ) -> str:
        has_blocked = any(c.validation_status == "blocked" for c in claims)
        has_flagged = any(c.validation_status == "flagged" for c in claims)
        has_corrected = any(c.validation_status == "corrected" for c in claims)
        any_block_action = any(
            e.action_type == "block" for e in audit.entries
        )

        if has_blocked or any_block_action:
            return "blocked"
        if has_flagged:
            return "flagged"
        if has_corrected:
            return "corrected"
        return "passed"

    def _build_correction_prompt(
        self,
        original_prompt: str,
        claims: list[Analysis],
        audit: AuditTrail,
        original_answer: str,
    ) -> str:
        failed_entries = [
            e for e in audit.entries
            if e.action_type in ("block", "flag") and e.severity == "error"
        ]

        correction_sections = []
        for entry in failed_entries:
            if entry.claim_index is not None and entry.claim_index >= 0:
                claim = claims[entry.claim_index] if entry.claim_index < len(claims) else None
                if claim:
                    correction_sections.append(
                        f"REQUIRED CORRECTION (Claim {entry.claim_index + 1}):\n"
                        f"  Original claim: {claim.claim_text}\n"
                        f"  Error: {entry.message}\n"
                    )

        correction_block = "\n".join(correction_sections)

        return (
            f"{original_prompt}\n\n"
            f"IMPORTANT \u2014 The following errors were found in your previous response. "
            f"Please correct them and regenerate the full JSON output.\n\n"
            f"{correction_block}\n"
            f"Output the corrected JSON with 'reasoning_steps' and 'answer' fields."
        )
