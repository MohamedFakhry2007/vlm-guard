import datetime
from dataclasses import dataclass, field
from typing import Callable

from PIL import Image

from vlm_guard.core.analysis import Analysis
from vlm_guard.core.engine import GuardrailEngine
from vlm_guard.core.audit import AuditTrail


# Most severe first. "pending" means no rule was registered, so nothing was verified.
_STATUS_RANK = {
    "blocked": 5,
    "flagged": 4,
    "unverifiable": 3,
    "corrected": 2,
    "passed": 1,
    "pending": 0,
}

# Audit actions that change or question the output map to the status they imply.
_ACTION_STATUS = {
    "block": "blocked",
    "flag": "flagged",
    "correct": "corrected",
    "promote": "corrected",
}


def derive_status(analysis: Analysis, audit: AuditTrail) -> str:
    """Release status from verification results only.

    Takes the most severe of the analysis' ``validation_status`` and the status
    implied by each audit action, so a rule that rewrote the output without
    setting ``validation_status`` still shows up. The producer's self-reported
    confidence plays no part.
    """
    statuses = [analysis.validation_status]
    statuses += [_ACTION_STATUS[e.action_type] for e in audit.entries if e.action_type in _ACTION_STATUS]
    return max(statuses, key=lambda s: _STATUS_RANK.get(s, 0))


@dataclass
class PipelineResult:
    analysis: Analysis
    raw_output: str
    status: str
    elapsed_seconds: float
    audit: AuditTrail | None = None
    image_enhanced: bool = False
    metadata: dict = field(default_factory=dict)


class VLMGuardPipeline:
    def __init__(
        self,
        *,
        model_fn: Callable[..., str],
        parser_fn: Callable[[str], tuple[Analysis, str]],
        guardrail_engine: GuardrailEngine,
        enhancer_fn: Callable[[Image.Image], Image.Image] | None = None,
    ):
        self.model_fn = model_fn
        self.parser_fn = parser_fn
        self.guardrail_engine = guardrail_engine
        self.enhancer_fn = enhancer_fn

    def run(
        self,
        image: Image.Image,
        prompt: str,
        context: dict | None = None,
    ) -> PipelineResult:
        start = datetime.datetime.now(datetime.timezone.utc)

        image_enhanced = False
        if self.enhancer_fn:
            image = self.enhancer_fn(image)
            image_enhanced = True

        raw = self.model_fn(image, prompt)

        analysis, raw_output = self.parser_fn(raw)

        context = dict(context or {})
        if image_enhanced:
            context["image_enhanced"] = True

        final, audit = self.guardrail_engine.apply_with_audit(analysis, context)

        elapsed = (datetime.datetime.now(datetime.timezone.utc) - start).total_seconds()

        return PipelineResult(
            analysis=final,
            raw_output=raw_output,
            status=derive_status(final, audit),
            elapsed_seconds=elapsed,
            audit=audit,
            image_enhanced=image_enhanced,
            # The model's own confidence is recorded, never used as a verdict.
            metadata={"producer_confidence": final.confidence},
        )
