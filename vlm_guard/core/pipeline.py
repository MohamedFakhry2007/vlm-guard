import datetime
from dataclasses import dataclass, field
from typing import Callable

from PIL import Image

from vlm_guard.core.analysis import Analysis
from vlm_guard.core.engine import GuardrailEngine
from vlm_guard.core.audit import AuditTrail


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

        context = context or {}
        if image_enhanced:
            context["image_enhanced"] = True

        final, audit = self.guardrail_engine.apply_with_audit(analysis, context)

        elapsed = (datetime.datetime.now(datetime.timezone.utc) - start).total_seconds()

        return PipelineResult(
            analysis=final,
            raw_output=raw_output,
            status="ok" if final.confidence != "Low" else "low_confidence",
            elapsed_seconds=elapsed,
            audit=audit,
            image_enhanced=image_enhanced,
        )
