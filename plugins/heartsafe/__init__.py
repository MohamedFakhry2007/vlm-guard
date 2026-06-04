from vlm_guard.core.engine import GuardrailEngine
from plugins.heartsafe.rules import (
    CORLevelRule,
    LVEFThresholdRule,
    AnswerConsistencyRule,
)


def register_heartsafe_rules(
    engine: GuardrailEngine,
    include_cross_claim: bool = True,
):
    """Register all HeartSafe rules into the engine."""
    engine.register(CORLevelRule())
    engine.register(LVEFThresholdRule())

    if include_cross_claim:
        engine.register_cross_claim(AnswerConsistencyRule())


__all__ = ["register_heartsafe_rules"]
