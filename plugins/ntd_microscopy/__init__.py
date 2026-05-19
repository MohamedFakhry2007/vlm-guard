from vlm_guard.core.engine import GuardrailEngine

from plugins.ntd_microscopy.rules import (
    BloodSmearAmbiguityRule,
    ThickSmearRule,
    SampleTypeImpossibilityRule,
    LeishmaniasisPromotionRule,
    MalariaValidationRule,
    TrypanosomiasisValidationRule,
    FilariasisValidationRule,
    SchistosomiasisValidationRule,
    OnchoLoaValidationRule,
    UnclearPromotionRule,
    NegativeValidationRule,
    SizeAwareRule,
)


def register_ntd_rules(engine: GuardrailEngine):
    engine.register_list([
        BloodSmearAmbiguityRule(),
        ThickSmearRule(),
        SampleTypeImpossibilityRule(),
        LeishmaniasisPromotionRule(),
        MalariaValidationRule(),
        TrypanosomiasisValidationRule(),
        FilariasisValidationRule(),
        SchistosomiasisValidationRule(),
        OnchoLoaValidationRule(),
        UnclearPromotionRule(),
        NegativeValidationRule(),
        SizeAwareRule(),
    ])


__all__ = ["register_ntd_rules"]
