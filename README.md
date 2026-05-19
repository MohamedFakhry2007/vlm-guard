# VLM-Guard

**Medical Verification Layer for Multimodal LLMs.**

VLM-Guard is a framework for building rule-based verification layers that catch physically impossible, biologically inconsistent, or logically contradictory outputs from multimodal LLMs — *before* they reach the end user.

```python
from vlm_guard import GuardrailEngine, BaseRule, RuleResult

# 1. Define a rule
class NoMalariaInTissueRule(BaseRule):
    name = "sample_type_check"
    description = "Malaria requires blood smear, not tissue biopsy"

    def condition(self, analysis, context):
        return (
            analysis.label == "Malaria"
            and "tissue" in context.get("sample_type", "").lower()
        )

    def action(self, analysis, context):
        analysis.label = "Unclear"
        analysis.confidence = "Low"
        return analysis, RuleResult(
            action_taken=True, action_type="block",
            message="Malaria cannot be diagnosed on tissue biopsy"
        )

# 2. Register and run
engine = GuardrailEngine()
engine.register(NoMalariaInTissueRule())

final = engine.apply(analysis, context={"sample_type": "Tissue Biopsy"})
```

## Why

Multimodal LLMs hallucinate confidently. In medical AI, a "hallucination" isn't a funny caption — it's a biologically impossible diagnosis that could lead to real-world harm.

VLM-Guard gives you a **composable, auditable rule engine** to:

- ✅ **Block** impossible outputs (malaria in a tissue biopsy)
- ✅ **Correct** misclassifications (flagellate in blood → trypanosomiasis)
- ✅ **Promote** clear patterns when the LLM is uncertain
- ✅ **Flag** ambiguous findings for human review
- ✅ **Audit** every modification with before/after snapshots

## Installation

```bash
pip install vlm-guard
```

For image processing extras:
```bash
pip install vlm-guard[image]
```

## Quick Start

### 1. Define your analysis schema

```python
from vlm_guard import Analysis

result = Analysis(
    label="Malaria",
    confidence="High",
    evidence="Ring forms observed inside RBCs",
    findings="Multiple ring-stage parasites in thin blood smear",
    recommendation="Confirm species with PCR",
    metadata={"severity": "Moderate (++)", "species": "P. falciparum"}
)
```

### 2. Build rules

Rules have two methods:
- **`condition(analysis, context)`** — should this rule fire?
- **`action(analysis, context)`** — what to do when it fires

```python
from vlm_guard import BaseRule, RuleResult

class SizeCheckRule(BaseRule):
    name = "size_check"
    description = "Rejects morphometrically impossible descriptions"

    def condition(self, analysis, context):
        text = (analysis.findings + " " + analysis.evidence).lower()
        return "7 μm" in text and "macrophage" in text

    def action(self, analysis, context):
        analysis.label = "Unclear"
        analysis.confidence = "Low"
        return analysis, RuleResult(
            action_taken=True, action_type="block",
            message="RBC-sized structures in macrophage cannot be amastigotes (2-4 μm)"
        )
```

### 3. Run the guardrail engine

```python
from vlm_guard import GuardrailEngine

engine = GuardrailEngine()
engine.register(SizeCheckRule())

final, audit = engine.apply_with_audit(result, context={"sample_type": "Bone Marrow"})
print(audit.summary())
# [{"rule": "size_check", "action": "block", "message": "...", "modified": ...}]
```

### 4. End-to-end pipeline

```python
from vlm_guard import VLMGuardPipeline
from vlm_guard.llm.parsing import parse_to_analysis
from vlm_guard.image.enhance import ImageEnhancer, EnhancementStrategy

pipeline = VLMGuardPipeline(
    model_fn=my_llm_inference_fn,      # any Callable[[Image, str], str]
    parser_fn=parse_to_analysis,        # built-in JSON parser
    guardrail_engine=engine,
    enhancer_fn=ImageEnhancer(EnhancementStrategy.HIGH_CONTRAST),
)

result = pipeline.run(image, prompt, context={"sample_type": "Blood Smear"})
print(result.analysis.label)       # final label after guardrails
print(result.audit.summary())      # everything that changed
```

## Architecture

```
                     +-----------+
                     |   Image   |
                     +-----+-----+
                           |
                           v
                +----------+----------+
                |  Enhancement (opt)  |
                +----------+----------+
                           |
                           v
                +----------+----------+
                |  Multimodal LLM     |
                +----------+----------+
                           |
                           v
                +----------+----------+
                |  JSON Parser         |
                +----------+----------+
                           |
                           v
                +----------+----------+
                |  Guardrail Engine    |
                |  +-- Rule 1 (block)  |
                |  +-- Rule 2 (flag)   |
                |  +-- Rule 3 (correct)|
                |  +-- Audit Trail     |
                +----------+----------+
                           |
                           v
                +----------+----------+
                |  Final Analysis      |
                +---------------------+
```

## Plugin System

Domain-specific rule packs can be distributed as plugins:

```python
from vlm_guard import GuardrailEngine
from my_plugin import register_my_rules

engine = GuardrailEngine()
register_my_rules(engine)
```

Check the `plugins/` directory for built-in examples:

- **`ntd_microscopy`** — 12 rule classes for Neglected Tropical Disease microscopy (migrated from [NTD-Assist](https://github.com/MohamedFakhry2007/ntd-assist))

## Rules

VLM-Guard supports four rule types:

| Type | Use Case | Example |
|------|----------|---------|
| **Block** | Prevent impossible outputs | Malaria on tissue biopsy → Unclear |
| **Correct** | Fix misidentifications | Flagellate in macrophage → Leishmaniasis |
| **Promote** | Upgrade confidence when strong signal | Amastigote + tissue → Leishmaniasis |
| **Flag** | Lower confidence / append to recommendation | "Ambiguous morphology, manual review suggested" |

## Extending

VLM-Guard is model-agnostic. The pipeline accepts any `Callable[[Image, str], str]`:

- HuggingFace `transformers` models
- OpenAI / Anthropic API clients
- Local GGUF inference
- Mock models for testing

## License

MIT

## Citation

```bibtex
@software{vlmguard2026,
  title = {VLM-Guard: Verification Layer for Multimodal LLMs},
  author = {Fakhry, Mohamed},
  year = {2026},
  url = {https://github.com/MohamedFakhry2007/vlm-guard}
}
```
