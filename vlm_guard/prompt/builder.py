from typing import Any


class PromptBuilder:
    def __init__(self, template: str):
        self.template = template

    def build(self, **kwargs: Any) -> str:
        return self.template.format(**kwargs)

    @classmethod
    def from_file(cls, path: str, encoding: str = "utf-8") -> "PromptBuilder":
        with open(path, encoding=encoding) as f:
            return cls(f.read())


class StepPromptBuilder:
    def __init__(self, steps: list[dict]):
        self.steps = steps

    def build(self, **kwargs: Any) -> str:
        parts = []
        for i, step in enumerate(self.steps, 1):
            title = step.get("title", f"Step {i}")
            instruction = step.get("instruction", "")
            template = step.get("template", "")
            parts.append(f"STEP {i}: {title}")
            parts.append("")
            parts.append(instruction)
            if template:
                parts.append("")
                parts.append(template.format(**kwargs))
            parts.append("")
        return "\n".join(parts).strip()
