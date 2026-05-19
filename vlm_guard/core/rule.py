from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from vlm_guard.core.analysis import Analysis


@dataclass
class RuleResult:
    action_taken: bool = False
    action_type: str = "pass"
    message: str = ""
    modified_fields: dict[str, Any] | None = None


class BaseRule(ABC):
    name: str = ""
    description: str = ""

    @abstractmethod
    def condition(self, analysis: Analysis, context: dict) -> bool:
        ...

    @abstractmethod
    def action(self, analysis: Analysis, context: dict) -> tuple[Analysis, RuleResult]:
        ...

    @property
    def metadata(self) -> dict:
        return {"name": self.name, "description": self.description}

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}: {self.name}>"
