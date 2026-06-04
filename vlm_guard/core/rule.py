from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Literal

from vlm_guard.core.analysis import Analysis


ActionType = Literal["pass", "block", "correct", "promote", "flag"]


@dataclass
class RuleResult:
    action_taken: bool = False
    action_type: ActionType = "pass"
    message: str = ""
    modified_fields: dict[str, Any] | None = None
    correction_suggestion: str | None = None
    severity: Literal["info", "warning", "error"] = "info"


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


@dataclass
class CrossClaimResult:
    action_taken: bool = False
    action_type: ActionType = "pass"
    message: str = ""
    modified_claims: list[int] | None = None
    modified_answer: str | None = None
    correction_suggestion: str | None = None
    severity: Literal["info", "warning", "error"] = "info"


class CrossClaimRule(ABC):
    """A rule that operates on the full set of claims + answer together.

    This enables cross-claim consistency checks:
    - Answer must be consistent with all claims
    - Claims must not contradict each other
    - Numerical values must be consistent (e.g., LVEF in claim 1 and claim 2)
    """

    name: str = ""
    description: str = ""
    order: int = 1000

    @abstractmethod
    def condition(
        self,
        claims: list[Analysis],
        answer: str,
        context: dict,
    ) -> bool:
        ...

    @abstractmethod
    def action(
        self,
        claims: list[Analysis],
        answer: str,
        context: dict,
    ) -> tuple[list[Analysis], str, CrossClaimResult]:
        ...

    @property
    def metadata(self) -> dict:
        return {"name": self.name, "description": self.description, "type": "cross_claim"}

    def __repr__(self) -> str:
        return f"<CrossClaimRule: {self.name}>"
