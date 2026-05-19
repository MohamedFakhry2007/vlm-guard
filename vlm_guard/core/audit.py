import datetime
from dataclasses import dataclass, field
from typing import Any

from vlm_guard.core.rule import BaseRule, RuleResult


@dataclass
class AuditEntry:
    rule_name: str
    rule_description: str
    timestamp: str
    action_type: str
    message: str
    before: dict[str, Any]
    after: dict[str, Any]
    modified_fields: dict[str, Any] | None = None


@dataclass
class AuditTrail:
    entries: list[AuditEntry] = field(default_factory=list)

    def record(
        self,
        rule: BaseRule,
        before: dict[str, Any],
        after: dict[str, Any],
        result: RuleResult,
    ):
        self.entries.append(AuditEntry(
            rule_name=rule.name,
            rule_description=rule.description,
            timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            action_type=result.action_type,
            message=result.message,
            before=before,
            after=after,
            modified_fields=result.modified_fields,
        ))

    def last(self) -> AuditEntry | None:
        return self.entries[-1] if self.entries else None

    def summary(self) -> list[dict]:
        return [
            {
                "rule": e.rule_name,
                "action": e.action_type,
                "message": e.message,
                "modified": e.modified_fields,
            }
            for e in self.entries
            if e.action_type != "pass"
        ]

    def clear(self):
        self.entries.clear()
