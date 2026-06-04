import datetime
from dataclasses import dataclass, field
from typing import Any, Optional, Literal

from vlm_guard.core.rule import BaseRule, RuleResult, CrossClaimRule, CrossClaimResult


@dataclass
class AuditEntry:
    rule_name: str
    rule_description: str
    timestamp: str
    action_type: str
    message: str
    before: dict[str, Any] | None = None
    after: dict[str, Any] | None = None
    rule_type: Literal["claim", "cross_claim"] = "claim"
    severity: str = "info"
    domain: str = "generic"
    context: dict[str, Any] = field(default_factory=dict)
    modified_fields: dict[str, Any] | None = None
    claim_index: int | None = None


@dataclass
class AuditTrail:
    entries: list[AuditEntry] = field(default_factory=list)

    def record(
        self,
        rule: BaseRule,
        before: dict[str, Any],
        after: dict[str, Any],
        result: RuleResult,
        context: dict | None = None,
        claim_index: int | None = None,
    ):
        self.entries.append(AuditEntry(
            rule_name=rule.name,
            rule_description=rule.description,
            rule_type="claim",
            timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            action_type=result.action_type,
            message=result.message,
            severity=result.severity,
            domain=before.get("domain", "generic") if isinstance(before, dict) else "generic",
            context=context or {},
            before=before,
            after=after,
            modified_fields=result.modified_fields,
            claim_index=claim_index,
        ))

    def record_cross_claim(
        self,
        rule: CrossClaimRule,
        before_claims: list[dict],
        after_claims: list[dict],
        before_answer: str,
        after_answer: str,
        result: CrossClaimResult,
        context: dict | None = None,
    ):
        self.entries.append(AuditEntry(
            rule_name=rule.name,
            rule_description=rule.description,
            rule_type="cross_claim",
            timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            action_type=result.action_type,
            message=result.message,
            severity=result.severity,
            domain="generic",
            context=context or {},
            before={
                "claims": before_claims,
                "answer": before_answer,
            },
            after={
                "claims": after_claims,
                "answer": after_answer,
            },
            modified_fields=result.modified_claims,
            claim_index=-1,
        ))

    def to_dict(self) -> list[dict]:
        """Return audit entries as plain dicts for API serialization."""
        return [
            {
                "rule_name": e.rule_name,
                "rule_description": e.rule_description,
                "rule_type": e.rule_type,
                "action_type": e.action_type,
                "message": e.message,
                "severity": e.severity,
                "domain": e.domain,
                "modified_fields": e.modified_fields,
                "claim_index": e.claim_index,
            }
            for e in self.entries
        ]

    def last(self) -> AuditEntry | None:
        return self.entries[-1] if self.entries else None

    def summary(self) -> list[dict]:
        return [
            {
                "rule": e.rule_name,
                "action": e.action_type,
                "message": e.message,
                "modified": e.modified_fields,
                "severity": e.severity,
            }
            for e in self.entries
            if e.action_type != "pass"
        ]

    def clear(self):
        self.entries.clear()
