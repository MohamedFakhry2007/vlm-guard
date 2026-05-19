from vlm_guard.core.analysis import Analysis
from vlm_guard.core.rule import BaseRule, RuleResult
from vlm_guard.core.audit import AuditTrail


class GuardrailEngine:
    def __init__(self):
        self._rules: list[BaseRule] = []
        self.audit = AuditTrail()

    def register(self, rule: BaseRule):
        self._rules.append(rule)

    def unregister(self, rule_name: str) -> bool:
        before = len(self._rules)
        self._rules = [r for r in self._rules if r.name != rule_name]
        return len(self._rules) < before

    def register_list(self, rules: list[BaseRule]):
        for r in rules:
            self.register(r)

    @property
    def rules(self) -> list[BaseRule]:
        return list(self._rules)

    def apply(self, analysis: Analysis, context: dict | None = None) -> Analysis:
        context = context or {}
        result = analysis
        for rule in self._rules:
            if rule.condition(result, context):
                before = result.model_dump()
                result, rule_result = rule.action(result, context)
                if rule_result.action_taken:
                    after = result.model_dump()
                    self.audit.record(rule, before, after, rule_result)
        return result

    def apply_with_audit(
        self, analysis: Analysis, context: dict | None = None
    ) -> tuple[Analysis, AuditTrail]:
        self.audit.clear()
        result = self.apply(analysis, context)
        return result, self.audit
