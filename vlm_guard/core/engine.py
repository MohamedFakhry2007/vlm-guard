from vlm_guard.core.analysis import Analysis
from vlm_guard.core.rule import BaseRule, CrossClaimRule, RuleResult, CrossClaimResult
from vlm_guard.core.audit import AuditTrail


class GuardrailEngine:
    def __init__(self):
        self._rules: list[BaseRule] = []
        self._cross_claim_rules: list[CrossClaimRule] = []
        self.audit = AuditTrail()

    def register(self, rule: BaseRule):
        self._rules.append(rule)

    def register_cross_claim(self, rule: CrossClaimRule):
        """Register a cross-claim rule. Runs after all claim-level rules."""
        self._cross_claim_rules.append(rule)

    def unregister(self, rule_name: str) -> bool:
        before = len(self._rules)
        self._rules = [r for r in self._rules if r.name != rule_name]
        self._cross_claim_rules = [r for r in self._cross_claim_rules if r.name != rule_name]
        return (len(self._rules) + len(self._cross_claim_rules)) < before

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
                    self.audit.record(rule, before, after, rule_result, context)
                else:
                    result.validation_status = "passed"
            # If no rule fired, default to pending
            else:
                if result.validation_status == "pending":
                    result.validation_status = "passed"
        return result

    def apply_with_audit(
        self, analysis: Analysis, context: dict | None = None
    ) -> tuple[Analysis, AuditTrail]:
        self.audit.clear()
        result = self.apply(analysis, context)
        return result, self.audit

    def apply_to_claims(
        self,
        claims: list[Analysis],
        answer: str,
        context: dict | None = None,
    ) -> tuple[list[Analysis], str, AuditTrail]:
        """Validate a list of claims (reasoning steps) + the final answer.

        Phase 1: Each claim runs through all registered BaseRules independently.
        Phase 2: All claims + answer run through all registered CrossClaimRules.

        Returns (modified_claims, modified_answer, audit_trail).
        """
        self.audit.clear()
        context = context or {}

        # Phase 1: Per-claim validation
        modified_claims = []
        for i, claim in enumerate(claims):
            claim_context = {**context, "claim_index": i, "total_claims": len(claims)}
            modified = self.apply(claim, claim_context)
            modified_claims.append(modified)

        # Phase 2: Cross-claim validation
        final_answer = answer
        for rule in sorted(self._cross_claim_rules, key=lambda r: r.order):
            if rule.condition(modified_claims, final_answer, context):
                before_claims = [c.model_dump() for c in modified_claims]
                before_answer = final_answer
                modified_claims, final_answer, result = rule.action(
                    modified_claims, final_answer, context
                )
                if result.action_taken:
                    after_claims = [c.model_dump() for c in modified_claims]
                    self.audit.record_cross_claim(
                        rule, before_claims, after_claims,
                        before_answer, final_answer, result
                    )

        return modified_claims, final_answer, self.audit
