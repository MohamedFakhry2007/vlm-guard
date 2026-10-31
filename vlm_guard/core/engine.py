from vlm_guard.core.analysis import Analysis
from vlm_guard.core.rule import BaseRule, CrossClaimRule, RuleResult, CrossClaimResult
from vlm_guard.core.audit import AuditTrail


class GuardrailEngine:
    def __init__(self):
        self._rules: list[BaseRule] = []
        self._cross_claim_rules: list[CrossClaimRule] = []
        # Audit trail of the most recent run, kept for convenience. Every run
        # creates a new trail; use the trail returned by the apply_* methods.
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

    def _apply_rules(
        self, analysis: Analysis, context: dict, audit: AuditTrail
    ) -> Analysis:
        """Run every claim-level rule once, recording into ``audit``.

        A rule may move the status between non-passing states, but once the
        status has left "pending" it is never downgraded to "passed".
        """
        result = analysis
        for rule in self._rules:
            if rule.condition(result, context):
                prior_status = result.validation_status
                before = result.model_dump()
                result, rule_result = rule.action(result, context)
                if (
                    prior_status not in ("pending", "passed")
                    and result.validation_status == "passed"
                ):
                    result.validation_status = prior_status
                if rule_result.action_taken:
                    after = result.model_dump()
                    audit.record(rule, before, after, rule_result, context)
                elif result.validation_status == "pending":
                    result.validation_status = "passed"
            # If no rule fired, default to pending
            else:
                if result.validation_status == "pending":
                    result.validation_status = "passed"
        return result

    def apply(self, analysis: Analysis, context: dict | None = None) -> Analysis:
        """Validate a single analysis. Equivalent to ``apply_with_audit(...)[0]``.

        The audit trail of the run is available as ``engine.audit`` until the
        next run replaces it.
        """
        result, _ = self.apply_with_audit(analysis, context)
        return result

    def apply_with_audit(
        self, analysis: Analysis, context: dict | None = None
    ) -> tuple[Analysis, AuditTrail]:
        """Validate a single analysis and return it with a fresh audit trail.

        Each call gets its own ``AuditTrail``; trails returned by earlier calls
        are never modified afterwards.
        """
        audit = AuditTrail()
        result = self._apply_rules(analysis, context or {}, audit)
        self.audit = audit
        return result, audit

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
        audit = AuditTrail()
        context = context or {}

        # Phase 1: Per-claim validation
        modified_claims = []
        for i, claim in enumerate(claims):
            claim_context = {**context, "claim_index": i, "total_claims": len(claims)}
            modified = self._apply_rules(claim, claim_context, audit)
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
                    audit.record_cross_claim(
                        rule, before_claims, after_claims,
                        before_answer, final_answer, result
                    )

        self.audit = audit
        return modified_claims, final_answer, audit
