from vlm_guard import GuardrailEngine, BaseRule, RuleResult, Analysis


class _UpperCaseRule(BaseRule):
    name = "test.uppercase"
    description = "Uppercases the label"

    def condition(self, analysis, context):
        return analysis.label.islower()

    def action(self, analysis, context):
        old = analysis.label
        analysis.label = analysis.label.upper()
        return analysis, RuleResult(
            action_taken=True,
            action_type="correct",
            message=f"Uppercased label: {old} -> {analysis.label}",
            modified_fields={"label": analysis.label},
        )


class _NeverFireRule(BaseRule):
    name = "test.never"
    description = "Never fires"

    def condition(self, analysis, context):
        return False

    def action(self, analysis, context):
        return analysis, RuleResult()


def test_register_and_run_single_rule():
    engine = GuardrailEngine()
    engine.register(_UpperCaseRule())
    assert len(engine.rules) == 1

    result = engine.apply(Analysis(label="malaria", confidence="Medium"))
    assert result.label == "MALARIA"


def test_register_unregister():
    engine = GuardrailEngine()
    engine.register(_UpperCaseRule())
    assert len(engine.rules) == 1
    engine.unregister("test.uppercase")
    assert len(engine.rules) == 0


def test_register_list():
    engine = GuardrailEngine()
    engine.register_list([_UpperCaseRule(), _NeverFireRule()])
    assert len(engine.rules) == 2


def test_rule_not_fired_when_condition_false():
    engine = GuardrailEngine()
    engine.register(_NeverFireRule())

    analysis = Analysis(label="test", confidence="High")
    result = engine.apply(analysis)
    assert result.label == "test"
    assert len(engine.audit.entries) == 0


def test_apply_with_audit_trail():
    engine = GuardrailEngine()
    engine.register(_UpperCaseRule())

    analysis = Analysis(label="hello", confidence="Low")
    result, audit = engine.apply_with_audit(analysis)
    assert result.label == "HELLO"
    summary = audit.summary()
    assert len(summary) == 1
    assert summary[0]["rule"] == "test.uppercase"
    assert summary[0]["action"] == "correct"


def test_context_passed_to_rules():
    class ContextCheckRule(BaseRule):
        name = "test.context"
        description = "Checks context exists"

        def condition(self, analysis, context):
            return "phase" in context

        def action(self, analysis, context):
            analysis.metadata["phase"] = context["phase"]
            return analysis, RuleResult(action_taken=True, action_type="flag", message="Context passed")

    engine = GuardrailEngine()
    engine.register(ContextCheckRule())

    result = engine.apply(Analysis(label="x", confidence="High"), context={"phase": "test"})
    assert result.metadata["phase"] == "test"


def test_multiple_rules_chain():
    class AppendRule(BaseRule):
        def __init__(self, suffix):
            self.suffix = suffix
            self.name = f"test.append_{suffix}"
            self.description = f"Appends {suffix}"

        def condition(self, analysis, context):
            return True

        def action(self, analysis, context):
            analysis.label += self.suffix
            return analysis, RuleResult(action_taken=True, action_type="correct", message=f"Appended {self.suffix}")

    engine = GuardrailEngine()
    engine.register_list([AppendRule("!"), AppendRule("?")])

    result = engine.apply(Analysis(label="hello", confidence="Medium"))
    assert result.label == "hello!?"
