from vlm_guard.core.analysis import Analysis


def test_analysis_defaults():
    a = Analysis(label="Test", confidence="High")
    assert a.label == "Test"
    assert a.confidence == "High"
    assert a.evidence == ""
    assert a.findings == ""
    assert a.recommendation == ""
    assert a.metadata == {}


def test_analysis_metadata():
    a = Analysis(label="Test", confidence="Medium", metadata={"species": "P. falciparum"})
    assert a.metadata["species"] == "P. falciparum"


def test_analysis_forbids_extra_fields():
    import pytest
    with pytest.raises(ValueError):
        Analysis(label="Test", confidence="High", nonexistent_field="should fail")


def test_analysis_dump():
    a = Analysis(label="X", confidence="Low", evidence="proof", findings="desc", recommendation="action")
    d = a.model_dump()
    assert d["label"] == "X"
    assert d["confidence"] == "Low"
