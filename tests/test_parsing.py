from vlm_guard.llm.parsing import parse_to_analysis, extract_json


def test_extract_json_with_label():
    text = '{"label": "Malaria", "confidence": "High"}'
    result = extract_json(text)
    assert result is not None
    assert result["label"] == "Malaria"


def test_extract_json_with_detected_disease():
    text = '{"detected_disease": "Leishmaniasis", "findings": "amastigote"}'
    result = extract_json(text)
    assert result is not None
    assert result["detected_disease"] == "Leishmaniasis"


def test_extract_json_with_code_fence():
    text = '```json\n{"label": "Test", "confidence": "High"}\n```'
    result = extract_json(text)
    assert result is not None
    assert result["label"] == "Test"


def test_extract_json_no_match():
    text = "This is not JSON"
    result = extract_json(text)
    assert result is None


def test_parse_to_analysis_with_label():
    text = '{"label": "Malaria", "confidence": "High", "findings": "ring forms in RBC"}'
    analysis, raw = parse_to_analysis(text)
    assert analysis.label == "Malaria"
    assert analysis.confidence == "High"
    assert analysis.findings == "ring forms in RBC"


def test_parse_to_analysis_with_detected_disease():
    text = '{"detected_disease": "Leishmaniasis", "confidence": "Medium"}'
    analysis, raw = parse_to_analysis(text)
    assert analysis.label == "Leishmaniasis"
    assert analysis.confidence == "Medium"


def test_parse_to_analysis_maps_fields():
    text = '{"label": "Test", "confidence": "Low", "evidence": "proof", "findings": "desc", "recommendation": "action"}'
    analysis, raw = parse_to_analysis(text)
    assert analysis.evidence == "proof"
    assert analysis.recommendation == "action"


def test_parse_to_analysis_strips_extra_fields():
    text = '{"label": "X", "confidence": "High", "extra_field": "should be ignored", "unknown_key": true}'
    analysis, raw = parse_to_analysis(text)
    assert analysis.label == "X"
    assert "extra_field" not in analysis.model_dump()


def test_parse_to_analysis_fallback_on_no_json():
    text = "I see ring forms inside RBC"
    analysis, raw = parse_to_analysis(text)
    assert analysis.label == "Unclear"
    assert analysis.confidence == "Low"
