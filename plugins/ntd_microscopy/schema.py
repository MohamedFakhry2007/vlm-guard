from typing import Literal

from vlm_guard.core.analysis import Analysis


NTD_DISEASES = [
    "Malaria", "Leishmaniasis", "Schistosomiasis",
    "Filariasis", "Trypanosomiasis", "Onchocerciasis",
    "Loiasis", "Negative for Parasites", "Unclear",
]

NTD_SEVERITY = ["Scanty (+)", "Moderate (++)", "Heavy (+++)", "N/A"]


def ntd_analysis_from_dict(data: dict) -> Analysis:
    allowed = set(Analysis.model_fields.keys())

    if "detected_disease" in data and "label" not in data:
        data["label"] = data["detected_disease"]

    ntd_meta = {}
    for field in ("severity", "species", "observed_background", "observed_organisms", "organism_location"):
        if field in data:
            ntd_meta[field] = data.pop(field)

    filtered = {k: v for k, v in data.items() if k in allowed}

    analysis = Analysis(**filtered)
    analysis.metadata.update(ntd_meta)
    return analysis
