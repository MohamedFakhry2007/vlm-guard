from vlm_guard import GuardrailEngine, Analysis
from plugins.ntd_microscopy import register_ntd_rules


def make_analysis(**overrides) -> Analysis:
    defaults = dict(
        label="Unclear",
        confidence="Medium",
        evidence="",
        findings="",
        recommendation="",
    )
    return Analysis(**{**defaults, **overrides})


def _apply(analysis: Analysis, sample_type: str = "") -> Analysis:
    engine = GuardrailEngine()
    register_ntd_rules(engine)
    return engine.apply(analysis, context={"sample_type": sample_type})


# --- Blood-smear branch ---

def test_blood_smear_microfilaria_forces_filariasis():
    out = _apply(make_analysis(label="Malaria", findings="microfilaria seen"), "Blood Smear (Thin)")
    assert out.label == "Filariasis"
    assert out.metadata.get("species") == "Wuchereria bancrofti"


def test_blood_smear_undulating_membrane_forces_trypanosomiasis():
    out = _apply(make_analysis(label="Malaria", findings="undulating membrane extracellular flagellum"), "Blood Smear (Thin)")
    assert out.label == "Trypanosomiasis"


def test_blood_smear_ambiguous_appends_recommendation():
    out = _apply(make_analysis(label="Malaria", findings="ring form inside RBC"), "Blood Smear (Thin)")
    assert "ambiguous" in out.recommendation.lower()


# --- Thick smear ---

def test_thick_smear_trypanosomiasis_appends_filariasis_hint():
    out = _apply(make_analysis(label="Trypanosomiasis", confidence="High", findings="extracellular organism"), "Blood Smear (Thick)")
    assert out.confidence == "Medium"
    assert "Thick blood smears" in out.recommendation


# --- Sample-type impossibilities ---

def test_tissue_malaria_with_macrophage_becomes_leishmaniasis():
    out = _apply(make_analysis(label="Malaria", findings="parasites inside RBC inside macrophage"), "Tissue Biopsy")
    assert out.label == "Leishmaniasis"
    assert out.metadata.get("species") == "Leishmania spp."


def test_tissue_malaria_with_rbc_only_becomes_unclear():
    out = _apply(make_analysis(label="Malaria", findings="ring form inside RBC"), "Tissue Biopsy")
    assert out.label == "Unclear"
    assert "Sample-type mismatch" in out.evidence


def test_bone_marrow_malaria_with_macrophage_becomes_visceral_leishmania():
    out = _apply(make_analysis(label="Malaria", findings="organisms in macrophage"), "Bone Marrow Aspirate")
    assert out.label == "Leishmaniasis"
    assert out.metadata.get("species") == "Leishmania donovani"


def test_bone_marrow_malaria_with_small_clusters_becomes_leishmania():
    out = _apply(make_analysis(label="Malaria", findings="small oval clusters of bodies"), "Bone Marrow Aspirate")
    assert out.label == "Leishmaniasis"


def test_schistosomiasis_without_excreta_becomes_unclear():
    out = _apply(make_analysis(label="Schistosomiasis", findings="no eggs seen"), "Tissue Biopsy")
    assert out.label == "Unclear"


def test_onchocerciasis_off_skin_becomes_unclear():
    out = _apply(make_analysis(label="Onchocerciasis", findings="microfilaria"), "Stool Sample")
    assert out.label == "Unclear"


# --- Leishmaniasis promotion ---

def test_amastigote_in_bone_marrow_promotes_to_l_donovani():
    out = _apply(make_analysis(label="Unclear", findings="amastigote in macrophage"), "Bone Marrow Aspirate")
    assert out.label == "Leishmaniasis"
    assert out.metadata.get("species") == "Leishmania donovani"


# --- Malaria validation ---

def test_malaria_on_non_blood_tissue_becomes_unclear():
    out = _apply(make_analysis(label="Malaria", findings="ring form"), "Skin Snip")
    assert out.label == "Unclear"


def test_malaria_without_rbc_or_morphology_becomes_unclear():
    out = _apply(make_analysis(label="Malaria", findings=""), "Other/Unknown")
    assert out.label == "Unclear"


def test_malaria_multiple_rings_infers_p_falciparum():
    out = _apply(make_analysis(label="Malaria", findings="ring form, multiple rings inside RBC, applique"), "Blood Smear (Thin)")
    assert out.label == "Malaria"
    assert out.metadata.get("species") == "P. falciparum"


def test_malaria_schuffner_infers_p_vivax():
    out = _apply(make_analysis(label="Malaria", findings="ring form schuffner dots inside RBC"), "Blood Smear (Thin)")
    assert out.label == "Malaria"
    assert out.metadata.get("species") == "P. vivax"


def test_malaria_band_form_infers_p_malariae():
    out = _apply(make_analysis(label="Malaria", findings="ring form band form inside RBC"), "Blood Smear (Thin)")
    assert out.label == "Malaria"
    assert out.metadata.get("species") == "P. malariae"


# --- Trypanosomiasis validation ---

def test_trypanosomiasis_inside_rbc_becomes_unclear():
    out = _apply(make_analysis(label="Trypanosomiasis", findings="organism inside RBC"), "Blood Smear (Thin)")
    assert out.label == "Unclear"


def test_trypanosomiasis_inside_macrophage_becomes_leishmaniasis():
    out = _apply(make_analysis(label="Trypanosomiasis", findings="organism inside macrophage"), "Tissue Biopsy")
    assert out.label == "Leishmaniasis"


def test_trypanosomiasis_c_shaped_infers_t_cruzi():
    out = _apply(make_analysis(label="Trypanosomiasis", findings="c-shaped extracellular flagellum"), "CSF (Cerebrospinal Fluid)")
    assert out.metadata.get("species") == "Trypanosoma cruzi"


def test_trypanosomiasis_in_csf_defaults_to_t_brucei():
    out = _apply(make_analysis(label="Trypanosomiasis", findings="extracellular flagellum"), "CSF (Cerebrospinal Fluid)")
    assert out.metadata.get("species") == "Trypanosoma brucei"


# --- Filariasis validation ---

def test_filariasis_inside_rbc_becomes_unclear():
    out = _apply(make_analysis(label="Filariasis", findings="microfilaria inside RBC"), "Blood Smear (Thin)")
    assert out.label == "Unclear"


def test_filariasis_sheathed_with_tail_nuclei_infers_b_malayi():
    out = _apply(make_analysis(label="Filariasis", findings="sheathed nuclei in tail"), "Blood Smear (Thin)")
    assert out.metadata.get("species") == "Brugia malayi"


def test_filariasis_sheathed_without_tail_nuclei_infers_w_bancrofti():
    out = _apply(make_analysis(label="Filariasis", findings="sheathed microfilaria"), "Blood Smear (Thin)")
    assert out.metadata.get("species") == "Wuchereria bancrofti"


# --- Schistosomiasis species ---

def test_schistosomiasis_terminal_spine_infers_haematobium():
    out = _apply(make_analysis(label="Schistosomiasis", findings="egg with terminal spine"), "Urine Sediment")
    assert out.label == "Schistosomiasis"
    assert out.metadata.get("species") == "Schistosoma haematobium"


def test_schistosomiasis_lateral_spine_infers_mansoni():
    out = _apply(make_analysis(label="Schistosomiasis", findings="egg with lateral spine"), "Stool Sample")
    assert out.metadata.get("species") == "Schistosoma mansoni"


# --- Onchocerciasis / Loiasis ---

def test_onchocerciasis_without_unsheathed_or_blunt_becomes_unclear():
    out = _apply(make_analysis(label="Onchocerciasis", findings="microfilaria present"), "Skin Snip")
    assert out.label == "Unclear"


def test_loiasis_without_sheathed_or_pointed_becomes_unclear():
    out = _apply(make_analysis(label="Loiasis", findings="microfilaria present"), "Skin Snip")
    assert out.label == "Unclear"


# --- Unclear promotion ---

def test_unclear_with_flagellate_in_csf_promotes_to_trypanosomiasis():
    out = _apply(make_analysis(label="Unclear", findings="extracellular flagellum"), "CSF (Cerebrospinal Fluid)")
    assert out.label == "Trypanosomiasis"
    assert out.metadata.get("species") == "Trypanosoma brucei"


def test_unclear_with_tissue_macrophage_promotes_to_leishmaniasis():
    out = _apply(make_analysis(label="Unclear", findings="organisms inside macrophage"), "Tissue Biopsy")
    assert out.label == "Leishmaniasis"


def test_unclear_unsheathed_microfilaria_in_skin_promotes_to_onchocerciasis():
    out = _apply(make_analysis(label="Unclear", findings="microfilaria unsheathed blunt tail"), "Skin Snip")
    assert out.label == "Onchocerciasis"
    assert out.metadata.get("species") == "Onchocerca volvulus"


def test_unclear_sheathed_microfilaria_in_blood_promotes_to_loiasis():
    out = _apply(make_analysis(label="Unclear", findings="microfilaria sheathed pointed tail"), "Peripheral Blood")
    assert out.label == "Loiasis"
    assert out.metadata.get("species") == "Loa loa"


def test_unclear_csf_flagellate_with_leishmania_indicators_does_not_promote():
    out = _apply(make_analysis(label="Unclear", findings="extracellular flagellum amastigote present"), "CSF (Cerebrospinal Fluid)")
    assert out.label != "Trypanosomiasis"


def test_unclear_with_eggs_in_excreta_promotes_to_schistosomiasis():
    out = _apply(make_analysis(label="Unclear", findings="egg with lateral spine"), "Stool Sample")
    assert out.label == "Schistosomiasis"
    assert out.metadata.get("species") == "Schistosoma mansoni"


# --- Negative validation ---

def test_negative_with_medium_confidence_becomes_unclear():
    out = _apply(make_analysis(label="Negative for Parasites", confidence="Medium"), "Tissue Biopsy")
    assert out.label == "Unclear"


def test_negative_missing_hpf_becomes_unclear():
    out = _apply(make_analysis(label="Negative for Parasites", confidence="High", findings="looked for malaria, adequate staining confirmed"), "Tissue Biopsy")
    assert out.label == "Unclear"
    assert "HPF" in out.recommendation


def test_negative_with_all_criteria_stays_negative():
    out = _apply(make_analysis(
        label="Negative for Parasites",
        confidence="High",
        findings="200 HPFs examined; looked for malaria leishmania trypanosoma; adequate staining confirmed; proper focus achieved",
    ), "Tissue Biopsy")
    assert out.label == "Negative for Parasites"


# --- Size-aware ---

def test_rbc_sized_in_macrophage_becomes_unclear():
    out = _apply(make_analysis(label="Leishmaniasis", findings="organisms inside macrophage 7 um"), "Bone Marrow Aspirate")
    assert out.label == "Unclear"
    assert "Size mismatch" in out.evidence


def test_small_schisto_egg_becomes_unclear():
    out = _apply(make_analysis(label="Schistosomiasis", findings="egg with spine, small egg"), "Stool Sample")
    assert out.label == "Unclear"
