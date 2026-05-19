from vlm_guard.core.analysis import Analysis
from vlm_guard.core.rule import BaseRule, RuleResult


def _text(analysis: Analysis) -> str:
    return (
        analysis.findings + " " + analysis.evidence + " "
        + analysis.metadata.get("observed_organisms", "")
        + analysis.metadata.get("observed_background", "")
    ).lower()


def _sample_lower(context: dict) -> str:
    return context.get("sample_type", "").lower()


def _is_tissue(s: str) -> bool:
    return any(x in s for x in ["tissue", "biopsy", "skin snip", "bone marrow", "lymph"])


def _is_blood(s: str) -> bool:
    return any(x in s for x in ["blood", "smear"])


def _is_excreta(s: str) -> bool:
    return any(x in s for x in ["stool", "urine"])


def _is_csf(s: str) -> bool:
    return "csf" in s


# --- Shared keyword helpers (ported from original) ---

def _says_in_rbc(t: str) -> bool:
    return any(p in t for p in [
        "within the red blood", "within red blood", "inside red blood",
        "within rbc", "inside rbc", "intracellular", "within the rbc",
        "contained within", "inside the erythrocyte", "intraerythrocytic",
        "in rbc",
    ])


def _says_in_macrophage(t: str) -> bool:
    return any(p in t for p in [
        "within macrophage", "inside macrophage", "in macrophage",
        "within histiocyte", "macrophage cytoplasm", "parasitophorous",
        "intracytoplasmic", "inside large cell", "within large cell",
    ])


def _says_extracellular(t: str) -> bool:
    return any(p in t for p in [
        "extracellular", "in plasma", "free in", "between cells",
        "free-swimming", "in the plasma", "between rbc",
    ])


def _says_ring(t: str) -> bool:
    return any(p in t for p in ["ring form", "ring-form", "signet ring", "delicate ring", "rings with"])


def _says_crescent(t: str) -> bool:
    return any(p in t for p in ["crescent", "banana", "banana-shaped", "crescentic"])


def _says_schizont(t: str) -> bool:
    return "schizont" in t or "merozoite" in t


def _says_gametocyte(t: str) -> bool:
    return "gametocyte" in t


def _malaria_indicators(t: str) -> bool:
    return _says_ring(t) or _says_crescent(t) or _says_schizont(t) or _says_gametocyte(t)


def _says_amastigote(t: str) -> bool:
    return any(p in t for p in ["amastigote", "ld bod", "leishman-donovan", "oval bod"])


def _says_amastigote_size(t: str) -> bool:
    return any(p in t for p in ["2-4 μm", "2-4um", "tiny", "much smaller than rbc"])


def _leishmania_indicators(t: str) -> bool:
    return _says_amastigote(t) or (_says_in_macrophage(t) and _says_amastigote_size(t))


def _tryp_indicators(t: str) -> bool:
    return "flagell" in t or "undulating membrane" in t or "trypomastigote" in t


def _says_microfilaria(t: str) -> bool:
    return "microfilar" in t


def _says_sheathed(t: str) -> bool:
    return "sheathed" in t


def _says_unsheathed(t: str) -> bool:
    return any(p in t for p in ["unsheathed", "no sheath"])


def _says_larval(t: str) -> bool:
    return any(p in t for p in ["larva", "worm-like", "long thin"])


def _microfilaria_indicators(t: str) -> bool:
    return _says_microfilaria(t) or (_says_larval(t) and _says_extracellular(t))


def _schisto_indicators(t: str) -> bool:
    return "egg" in t and any(p in t for p in ["spine", "terminal spine", "lateral spine"])


# ---------------------------------------------------------------------------
# RULE: Blood Smear Ambiguity (microfilaria vs trypanosome)
# ---------------------------------------------------------------------------

class BloodSmearAmbiguityRule(BaseRule):
    name = "ntd.blood_smear_ambiguity"
    description = "Disambiguates microfilaria vs trypanosome in blood smears"

    def condition(self, analysis: Analysis, context: dict) -> bool:
        s = _sample_lower(context)
        return s.startswith("blood smear")

    def action(self, analysis: Analysis, context: dict) -> tuple[Analysis, RuleResult]:
        t = _text(analysis)
        if _says_microfilaria(t) or _says_sheathed(t):
            analysis.label = "Filariasis"
            analysis.metadata["species"] = "Wuchereria bancrofti"
            return analysis, RuleResult(
                action_taken=True, action_type="correct",
                message="Blood smear with microfilaria → Filariasis",
                modified_fields={"label": "Filariasis", "species": "Wuchereria bancrofti"},
            )
        if any(x in t for x in ["undulating membrane", "free flagellum", "kinetoplast"]):
            analysis.label = "Trypanosomiasis"
            return analysis, RuleResult(
                action_taken=True, action_type="correct",
                message="Blood smear with flagellate → Trypanosomiasis",
                modified_fields={"label": "Trypanosomiasis"},
            )
        analysis.confidence = "Medium"
        analysis.recommendation += (
            " Morphology is ambiguous between microfilaria and trypanosome; "
            "evaluate sheath, nuclear pattern, and tail morphology."
        )
        return analysis, RuleResult(
            action_taken=True, action_type="flag",
            message="Ambiguous morphology in blood smear",
            modified_fields={"confidence": "Medium", "recommendation": "appended"},
        )


# ---------------------------------------------------------------------------
# RULE: Thick Smear
# ---------------------------------------------------------------------------

class ThickSmearRule(BaseRule):
    name = "ntd.thick_smear"
    description = "Adjusts confidence for thick smear trypanosomiasis"

    def condition(self, analysis: Analysis, context: dict) -> bool:
        return "thick" in _sample_lower(context) and analysis.label == "Trypanosomiasis"

    def action(self, analysis: Analysis, context: dict) -> tuple[Analysis, RuleResult]:
        analysis.confidence = "Medium"
        analysis.recommendation += (
            " Thick blood smears are more commonly used for microfilariae detection; "
            "consider Filariasis if sheath or nuclear column is identified."
        )
        return analysis, RuleResult(
            action_taken=True, action_type="flag",
            message="Thick smear → lowered confidence for trypanosomiasis",
            modified_fields={"confidence": "Medium"},
        )


# ---------------------------------------------------------------------------
# RULE: Sample-Type Impossibilities
# ---------------------------------------------------------------------------

class SampleTypeImpossibilityRule(BaseRule):
    name = "ntd.sample_type_impossibility"
    description = "Catches biologically impossible sample-type combinations"

    def condition(self, analysis: Analysis, context: dict) -> bool:
        return True

    def action(self, analysis: Analysis, context: dict) -> tuple[Analysis, RuleResult]:
        t = _text(analysis)
        s = _sample_lower(context)
        label = analysis.label

        if label == "Malaria":
            if _is_tissue(s) and _says_in_rbc(t) and not _is_blood(s):
                if _says_in_macrophage(t) or _says_amastigote(t) or _leishmania_indicators(t):
                    analysis.label = "Leishmaniasis"
                    analysis.confidence = "Medium"
                    analysis.metadata["species"] = "Leishmania spp."
                    analysis.recommendation = (
                        "Tissue sample with intracellular organisms in macrophages suggests Leishmaniasis. "
                        "Confirm with PCR or culture."
                    )
                    analysis.evidence = (
                        f"Organisms observed inside macrophages in tissue section. "
                        f"Original description: {analysis.evidence}"
                    )
                    return analysis, RuleResult(
                        action_taken=True, action_type="correct",
                        message="Tissue sample + macrophage organisms → Leishmaniasis",
                        modified_fields={"label": "Leishmaniasis"},
                    )
                else:
                    analysis.label = "Unclear"
                    analysis.confidence = "Low"
                    analysis.recommendation = (
                        f"Tissue biopsy reported as Malaria with RBC findings is inconsistent. "
                        f"Malaria is diagnosed on blood smears. Re-evaluate the sample type and findings."
                    )
                    analysis.evidence = f"Sample-type mismatch: {context.get('sample_type', '')} is not appropriate for malaria diagnosis."
                    return analysis, RuleResult(
                        action_taken=True, action_type="block",
                        message="Malaria impossible on tissue sample",
                        modified_fields={"label": "Unclear"},
                    )

            if "bone marrow" in s:
                if _says_in_macrophage(t) or "macrophage" in t:
                    analysis.label = "Leishmaniasis"
                    analysis.confidence = "Medium"
                    analysis.metadata["species"] = "Leishmania donovani"
                    analysis.recommendation = (
                        "Bone marrow with intracellular organisms in macrophages is classic for Visceral Leishmaniasis. "
                        "Confirm with rK39 serology or PCR."
                    )
                    analysis.evidence = "Amastigotes identified within bone marrow macrophages."
                    return analysis, RuleResult(
                        action_taken=True, action_type="correct",
                        message="Bone marrow + macrophage → Leishmaniasis",
                        modified_fields={"label": "Leishmaniasis", "species": "Leishmania donovani"},
                    )
                if any(p in t for p in ["small", "oval", "round bodies", "clusters"]):
                    analysis.label = "Leishmaniasis"
                    analysis.confidence = "Medium"
                    analysis.metadata["species"] = "Leishmania spp."
                    analysis.recommendation = (
                        "Bone marrow aspirate with small intracellular organisms suggests Visceral Leishmaniasis. "
                        "Peripheral blood is preferred for malaria diagnosis."
                    )
                    return analysis, RuleResult(
                        action_taken=True, action_type="correct",
                        message="Bone marrow + small bodies → Leishmaniasis",
                        modified_fields={"label": "Leishmaniasis"},
                    )

        if label == "Schistosomiasis" and not _is_excreta(s) and not _schisto_indicators(t):
            analysis.label = "Unclear"
            analysis.confidence = "Low"
            analysis.recommendation = "Schistosomiasis requires eggs in urine/stool. Re-evaluate for microfilariae if worm-like."
            return analysis, RuleResult(
                action_taken=True, action_type="block",
                message="Schistosomiasis requires excreta sample",
                modified_fields={"label": "Unclear"},
            )

        if label == "Onchocerciasis" and "skin" not in s:
            analysis.label = "Unclear"
            analysis.confidence = "Low"
            analysis.recommendation = "Onchocerciasis microfilariae are in skin snips. Check sample type."
            return analysis, RuleResult(
                action_taken=True, action_type="block",
                message="Onchocerciasis requires skin snip",
                modified_fields={"label": "Unclear"},
            )

        return analysis, RuleResult()


# ---------------------------------------------------------------------------
# RULE: Leishmaniasis Promotion
# ---------------------------------------------------------------------------

class LeishmaniasisPromotionRule(BaseRule):
    name = "ntd.leishmaniasis_promotion"
    description = "Promotes to Leishmaniasis when amastigote indicators found"

    def condition(self, analysis: Analysis, context: dict) -> bool:
        return analysis.label != "Leishmaniasis" and _leishmania_indicators(_text(analysis))

    def action(self, analysis: Analysis, context: dict) -> tuple[Analysis, RuleResult]:
        s = _sample_lower(context)
        if _is_tissue(s) or _says_in_macrophage(_text(analysis)):
            species = "Leishmania donovani" if any(x in s for x in ["bone", "spleen"]) else "Leishmania spp."
            analysis.label = "Leishmaniasis"
            analysis.confidence = "Medium"
            analysis.metadata["species"] = species
            analysis.recommendation = (
                "Amastigotes in macrophages/tissue indicate Leishmaniasis. "
                "Speciate with PCR. Assess for visceral involvement if bone marrow/spleen positive."
            )
            return analysis, RuleResult(
                action_taken=True, action_type="promote",
                message="Amastigote indicators → Leishmaniasis",
                modified_fields={"label": "Leishmaniasis", "species": species},
            )
        return analysis, RuleResult()


# ---------------------------------------------------------------------------
# RULE: Malaria Validation
# ---------------------------------------------------------------------------

class MalariaValidationRule(BaseRule):
    name = "ntd.malaria_validation"
    description = "Validates malaria findings against sample type and morphology"

    def condition(self, analysis: Analysis, context: dict) -> bool:
        return analysis.label == "Malaria"

    def action(self, analysis: Analysis, context: dict) -> tuple[Analysis, RuleResult]:
        t = _text(analysis)
        s = _sample_lower(context)

        if _is_tissue(s) and not _is_blood(s):
            analysis.label = "Unclear"
            analysis.confidence = "Low"
            analysis.recommendation = f"Malaria diagnosis on {context.get('sample_type', '')} is unusual. Use peripheral blood smear."
            return analysis, RuleResult(
                action_taken=True, action_type="block",
                message="Malaria on non-blood tissue",
                modified_fields={"label": "Unclear"},
            )

        if not (_says_in_rbc(t) or _malaria_indicators(t)):
            analysis.label = "Unclear"
            analysis.confidence = "Low"
            analysis.recommendation = (
                "Malaria diagnosis requires intraerythrocytic parasites "
                "(rings, trophozoites, schizonts, or gametocytes)."
            )
            return analysis, RuleResult(
                action_taken=True, action_type="block",
                message="Malaria without RBC morphology evidence",
                modified_fields={"label": "Unclear"},
            )

        species = analysis.metadata.get("species", "Unknown")
        current_species = species if isinstance(species, str) else "Unknown"
        if current_species == "Unknown":
            if _says_multiple_rings(t) or _says_crescent(t):
                analysis.metadata["species"] = "P. falciparum"
            elif _says_schuffner(t):
                analysis.metadata["species"] = "P. vivax"
            elif _says_band_form(t):
                analysis.metadata["species"] = "P. malariae"
            if analysis.metadata.get("species", "Unknown") != "Unknown":
                return analysis, RuleResult(
                    action_taken=True, action_type="correct",
                    message=f"Inferred species: {analysis.metadata['species']}",
                    modified_fields={"species": analysis.metadata["species"]},
                )

        return analysis, RuleResult()


def _says_multiple_rings(t: str) -> bool:
    return any(p in t for p in ["multiple rings", "multiple per rbc", "appliqué"])


def _says_schuffner(t: str) -> bool:
    return "schüffner" in t or "schuffner" in t


def _says_band_form(t: str) -> bool:
    return "band form" in t


# ---------------------------------------------------------------------------
# RULE: Trypanosomiasis Validation
# ---------------------------------------------------------------------------

class TrypanosomiasisValidationRule(BaseRule):
    name = "ntd.trypanosomiasis_validation"
    description = "Validates trypanosomiasis against location and morphology"

    def condition(self, analysis: Analysis, context: dict) -> bool:
        return analysis.label == "Trypanosomiasis"

    def action(self, analysis: Analysis, context: dict) -> tuple[Analysis, RuleResult]:
        t = _text(analysis)
        s = _sample_lower(context)

        if _says_in_rbc(t) and not _says_extracellular(t):
            if _malaria_indicators(t) or _says_in_rbc(t):
                analysis.label = "Unclear"
                analysis.confidence = "Low"
                analysis.recommendation = "Trypanosomes are extracellular. Organisms inside RBCs suggest Malaria instead."
                return analysis, RuleResult(
                    action_taken=True, action_type="block",
                    message="Trypanosomiasis but organisms inside RBCs",
                    modified_fields={"label": "Unclear"},
                )

        if _says_in_macrophage(t) and not _says_extracellular(t):
            analysis.label = "Leishmaniasis"
            analysis.confidence = "Medium"
            analysis.metadata["species"] = "Leishmania spp."
            analysis.recommendation = "Organisms inside macrophages indicate Leishmaniasis, not Trypanosomiasis."
            return analysis, RuleResult(
                action_taken=True, action_type="correct",
                message="Trypanosomiasis with macrophage → Leishmaniasis",
                modified_fields={"label": "Leishmaniasis"},
            )

        if "c-shaped" in t or "c shaped" in t:
            analysis.metadata["species"] = "Trypanosoma cruzi"
        elif _is_csf(s):
            analysis.metadata["species"] = "Trypanosoma brucei"

        return analysis, RuleResult(
            action_taken=True, action_type="correct" if analysis.metadata.get("species", "Unknown") != "Unknown" else "pass",
            message="Species inferred",
            modified_fields={"species": analysis.metadata.get("species", "Unknown")} if analysis.metadata.get("species", "Unknown") != "Unknown" else None,
        )


# ---------------------------------------------------------------------------
# RULE: Filariasis Validation
# ---------------------------------------------------------------------------

class FilariasisValidationRule(BaseRule):
    name = "ntd.filariasis_validation"
    description = "Validates filariasis against location and sheath morphology"

    def condition(self, analysis: Analysis, context: dict) -> bool:
        return analysis.label == "Filariasis"

    def action(self, analysis: Analysis, context: dict) -> tuple[Analysis, RuleResult]:
        t = _text(analysis)

        if _says_in_rbc(t) or ("flagell" in t and not _says_microfilaria(t)):
            analysis.label = "Unclear"
            analysis.confidence = "Low"
            analysis.recommendation = "Microfilariae are extracellular larvae without flagella. Re-examine."
            return analysis, RuleResult(
                action_taken=True, action_type="block",
                message="Filariasis inconsistent with RBC/flagella",
                modified_fields={"label": "Unclear"},
            )

        if _says_sheathed(t):
            if _says_tail_nuclei(t):
                analysis.metadata["species"] = "Brugia malayi"
            else:
                analysis.metadata["species"] = "Wuchereria bancrofti"
            return analysis, RuleResult(
                action_taken=True, action_type="correct",
                message=f"Inferred species: {analysis.metadata['species']}",
                modified_fields={"species": analysis.metadata["species"]},
            )

        return analysis, RuleResult()


def _says_tail_nuclei(t: str) -> bool:
    return any(p in t for p in ["tail nuclei", "nuclei in tail"])


# ---------------------------------------------------------------------------
# RULE: Schistosomiasis Validation
# ---------------------------------------------------------------------------

class SchistosomiasisValidationRule(BaseRule):
    name = "ntd.schistosomiasis_validation"
    description = "Validates schistosomiasis egg morphology"

    def condition(self, analysis: Analysis, context: dict) -> bool:
        return analysis.label == "Schistosomiasis"

    def action(self, analysis: Analysis, context: dict) -> tuple[Analysis, RuleResult]:
        t = _text(analysis)

        if not _schisto_indicators(t):
            analysis.label = "Unclear"
            analysis.confidence = "Low"
            analysis.recommendation = "Schistosomiasis requires eggs with spines. Check for other helminths."
            return analysis, RuleResult(
                action_taken=True, action_type="block",
                message="Schistosomiasis without egg+spine evidence",
                modified_fields={"label": "Unclear"},
            )

        if "terminal spine" in t:
            analysis.metadata["species"] = "Schistosoma haematobium"
        elif "lateral spine" in t:
            analysis.metadata["species"] = "Schistosoma mansoni"

        return analysis, RuleResult(
            action_taken=True, action_type="correct" if analysis.metadata.get("species", "Unknown") != "Unknown" else "pass",
            message="Species inferred from spine position",
            modified_fields={"species": analysis.metadata.get("species", "Unknown")} if analysis.metadata.get("species", "Unknown") != "Unknown" else None,
        )


# ---------------------------------------------------------------------------
# RULE: Onchocerciasis / Loiasis Validation
# ---------------------------------------------------------------------------

class OnchoLoaValidationRule(BaseRule):
    name = "ntd.oncho_loa_validation"
    description = "Validates Onchocerciasis and Loiasis sheath/tail features"

    def condition(self, analysis: Analysis, context: dict) -> bool:
        return analysis.label in ("Onchocerciasis", "Loiasis")

    def action(self, analysis: Analysis, context: dict) -> tuple[Analysis, RuleResult]:
        t = _text(analysis)

        if not _microfilaria_indicators(t):
            analysis.label = "Unclear"
            analysis.confidence = "Low"
            analysis.recommendation = "Requires microfilariae. Re-examine sheath and tail."
            return analysis, RuleResult(
                action_taken=True, action_type="block",
                message="Oncho/Loa without microfilaria evidence",
                modified_fields={"label": "Unclear"},
            )

        if analysis.label == "Onchocerciasis":
            if not _says_unsheathed(t) and not _says_blunt_tail(t):
                analysis.label = "Unclear"
                analysis.confidence = "Low"
                analysis.recommendation = "Onchocerciasis microfilariae are unsheathed with blunt tail."
                return analysis, RuleResult(
                    action_taken=True, action_type="block",
                    message="Onchocerciasis requires unsheathed + blunt tail",
                    modified_fields={"label": "Unclear"},
                )

        if analysis.label == "Loiasis":
            if not _says_sheathed(t) and not _says_pointed_tail(t):
                analysis.label = "Unclear"
                analysis.confidence = "Low"
                analysis.recommendation = "Loiasis microfilariae are sheathed with pointed tail and continuous nuclei."
                return analysis, RuleResult(
                    action_taken=True, action_type="block",
                    message="Loiasis requires sheathed + pointed tail",
                    modified_fields={"label": "Unclear"},
                )

        return analysis, RuleResult()


def _says_blunt_tail(t: str) -> bool:
    return "blunt tail" in t


def _says_pointed_tail(t: str) -> bool:
    return "pointed tail" in t


# ---------------------------------------------------------------------------
# RULE: Unclear → Diagnosis Promotion
# ---------------------------------------------------------------------------

class UnclearPromotionRule(BaseRule):
    name = "ntd.unclear_promotion"
    description = "Promotes Unclear to specific diagnosis when strong indicators exist"

    def condition(self, analysis: Analysis, context: dict) -> bool:
        return analysis.label == "Unclear"

    def action(self, analysis: Analysis, context: dict) -> tuple[Analysis, RuleResult]:
        t = _text(analysis)
        s = _sample_lower(context)

        if _microfilaria_indicators(t) and _says_unsheathed(t) and _says_blunt_tail(t) and "skin" in s:
            analysis.label = "Onchocerciasis"
            analysis.confidence = "Medium"
            analysis.metadata["species"] = "Onchocerca volvulus"
            analysis.recommendation = "Unsheathed microfilariae in skin suggest Onchocerciasis."
            return analysis, RuleResult(
                action_taken=True, action_type="promote",
                message="Unclear → Onchocerciasis (unsheathed microfilariae in skin)",
                modified_fields={"label": "Onchocerciasis", "species": "Onchocerca volvulus"},
            )

        if _microfilaria_indicators(t) and _says_sheathed(t) and _says_pointed_tail(t) and _is_blood(s):
            analysis.label = "Loiasis"
            analysis.confidence = "Medium"
            analysis.metadata["species"] = "Loa loa"
            analysis.recommendation = "Sheathed microfilariae with pointed tail suggest Loiasis."
            return analysis, RuleResult(
                action_taken=True, action_type="promote",
                message="Unclear → Loiasis (sheathed microfilariae in blood)",
                modified_fields={"label": "Loiasis", "species": "Loa loa"},
            )

        if _tryp_indicators(t) and _says_extracellular(t) and not _says_in_rbc(t) and not _says_in_macrophage(t):
            if (_is_blood(s) or _is_csf(s)) and not _leishmania_indicators(t):
                species = "Trypanosoma cruzi" if ("c-shaped" in t or "c shaped" in t) else "Trypanosoma brucei"
                analysis.label = "Trypanosomiasis"
                analysis.confidence = "Medium"
                analysis.metadata["species"] = species
                analysis.recommendation = "Extracellular flagellated organisms suggest Trypanosomiasis."
                return analysis, RuleResult(
                    action_taken=True, action_type="promote",
                    message="Unclear → Trypanosomiasis (extracellular flagellate)",
                    modified_fields={"label": "Trypanosomiasis", "species": species},
                )

        if _is_tissue(s) and (_says_in_macrophage(t) or _says_amastigote(t)):
            analysis.label = "Leishmaniasis"
            analysis.confidence = "Medium"
            analysis.metadata["species"] = "Leishmania spp."
            analysis.recommendation = "Intracellular organisms in tissue macrophages suggest Leishmaniasis."
            return analysis, RuleResult(
                action_taken=True, action_type="promote",
                message="Unclear → Leishmaniasis (macrophage organisms in tissue)",
                modified_fields={"label": "Leishmaniasis"},
            )

        if _schisto_indicators(t) and _is_excreta(s):
            species = (
                "Schistosoma haematobium" if "terminal spine" in t
                else "Schistosoma mansoni" if "lateral spine" in t
                else "Schistosoma spp."
            )
            analysis.label = "Schistosomiasis"
            analysis.confidence = "Medium"
            analysis.metadata["species"] = species
            analysis.recommendation = "Eggs with spines in excreta suggest Schistosomiasis."
            return analysis, RuleResult(
                action_taken=True, action_type="promote",
                message="Unclear → Schistosomiasis (eggs with spines)",
                modified_fields={"label": "Schistosomiasis", "species": species},
            )

        if _microfilaria_indicators(t) and _says_sheathed(t) and _is_blood(s):
            species = "Brugia malayi" if _says_tail_nuclei(t) else "Wuchereria bancrofti"
            analysis.label = "Filariasis"
            analysis.confidence = "Medium"
            analysis.metadata["species"] = species
            analysis.recommendation = "Sheathed microfilariae in blood suggest lymphatic Filariasis."
            return analysis, RuleResult(
                action_taken=True, action_type="promote",
                message="Unclear → Filariasis (sheathed microfilariae in blood)",
                modified_fields={"label": "Filariasis", "species": species},
            )

        return analysis, RuleResult()


# ---------------------------------------------------------------------------
# RULE: Negative Validation
# ---------------------------------------------------------------------------

class NegativeValidationRule(BaseRule):
    name = "ntd.negative_validation"
    description = "Enforces strict criteria for Negative for Parasites"

    def condition(self, analysis: Analysis, context: dict) -> bool:
        return analysis.label == "Negative for Parasites"

    def action(self, analysis: Analysis, context: dict) -> tuple[Analysis, RuleResult]:
        t = _text(analysis)
        if analysis.confidence != "High":
            analysis.label = "Unclear"
            analysis.confidence = "Low"
            analysis.recommendation = "Cannot confidently exclude parasites. Examine additional fields."
            return analysis, RuleResult(
                action_taken=True, action_type="flag",
                message="Negative with low confidence → Unclear",
                modified_fields={"label": "Unclear", "confidence": "Low"},
            )

        if any(p in t for p in ["organism", "parasite", "seen", "observed", "identified", "present"]) and "no " not in t and "none" not in t:
            analysis.label = "Unclear"
            analysis.confidence = "Low"
            analysis.recommendation = "Findings mention structures but diagnosis is negative. Manual review required."
            return analysis, RuleResult(
                action_taken=True, action_type="flag",
                message="Negative but structures mentioned → Unclear",
                modified_fields={"label": "Unclear"},
            )

        if not any(x in t for x in ["200", "fields examined", "hpf", "systematically scanned", "high power fields"]):
            analysis.label = "Unclear"
            analysis.confidence = "Low"
            analysis.recommendation = "Negative diagnosis requires examining at least 200 HPFs. Use 'Unclear' if extent not documented."
            return analysis, RuleResult(
                action_taken=True, action_type="flag",
                message="Negative without HPF count → Unclear",
                modified_fields={"label": "Unclear"},
            )

        if not any(x in t for x in ["looked for", "searched for", "specifically examined",
                                      "malaria", "leishmania", "trypanosoma", "schistosoma",
                                      "filaria", "oncho", "loa"]):
            analysis.label = "Unclear"
            analysis.confidence = "Low"
            analysis.recommendation = "Negative diagnosis requires stating all supported parasites were specifically looked for."
            return analysis, RuleResult(
                action_taken=True, action_type="flag",
                message="Negative without search targets → Unclear",
                modified_fields={"label": "Unclear"},
            )

        if not any(x in t for x in ["adequate staining", "good quality", "proper focus",
                                      "well-stained", "clear visualization"]):
            analysis.label = "Unclear"
            analysis.confidence = "Low"
            analysis.recommendation = "Negative diagnosis requires confirming adequate staining and focus."
            return analysis, RuleResult(
                action_taken=True, action_type="flag",
                message="Negative without quality confirmation → Unclear",
                modified_fields={"label": "Unclear"},
            )

        return analysis, RuleResult()


# ---------------------------------------------------------------------------
# RULE: Size-Aware Guardrail
# ---------------------------------------------------------------------------

class SizeAwareRule(BaseRule):
    name = "ntd.size_aware"
    description = "Catches morphometric impossibilities"

    def condition(self, analysis: Analysis, context: dict) -> bool:
        return True

    def action(self, analysis: Analysis, context: dict) -> tuple[Analysis, RuleResult]:
        t = _text(analysis)

        if _says_in_macrophage(t) and any(p in t for p in ["rbc-sized", "same size as rbc", "7 μm", "7 um", "7um", "similar to rbc"]):
            analysis.label = "Unclear"
            analysis.confidence = "Low"
            analysis.recommendation = (
                "RBC-sized structures inside macrophages cannot be amastigotes (2-4 μm). Re-evaluate identification."
            )
            analysis.evidence = "Size mismatch: described structures are RBC-sized, but amastigotes are much smaller (2-4 μm)."
            return analysis, RuleResult(
                action_taken=True, action_type="block",
                message="Size mismatch: RBC-sized in macrophage cannot be amastigote",
                modified_fields={"label": "Unclear"},
            )

        if _schisto_indicators(t) and any(p in t for p in ["small egg", "<50 μm", "tiny egg"]):
            analysis.label = "Unclear"
            analysis.confidence = "Low"
            analysis.recommendation = "Schistosome eggs are 100-150μm. Small eggs may indicate other helminths."
            return analysis, RuleResult(
                action_taken=True, action_type="block",
                message="Size mismatch: egg too small for schistosomiasis",
                modified_fields={"label": "Unclear"},
            )

        if _microfilaria_indicators(t) and any(p in t for p in ["short larva", "<100 μm", "tiny worm"]):
            analysis.label = "Unclear"
            analysis.confidence = "Low"
            analysis.recommendation = "Microfilariae are >200μm. Re-evaluate for protozoa if smaller."
            return analysis, RuleResult(
                action_taken=True, action_type="block",
                message="Size mismatch: larva too small for microfilaria",
                modified_fields={"label": "Unclear"},
            )

        return analysis, RuleResult()
