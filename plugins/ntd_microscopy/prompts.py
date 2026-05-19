import re


def get_sample_specific_guidance(sample_type: str) -> str:
    s = sample_type.lower()

    if "tissue" in s or "biopsy" in s or "skin" in s:
        return """
SAMPLE-SPECIFIC GUIDANCE (Tissue/Biopsy):
- This is a TISSUE section - you will see tissue architecture, NOT free-flowing blood
- Look for: macrophages, histiocytes, inflammatory cells, tissue structure
- Parasites here are typically INSIDE tissue macrophages (Leishmaniasis) or in tissue spaces
- You should NOT see free RBCs floating as in a blood smear
- Malaria ring forms would NOT be expected in tissue sections
"""

    if "bone marrow" in s:
        return """
SAMPLE-SPECIFIC GUIDANCE (Bone Marrow):
- Bone marrow contains: hematopoietic cells, megakaryocytes, fat cells, macrophages
- Key finding: Look for macrophages containing intracellular organisms (Leishmaniasis)
- Leishmaniasis amastigotes appear as small oval bodies clustered inside large macrophages
- While RBCs are present, malaria is rarely diagnosed on marrow (use peripheral blood)
- If you see small organisms inside MACROPHAGES -> think Leishmaniasis
"""

    if "lymph" in s:
        return """
SAMPLE-SPECIFIC GUIDANCE (Lymph Node):
- Contains lymphoid tissue with macrophages in sinuses
- Leishmaniasis: look for amastigotes inside macrophages
- Trypanosomiasis: may see trypomastigotes in aspirate fluid
"""

    if "blood" in s:
        return """
SAMPLE-SPECIFIC GUIDANCE (Blood Smear):
- Background should show RBCs (pink/salmon colored biconcave discs)
- MALARIA: Look for parasites INSIDE RBCs - rings, trophozoites, schizonts, gametocytes
- TRYPANOSOMIASIS: Look for elongated flagellates BETWEEN RBCs (extracellular)
- FILARIASIS: Look for very long thin worm-like larvae in plasma
- Pay attention to RBC size (enlarged in P. vivax/ovale)
"""

    if "csf" in s or "cerebrospinal" in s:
        return """
SAMPLE-SPECIFIC GUIDANCE (CSF):
- Normally acellular or few lymphocytes
- Trypanosomiasis: may see motile trypomastigotes
- Low cellularity - careful examination needed
"""

    if "stool" in s:
        return """
SAMPLE-SPECIFIC GUIDANCE (Stool):
- Look for: helminth eggs, larvae, protozoan cysts/trophozoites
- Schistosomiasis: eggs with lateral or terminal spine
- Background: fecal debris, bacteria, food particles
"""

    if "urine" in s:
        return """
SAMPLE-SPECIFIC GUIDANCE (Urine):
- Schistosoma haematobium: eggs with terminal spine
- Background: epithelial cells, crystals, possibly RBCs
"""

    return """
SAMPLE-SPECIFIC GUIDANCE:
- Carefully examine the background to understand the sample type
- Note what cell types are visible before looking for parasites
"""


def build_ntd_prompt(
    sample_type: str,
    magnification: str,
    stain: str,
    patient_context: str = "",
) -> str:
    ctx = re.sub(r"[^\w\s.,-]", "", patient_context or "").strip()
    guidance = get_sample_specific_guidance(sample_type)

    return f"""You are an expert parasitologist. Analyze this microscopy image by FIRST describing exactly what you observe, THEN making a diagnosis.

SAMPLE INFORMATION:
- Type: {sample_type}
- Stain: {stain}
- Magnification: {magnification}
- Clinical Context: {ctx if ctx else "None provided"}

{guidance}

================================================================================
STEP 1: OBSERVATION (describe ONLY what you actually see)
================================================================================

Look at the image and describe:
1. BACKGROUND:
   - Dominant cell type(s)
   - Tissue architecture or smear quality

2. ABNORMAL STRUCTURES:
   - COUNT: approximate number per high power field (HPF)
   - DISTRIBUTION: focal vs diffuse
   - LOCATION: intracellular (specify cell) vs extracellular
   - SIZE: relative to RBC or nucleus
   - INTERNAL STRUCTURES: nucleus, kinetoplast, pigment, sheath

3. If you see NO abnormal structures, state that clearly.

DO NOT use diagnostic terminology yet. Just describe what you observe.

**MORPHOLOGY CONSTRAINTS:**
When describing morphology:
- Do NOT assume presence of flagella unless clearly visible.
- Distinguish between:
  - flagellated protozoa (undulating membrane, kinetoplast)
  - helminth larvae (microfilaria: sheath, tapered tail, nuclear column)
- If sheath or nuclear pattern cannot be confirmed, state uncertainty explicitly.

================================================================================
STEP 2: INTERPRETATION (match observations to diagnosis)
================================================================================

Based ONLY on your observations above, determine if they match any of these patterns:

MALARIA: Ring forms, trophozoites, schizonts, or crescents INSIDE RED BLOOD CELLS
- Requires: clear RBCs visible with parasites contained within them
- Ring forms: small rings with chromatin dot(s) inside RBC
- P. falciparum: thin delicate rings, often multiple per RBC, applique forms, crescent gametocytes
- P. vivax/ovale: larger rings, Schuffner's dots, enlarged/amoeboid trophozoites, round gametocytes
- P. malariae: band forms, compact schizonts

LEISHMANIASIS: Small oval amastigotes (2-4um) INSIDE MACROPHAGES
- Requires: large cells (macrophages) containing clusters of tiny oval bodies
- Each amastigote has nucleus + kinetoplast ("double dot")
- Found in tissue, bone marrow, NOT typically in peripheral blood smears

TRYPANOSOMIASIS: Elongated flagellates FREE IN PLASMA (extracellular)
- Requires: serpentine organisms BETWEEN cells, not inside them
- Has undulating membrane and free flagellum
- African (T. brucei): slender, in blood/CSF; American (T. cruzi): C-shaped, broader kinetoplast

FILARIASIS: Long thin larvae (microfilariae) FREE IN BLOOD
- Requires: very long worm-like structures (200-300um)
- May be sheathed; NO flagellum
- Wuchereria bancrofti: sheathed, no nuclei in tail tip
- Brugia malayi: sheathed, two nuclei in tail tip

SCHISTOSOMIASIS: Oval eggs with spines in urine/stool/tissue
- Requires: large eggs (100-150um) with lateral (S. mansoni) or terminal (S. haematobium) spine
- Eggs may contain miracidium; often in clusters with inflammatory response

ONCHOCERCIASIS: Unsheathed microfilariae in skin snips
- Requires: short (220-360um) unsheathed larvae in tissue fluid
- No sheath; nuclei extend to blunt tail

LOIASIS: Sheathed microfilariae in blood
- Requires: medium (230-250um) sheathed larvae
- Nuclei continuous to pointed tail

================================================================================
STEP 3: DIAGNOSIS CHECKLIST
================================================================================

Before choosing a diagnosis, confirm ALL required hallmarks:

Malaria:
- Parasite INSIDE RBC
- Ring/trophozoite/schizont/gametocyte identified

Leishmaniasis:
- Parasites INSIDE macrophages
- Size 2-4 um
- Nucleus + kinetoplast visible

Trypanosomiasis:
- Extracellular organism
- Undulating membrane
- Free flagellum

Filariasis:
- Extracellular long larvae
- Size >200um
- Sheath presence/absence

Schistosomiasis:
- Eggs with distinct spine
- Size 100-150um

Onchocerciasis:
- Unsheathed microfilariae
- In skin/tissue
- Blunt tail with nuclei

Loiasis:
- Sheathed microfilariae
- Pointed tail with continuous nuclei

If ANY box cannot be confidently checked -> choose "Unclear".

================================================================================
STEP 4: FINAL DIAGNOSIS
================================================================================

Choose ONE: Malaria, Leishmaniasis, Schistosomiasis, Filariasis, Trypanosomiasis, Onchocerciasis, Loiasis, Negative for Parasites, Unclear

RULES:
- Your diagnosis MUST be supported by your Step 1 observations
- If you described organisms inside macrophages -> cannot be Malaria
- If you described organisms inside RBCs -> cannot be Leishmaniasis or Trypanosomiasis
- If sample is tissue/biopsy and you see organisms in large cells -> likely Leishmaniasis
- If you cannot see clear parasites, choose "Unclear" (not "Negative" unless HIGH confidence)

Return as JSON:
{{
  "detected_disease": "<diagnosis>",
  "severity": "Scanty (+)|Moderate (++)|Heavy (+++)|N/A",
  "morphology_proof": "<specific features that support your diagnosis>",
  "confidence": "High|Medium|Low",
  "findings": "<comprehensive description>",
  "recommendation": "<next steps>",
  "species": "<species if identifiable, else Unknown>",
  "observed_background": "<what cells/structures form the background>",
  "observed_organisms": "<description of any organisms seen, or 'None identified'>",
  "organism_location": "<inside RBCs | inside macrophages | extracellular | none seen>"
}}"""
