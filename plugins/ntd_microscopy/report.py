import datetime
import unicodedata

from fpdf import FPDF

from vlm_guard.core.analysis import Analysis


def _latinize(s: str) -> str:
    if s is None:
        return ""
    s = str(s).replace("\u03bc", "u")
    s = unicodedata.normalize("NFKD", s)
    return s.encode("latin-1", "replace").decode("latin-1")


def create_ntd_pdf(
    analysis: Analysis,
    sample_type: str,
    stain: str,
    magnification: str,
    patient_context: str = "",
) -> bytes:
    pdf = FPDF()
    pdf.add_page()

    # Header
    pdf.set_font("Arial", "B", 18)
    pdf.cell(0, 12, "NTD-Assist Diagnostic Report", ln=True, align="C")
    pdf.ln(5)

    pdf.set_font("Arial", "", 10)
    pdf.cell(0, 6, f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True)
    pdf.ln(3)

    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 8, "Sample Information", ln=True)
    pdf.set_font("Arial", "", 11)
    pdf.cell(0, 6, _latinize(f"Sample Type: {sample_type}"), ln=True)
    pdf.cell(0, 6, _latinize(f"Stain: {stain}"), ln=True)
    pdf.cell(0, 6, _latinize(f"Magnification: {magnification}"), ln=True)
    pdf.cell(0, 6, _latinize(f"Patient Context: {patient_context if patient_context else 'Not provided'}"), ln=True)
    pdf.ln(5)

    # Diagnosis
    pdf.set_font("Arial", "B", 14)
    if analysis.label not in ("Negative for Parasites", "Unclear"):
        pdf.set_fill_color(255, 230, 230)
    else:
        pdf.set_fill_color(230, 255, 230)
    pdf.cell(0, 10, _latinize(f"DIAGNOSIS: {analysis.label}"), ln=True, fill=True)

    pdf.set_font("Arial", "", 11)
    pdf.cell(0, 7, _latinize(f"Species: {analysis.metadata.get('species', 'Unknown')}"), ln=True)
    pdf.cell(0, 7, _latinize(f"Severity: {analysis.metadata.get('severity', 'N/A')}"), ln=True)
    pdf.cell(0, 7, _latinize(f"Confidence: {analysis.confidence}"), ln=True)
    pdf.ln(5)

    # Evidence
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 8, "Morphological Evidence", ln=True)
    pdf.set_font("Arial", "", 10)
    pdf.multi_cell(0, 6, _latinize(analysis.evidence))
    pdf.ln(3)

    # Findings
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 8, "Detailed Findings", ln=True)
    pdf.set_font("Arial", "", 10)
    pdf.multi_cell(0, 6, _latinize(analysis.findings))
    pdf.ln(3)

    # Recommendation
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 8, "Recommendation", ln=True)
    pdf.set_font("Arial", "", 10)
    pdf.multi_cell(0, 6, _latinize(analysis.recommendation))
    pdf.ln(5)

    # Disclaimer
    pdf.set_font("Arial", "I", 9)
    pdf.multi_cell(0, 5, (
        "DISCLAIMER: This AI-assisted analysis is for educational and screening purposes only. "
        "All findings must be confirmed by a qualified medical professional. "
        "Do not use as sole basis for clinical decisions."
    ))

    out = pdf.output(dest="S")
    if isinstance(out, str):
        return out.encode("latin-1", "replace")
    return bytes(out)
