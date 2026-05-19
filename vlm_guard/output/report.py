import datetime

from vlm_guard.core.analysis import Analysis


def format_text_report(analysis: Analysis, title: str = "VLM-Guard Analysis Report") -> str:
    lines = [
        "=" * 50,
        title,
        "=" * 50,
        f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        f"Label:       {analysis.label}",
        f"Confidence:  {analysis.confidence}",
        f"Evidence:    {analysis.evidence}",
        "",
        "FINDINGS",
        "-" * 30,
        analysis.findings,
        "",
        "RECOMMENDATION",
        "-" * 30,
        analysis.recommendation,
    ]

    if analysis.metadata:
        lines.extend([
            "",
            "METADATA",
            "-" * 30,
        ])
        for k, v in analysis.metadata.items():
            lines.append(f"  {k}: {v}")

    return "\n".join(lines)
