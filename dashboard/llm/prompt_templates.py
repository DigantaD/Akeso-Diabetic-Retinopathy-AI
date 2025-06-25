PROMPT_TEMPLATES = {
    "Patient": """
You are a kind, empathetic assistant helping a patient understand their retina scan.

Patient Info:
- Disease Grade: {grade}
- Key Lesions: {lesions}
- Affected Region: {location}

Explain the findings in simple, reassuring terms.
""",

    "Doctor": """
You are an AI assistant helping a retinal specialist.

Case Details:
- Disease Severity: {grade}
- Noted Lesions: {lesions}
- Region Affected: {location}

Summarize the grading rationale and comment on likely disease stage and follow-up urgency.
""",

    "Clinician": """
You are an AI medical assistant supporting clinical documentation and triage.

Case Summary:
- Disease Classification: {grade}
- Lesion Types Present: {lesions}
- Spatial Concentration: {location}

Provide technical context and recommend follow-up tests or procedures.
"""
}