import os
from openai import AzureOpenAI
from dotenv import load_dotenv
from dashboard.llm.prompt_templates import PROMPT_TEMPLATES

# Load environment variables
load_dotenv()

AZURE_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_API_BASE = os.getenv("AZURE_OPENAI_API_BASE")
AZURE_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")
DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o")

if not all([AZURE_API_KEY, AZURE_API_BASE, AZURE_API_VERSION]):
    raise EnvironmentError("Missing required Azure OpenAI environment variables.")

client = AzureOpenAI(
    api_key=AZURE_API_KEY,
    azure_endpoint=AZURE_API_BASE,
    api_version=AZURE_API_VERSION
)

def generate_llm_report(user_role: str, grade: str, lesions: str, location: str) -> str:
    prompt_template = PROMPT_TEMPLATES.get(user_role)
    
    if not prompt_template:
        return f"⚠️ Unknown role '{user_role}'. Cannot generate report."

    if not any([grade, lesions, location]):
        return "⚠️ Insufficient data to generate a report."

    filled_prompt = prompt_template.format(
        grade=grade or "Unknown",
        lesions=lesions or "Not detected",
        location=location or "Undetermined"
    )

    try:
        response = client.chat.completions.create(
            model=DEPLOYMENT_NAME,
            messages=[
                {"role": "system", "content": "You are an expert AI medical assistant generating customized diagnostic summaries."},
                {"role": "user", "content": filled_prompt}
            ],
            temperature=0
        )

        if not response or not response.choices:
            return "❌ No response generated from the LLM."

        explanation = response.choices[0].message.content.strip()

        # Role-specific tables
        if user_role == "Patient":
            table = (
                "\n\n### 📝 Summary Table\n"
                "| Section         | Details                             |\n"
                "|-----------------|--------------------------------------|\n"
                f"| Disease Grade   | {grade or 'N/A'}                    |\n"
                f"| Visual Issues   | {lesions or 'Minimal'}              |\n"
                f"| Recommendation  | Routine checkup / follow-up advised |\n"
            )
        elif user_role == "Doctor":
            table = (
                "\n\n### 🧾 Clinical Snapshot\n"
                "| Parameter          | Observation                          |\n"
                "|--------------------|---------------------------------------|\n"
                f"| Grading Outcome    | {grade or 'N/A'}                      |\n"
                f"| Lesions Observed   | {lesions or 'None detected'}          |\n"
                f"| Affected Region    | {location or 'Central region'}        |\n"
                f"| Suggested Focus    | Evaluate for NPDR/PDR staging         |\n"
            )
        elif user_role == "Clinician":
            table = (
                "\n\n### 📊 Diagnostic Summary\n"
                "| Metric                 | Value                               |\n"
                "|------------------------|-------------------------------------|\n"
                f"| DR Classification      | {grade or 'N/A'} (ICD: H35.0*)      |\n"
                f"| Pathological Features  | {lesions or 'None'}                 |\n"
                f"| Lesion Distribution    | {location or 'Undetermined'}        |\n"
                f"| Recommended Follow-Up  | OCT + Systemic Evaluation           |\n"
            )
        else:
            table = (
                "\n\n### 📝 Summary Table\n"
                "| Section        | Details                        |\n"
                "|----------------|-------------------------------|\n"
                f"| Disease Grade  | {grade or 'N/A'}              |\n"
                f"| Lesions Found  | {lesions or 'N/A'}            |\n"
                f"| Affected Area  | {location or 'N/A'}           |\n"
            )

        return explanation + table

    except Exception as e:
        return f"❌ LLM Error: {str(e)}"