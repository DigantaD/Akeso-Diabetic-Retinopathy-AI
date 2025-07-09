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

def generate_llm_report(user_role: str, grade=None, lesions=None, location=None) -> str:
    prompt_template = PROMPT_TEMPLATES.get(user_role)
    if not prompt_template:
        return f"⚠️ Unknown role '{user_role}'. Cannot generate report."

    # Apply graceful defaults
    grade = grade or "not available"
    lesions = lesions or "no specific lesions detected"
    location = location or "not localized"

    # Fill the prompt
    filled_prompt = prompt_template.format(
        grade=grade,
        lesions=lesions,
        location=location
    )

    try:
        response = client.chat.completions.create(
            model=DEPLOYMENT_NAME,
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert AI medical assistant generating customized diagnostic summaries based on retina scan findings."
                },
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
                f"| Disease Grade   | {grade}                              |\n"
                f"| Visual Findings | {lesions}                            |\n"
                f"| Region Affected | {location}                           |\n"
                "| Recommendation  | Routine follow-up and healthy habits|\n"
            )
        elif user_role == "Doctor":
            table = (
                "\n\n### 🧾 Clinical Snapshot\n"
                "| Parameter          | Observation                       |\n"
                "|--------------------|-----------------------------------|\n"
                f"| Grading Outcome    | {grade}                           |\n"
                f"| Lesions Observed   | {lesions}                         |\n"
                f"| Affected Region    | {location}                        |\n"
                f"| Suggested Focus    | DR staging / treatment planning   |\n"
            )
        elif user_role == "Clinician":
            table = (
                "\n\n### 📊 Diagnostic Summary\n"
                "| Metric                 | Value                            |\n"
                "|------------------------|----------------------------------|\n"
                f"| DR Classification      | {grade} (ICD: H35.0*)             |\n"
                f"| Pathological Features  | {lesions}                        |\n"
                f"| Lesion Distribution    | {location}                       |\n"
                f"| Recommended Follow-Up  | OCT + Systemic Evaluation        |\n"
            )
        else:
            table = (
                "\n\n### 📝 Summary Table\n"
                "| Section        | Details                |\n"
                "|----------------|------------------------|\n"
                f"| Disease Grade  | {grade}                |\n"
                f"| Lesions Found  | {lesions}              |\n"
                f"| Affected Area  | {location}             |\n"
            )

        return explanation + table

    except Exception as e:
        return f"❌ LLM Error: {str(e)}"