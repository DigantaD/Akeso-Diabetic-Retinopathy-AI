import os
from openai import AzureOpenAI
from dotenv import load_dotenv
from dashboard.llm.prompt_templates import PROMPT_TEMPLATES

# Load environment variables first
load_dotenv()

# Safely fetch required credentials
AZURE_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_API_BASE = os.getenv("AZURE_OPENAI_API_BASE")
AZURE_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")
DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o")

# Sanity check for environment setup
if not all([AZURE_API_KEY, AZURE_API_BASE, AZURE_API_VERSION]):
    raise EnvironmentError("Missing required Azure OpenAI environment variables. Please check your .env file.")

# Initialize Azure OpenAI client
client = AzureOpenAI(
    api_key=AZURE_API_KEY,
    azure_endpoint=AZURE_API_BASE,
    api_version=AZURE_API_VERSION
)

def generate_llm_report(user_role: str, grade: str, lesions: str, location: str) -> str:
    prompt_template = PROMPT_TEMPLATES.get(user_role)
    if not prompt_template:
        return f"⚠️ Unknown role '{user_role}' for report generation."

    filled_prompt = prompt_template.format(
        grade=grade,
        lesions=lesions,
        location=location
    )

    try:
        response = client.chat.completions.create(
            model=DEPLOYMENT_NAME,
            messages=[
                {"role": "system", "content": "You are an expert AI medical assistant."},
                {"role": "user", "content": filled_prompt}
            ],
            temperature=0.3,
            max_tokens=250
        )

        if response and response.choices:
            return response.choices[0].message.content.strip()
        else:
            return "❌ LLM returned empty response."

    except Exception as e:
        return f"❌ LLM error: {str(e)}"