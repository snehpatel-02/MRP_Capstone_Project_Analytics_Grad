import json
import requests
from rapidfuzz import process, fuzz

with open("faq_data.json", "r") as f:
    faq_data = json.load(f)

questions = [item["question"] for item in faq_data]


def get_faq_answer(user_input):
    result = process.extractOne(
        user_input,
        questions,
        scorer=fuzz.token_set_ratio
    )

    if not result:
        return None

    match, score, idx = result

    if score >= 90:
        return faq_data[idx]["answer"]

    return None


def ask_llm(user_input):
    prompt = f"""
You are NEXOR AI, the intelligence engine for a student project called Market Access Analytics Platform.

NEXOR AI represents a central hub that connects multiple healthcare data sources like claims, payer performance, and disease trends.
It acts as a bridge between raw data and actionable insights.

You must answer only using this project context.

Project context:
- This is a healthcare analytics dashboard developed by Sneh, Rahul and Tarun under the guidance of Dr. Srikanth Mudigonda.
- It has an Insurance Dashboard and a Pharma Dashboard.
- Insurance dashboard includes total claims, coverage percentage, denied amount, payer performance, revenue leakage, and claim trends.
- Pharma dashboard includes disease burden, opportunity matrix, and forecasted claims trends.
- The purpose is to help end users understand KPIs, trends, risks, and opportunities for decision making.

Rules:
- Do not say you are Microsoft, OpenAI, or a general AI assistant.
- Do not give generic internet-style answers.
- Keep answers short, clear, and relevant to this dashboard only.
- Keep answers under 2 to 3 lines.
- If the question is outside this project, say: "This chatbot is limited to the Market Access Analytics Platform."

User question:
{user_input}

Answer:
"""

    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "phi3",
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.2
            }
        },
        timeout=60
    )

    response.raise_for_status()
    data = response.json()
    return data.get("response", "Sorry, I could not generate a response.").strip()


def get_chatbot_response(user_input):
    faq_answer = get_faq_answer(user_input)
    if faq_answer:
        return faq_answer

    return ask_llm(user_input)