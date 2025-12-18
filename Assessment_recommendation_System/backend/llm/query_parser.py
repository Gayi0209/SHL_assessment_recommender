import google.generativeai as genai
import os
import json

api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise RuntimeError("GOOGLE_API_KEY not set")

genai.configure(api_key=api_key)


# ✅ USE THE ONLY WORKING MODEL
model = genai.GenerativeModel("models/gemini-2.5-flash")
import re

def parse_query(query: str):
    prompt = f"""
You are an expert HR analyst.

Given a hiring query, you MUST extract:
1. Technical skills (programming languages, tools, technologies)
2. Behavioral or soft skills (communication, teamwork, collaboration, leadership)
3. Overall intent:
   - "technical" → only technical skills
   - "behavioral" → only soft skills
   - "mixed" → both technical and behavioral skills

IMPORTANT RULES:
- Return ONLY valid JSON.
- Do NOT include explanations.
- Do NOT wrap JSON in markdown fences.

EXAMPLE:
{{
  "technical_skills": ["Java"],
  "behavioral_skills": ["communication"],
  "intent": "mixed"
}}

Query:
"{query}"
"""

    response = model.generate_content(
        prompt,
        generation_config={"temperature": 0}
    )

    text = response.text.strip()

    # ✅ REMOVE ```json ``` wrappers if present
    text = re.sub(r"^```json\s*|\s*```$", "", text, flags=re.DOTALL).strip()

    try:
        return json.loads(text)
    except Exception as e:
        print("⚠️ JSON parsing failed. Raw LLM output:\n", text)
        return {
            "technical_skills": [],
            "behavioral_skills": [],
            "intent": "technical"
        }
