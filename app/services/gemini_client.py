import os
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise ValueError("❌ GEMINI_API_KEY missing! Add it to your .env file.")

genai.configure(api_key=API_KEY)

def run_gemini(prompt: str, model: str = "gemini-1.5-flash"):
    """
    Runs Gemini with a plain text prompt and returns the text output.
    """
    try:
        response = genai.GenerativeModel(model).generate_content(prompt)
        return response.text
    except Exception as e:
        print(f"Gemini API Error: {e}")
        return ""
