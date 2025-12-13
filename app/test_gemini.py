# app/test_gemini.py
from dotenv import load_dotenv
import os
from pathlib import Path
import google.generativeai as genai

# Load .env explicitly from project root
env_path = Path.cwd() / ".env"
print("CWD:", Path.cwd())
print("ENV PATH EXISTS:", env_path.exists(), env_path)
load_dotenv(dotenv_path=env_path, override=True)

key = (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or "").strip().strip('"').strip("'")
print("Key loaded:", bool(key), f"(len={len(key)})")

if not key:
    raise SystemExit("❌ No API key found in GEMINI_API_KEY or GOOGLE_API_KEY.")

genai.configure(api_key=key)

print("\n=== AVAILABLE MODELS (filter: supports generateContent) ===")
models = list(genai.list_models())

supported = []
for m in models:
    # m.name looks like "models/gemini-1.5-flash" etc.
    methods = getattr(m, "supported_generation_methods", []) or []
    if "generateContent" in methods:
        supported.append(m)

# Print a clean list
for m in supported:
    methods = getattr(m, "supported_generation_methods", []) or []
    print(f"- {m.name} | methods={methods}")

print("\n=== QUICK TEST (try a few common ones) ===")
candidates = [
    "models/gemini-2.5-flash",
    "models/gemini-2.0-flash",
    "models/gemini-1.5-flash",
    "models/gemini-1.5-pro",
]

for name in candidates:
    try:
        resp = genai.GenerativeModel(name).generate_content("Say 'ok' in one word.")
        out = (getattr(resp, "text", "") or "").strip()
        print(f"✅ {name}: {out}")
    except Exception as e:
        print(f"❌ {name}: {e}")
