import os
import sys
import requests

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../backend")))
from app.config import settings

api_key = settings.GEMINI_API_KEY
print(f"[*] Testing Gemini API Key (Length: {len(api_key)})")

url = f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key}"
res = requests.get(url)

if res.status_code == 200:
    data = res.json()
    print("[+] Available Models for your API Key:")
    for m in data.get("models", []):
        name = m.get("name", "")
        methods = m.get("supportedGenerationMethods", [])
        if "generateContent" in methods:
            print(f"  • {name}")
else:
    print(f"[!] Error listing models ({res.status_code}): {res.text}")
