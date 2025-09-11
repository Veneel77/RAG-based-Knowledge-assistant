import os, requests

HF_TOKEN = os.environ.get("HF_TOKEN")
model = "tiiuae/falcon-7b-instruct"   # ✅ start simple

url = f"https://api-inference.huggingface.co/models/{model}"
print("Calling:", url)

headers = {"Authorization": f"Bearer {HF_TOKEN}"}
payload = {"inputs": "Hello, what is 2+2?"}

r = requests.post(url, headers=headers, json=payload, timeout=30)
print("status:", r.status_code)
print("raw text:", r.text)

try:
    print("response:", r.json())
except Exception as e:
    print("JSON parse error:", e)
