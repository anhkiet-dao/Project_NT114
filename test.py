import requests

response = requests.post(
    "http://34.29.7.198:8000/generate-proof",
    json={
        "model_hash":"abc123"
    }
)

print(response.json())