import requests

URL = "http://localhost:8990/2015-03-31/functions/function/invocations"

data = {
    "url": "https://upload.wikimedia.org/wikipedia/en/e/e9/GodzillaEncounterModel.jpg"
}

result = requests.post(URL, json=data, timeout=13).json()
print(result)
