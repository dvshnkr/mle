import requests

URL = "http://localhost:3998/predict"
client = {"job": "unknown", "duration": 270, "poutcome": "failure"}
response = requests.post(URL, json=client, timeout=3).json()

print(response)
