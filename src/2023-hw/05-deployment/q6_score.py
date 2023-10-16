import requests

URL = "http://localhost:3998/predict"
client = {"job": "retired", "duration": 445, "poutcome": "success"}
response = requests.post(URL, json=client, timeout=3).json()

print(response)
