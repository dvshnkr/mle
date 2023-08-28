import requests

URL = "http://localhost:3998/predict"
client = {"reports": 0, "share": 0.245, "expenditure": 3.438, "owner": "yes"}
response = requests.post(URL, json=client, timeout=3).json()

print(response)
