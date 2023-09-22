import requests

URL = "http://localhost:8990/predict"
CL_URL = (
    "https://upload.wikimedia.org/"
    "wikipedia/commons/thumb/d/df/"
    "Smaug_par_David_Demaret.jpg/1280px-Smaug_par_David_Demaret.jpg"
)

client = {"url": CL_URL}

response = requests.post(URL, json=client, timeout=13).json()

print(response)
