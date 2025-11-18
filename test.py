import requests

# URL of your running FastAPI server
url = 'http://127.0.0.1:8000/predict'

# Example texts to classify
data = {
    "texts": [
        "Hello, how are you?",
        "Bonjour, comment ça va?",
        "Hola, ¿cómo estás?",
        "Hallo, wie geht's?"
    ]
}

# Send POST request
response = requests.post(url, json=data)

# Check if request was successful
if response.status_code == 200:
    predictions = response.json().get("predictions", [])
    for text, pred in zip(data["texts"], predictions):
        print(f"'{text}' --> {pred}")
else:
    print(f"Error: {response.status_code}, {response.text}")
