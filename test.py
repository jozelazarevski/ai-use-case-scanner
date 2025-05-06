import google.generativeai as genai
from config import Config

GOOGLE_API_KEY = Config.GOOGLE_API_KEY
genai.configure(api_key=GOOGLE_API_KEY)

model = genai.GenerativeModel("gemini-2.0-pro-exp")
try:
    response = model.generate_content("What is the capital of Switzerland?")
    print(response.text)
except Exception as e:
    print(f"Error: {e}")
    models = genai.list_models()  # Replace with the actual function name if different
    for model_info in models:
        print(f"Model: {model_info.name}")
        print(f"  Supported Methods: {model_info.supported_generation_methods}") # Or similar attribute
        print("-" * 20)

