import google.generativeai as genai

# Configure the API key
genai.configure(api_key="AIzaSyBJTUxty4oKoOuIXus5QeLWdZoItrrr9t4")

# List available models
for model in genai.list_models():
    if "generateContent" in model.supported_generation_methods:
        print(f"Model: {model.name}")
        print(f"  Display name: {model.display_name}")
        print(f"  Supported methods: {model.supported_generation_methods}")
        print()