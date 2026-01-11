# Filter warnings and load .env
import warnings
warnings.filterwarnings("ignore")
# Imports
import os
from pathlib import Path
from dotenv import load_dotenv
from utils.token_tracker import TokenTracker
from google import genai
from google.genai import types

load_dotenv(Path(__file__).parent / ".env")
api_key = os.environ.get("GEMINI_API_KEY")

client = genai.Client(api_key=api_key)
tracker = TokenTracker()
print("Connection success!")

# Create a new list to store sesion conversation history
messages = []

try:
    while True:
        user_prompt = input("Enter prompt: ")
        messages.append(types.Content(role="user", parts=[types.Part(text=user_prompt)]))

        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=messages
        )
        messages.append(types.Content(role="model", parts=[types.Part(text=response.text)]))
        tracker.track(response)
        print(f"Response: {response.text}")
        tracker.summary()
except KeyboardInterrupt:
    print("\nGoodbye!")
    tracker.summary()

