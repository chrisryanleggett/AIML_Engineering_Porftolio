# Sample agent.py file demonstrating basic API key setup in python file
# Before running this code install dependencies: pip install -r requirements.txt

from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Get the API key from environment
api_key = os.getenv('OPENAI_API_KEY')

# Set it in environment (if needed by other libraries)
os.environ['OPENAI_API_KEY'] = api_key

# Quick validation check - ensure key exists and starts correctly
if api_key and api_key.startswith('sk-'):
    print("Success")
else:
    print("API key not found or invalid format")



    