# Simple Gemini Chat

A classic interactive chat interface with Google's Gemini API that tracks token consumption in real-time. This project serves as a starter building block for other Agent examples in this directory.

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Create a `.env` file with your Gemini API key:
```
GEMINI_API_KEY=your_api_key_here
```

3. Run the chat:
```bash
python main.py
```

## Details

Uses the `google-genai` library with the `gemini-2.0-flash` model for conversational interactions. The application maintains session history and displays token usage after each exchange using a custom `TokenTracker` utility. Built with a simple interactive loop approach using `python-dotenv` for environment configuration.

