class TokenTracker:
    def __init__(self):
        self.total_prompt_tokens = 0
        self.total_response_tokens = 0


    def track(self, response): 
   # Tracks token consumption from a Gemini API response 
        self.total_prompt_tokens += response.usage_metadata.prompt_token_count
        self.total_response_tokens += response.usage_metadata.candidates_token_count
    def summary(self):
    # Prints current token totals.
        print(f"Total prompt tokens: {self.total_prompt_tokens}")
        print(f"Total response tokens: {self.total_response_tokens}")
        print(f"Total tokens: {self.total_prompt_tokens + self.total_response_tokens}")