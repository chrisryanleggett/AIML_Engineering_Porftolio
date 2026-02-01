import os, sys, subprocess, logging, warnings
from pathlib import Path

# --- SILENCE WARNINGS ---
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*urllib3 v2.*")

def bootstrap():
    """Installs core dependencies and the syntax linter (flake8)."""
    reqs = ["google-genai", "python-dotenv", "pyfiglet", "flake8"]
    try:
        import dotenv, pyfiglet, flake8
        from google import genai
    except ImportError:
        print("Initializing Codal Bot environment and installing dependencies...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", *reqs])

bootstrap()
from dotenv import load_dotenv
_env_path = Path(__file__).resolve().parent / "api_keys" / ".env"
load_dotenv(_env_path)
from google import genai
from google.genai import types

# Local Package Imports
from utils.logger import setup_logger
from terminalUI import run_landing_ui
from tools import (
    get_file_tree, 
    read_raw_file, 
    summarize_file_content, 
    summarize_directory_behavior,
    explain_architecture,
    web_research, 
    run_linter, 
    write_to_file
)

logger = setup_logger()

class ModularAgent:
    def __init__(self, root_dir, model_id="gemini-2.0-flash"):
        self.client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        self.history = []
        self.root_dir = root_dir
        self.model_id = model_id
        
        self.instructions = self._load_instructions()
        
        # Mapping for the manual tool execution loop
        self.available_tools = {
            "get_file_tree": get_file_tree,
            "read_raw_file": read_raw_file,
            "summarize_file_content": summarize_file_content,
            "summarize_directory_behavior": summarize_directory_behavior,
            "explain_architecture": explain_architecture,
            "web_research": web_research,
            "run_linter": run_linter,
            "write_to_file": write_to_file 
        }
        
        self.config = types.GenerateContentConfig(
            tools=list(self.available_tools.values()),
            system_instruction=self.instructions
        )

    def _load_instructions(self):
        """Reads instructions from the systemInstructions subfolder."""
        base_path = Path(__file__).parent
        file_path = base_path / "systemInstructions" / "systemInstructions.txt"
        
        if file_path.exists():
            logger.info(f"System instructions loaded.")
            return file_path.read_text().strip()
        else:
            return "You are a helpful coding assistant. Use tools to analyze the repo."

    def run(self, prompt):
        """Orchestrates the Discovery, Proposal, and Authorized Execution."""
        # 1. Add user message to history
        self.history.append(types.Content(role="user", parts=[types.Part.from_text(text=prompt)]))
        
        while True:
            try:
                # 2. Call the Model
                response = self.client.models.generate_content(
                    model=self.model_id,
                    contents=self.history,
                    config=self.config
                )
                
                if not response.candidates or not response.candidates[0].content.parts:
                    return "AI reached a conclusion without further action."

                # Add the model's response to history
                self.history.append(response.candidates[0].content)
                
                tool_requests = []
                text_content = ""

                # 3. Parse the response parts
                for part in response.candidates[0].content.parts:
                    if hasattr(part, 'thought') and part.thought:
                        print(f"\n[THOUGHT]: {part.thought}")
                    if hasattr(part, 'text') and part.text:
                        text_content += part.text
                    if hasattr(part, 'function_call') and part.function_call:
                        tool_requests.append(part)

                # 4. Handle Text-Only Proposals
                if text_content and not tool_requests:
                    return text_content

                # 5. Handle Tool Requests
                if tool_requests:
                    tool_responses = []
                    executed_write = False
                    write_result = ""

                    for part in tool_requests:
                        call = part.function_call
                        tool_func = self.available_tools.get(call.name)
                        
                        if tool_func:
                            print(f"--> [DEBUG] Executing tool: {call.name}")
                            result = tool_func(**call.args)
                            result_str = str(result)
                            
                            tool_responses.append(
                                types.Part.from_function_response(
                                    name=call.name,
                                    response={"result": result_str}
                                )
                            )

                            if call.name == "write_to_file":
                                executed_write = True
                                write_result = result_str

                    # 6. FEEDBACK & TERMINATION LOGIC
                    if tool_responses:
                        self.history.append(types.Content(role="user", parts=tool_responses))
                        
                        # --- PINPOINT FIX ---
                        # If a write succeeded or was blocked, we return to the user.
                        if executed_write:
                            if "Successfully updated" in write_result or "ERROR: User blocked" in write_result:
                                return f"AI: {write_result}"
                        
                        # IF NO SUCCESSFUL WRITE HAPPENED (e.g., Error reading file):
                        # We MUST continue the loop so the AI can use Discovery to find the path.
                        continue 

                return text_content or "Task completed."

            except Exception as e:
                return f"Agent Error: {str(e)}"

if __name__ == "__main__":
    selected_path = run_landing_ui()
    agent = ModularAgent(selected_path)
    print(f"\n--- Codal Bot Agent Active in: {selected_path} ---")

    while True:
        query = input("\nUser: ").strip()
        if not query: continue
        if query.lower() in ["exit", "quit"]: break
        print("Thinking...")
        result = agent.run(query)
        print(f"\nAI: {result}")