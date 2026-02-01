import os
import pyfiglet

def display_banner():
    """Clears terminal and shows the Codal Bot Agent ASCII banner."""
    os.system('cls' if os.name == 'nt' else 'clear')
    banner = pyfiglet.figlet_format("Codal Bot", font="slant")
    print(banner)
    print("Coding Assistant")
    print("=" * 30)

def get_directory_choice():
    """Handles the user selection menu and path validation."""
    print("1. Bring your own directory")
    print("2. Use this repository")
    
    choice = input("\nSelect an option (1-2): ").strip()
    
    if choice == "1":
        path = input("Enter the full path to the directory: ").strip()
        if os.path.isdir(path):
            os.chdir(path)
            return os.getcwd()
        else:
            print(f"Path '{path}' not found. Defaulting to current repository.")
    
    return os.getcwd()

def run_landing_ui():
    """Orchestrator for the terminal startup UI."""
    display_banner()
    return get_directory_choice()