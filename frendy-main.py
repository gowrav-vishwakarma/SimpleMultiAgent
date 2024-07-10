# File: main.py
import yaml
from dotenv import load_dotenv
from MultiAgent.MultiAgentFramework import MultiAgentFramework

# Load environment variables
load_dotenv()

if __name__ == "__main__":
    # Get the singleton instance of the framework
    framework = MultiAgentFramework('./Frendy/', debug_mode=True)

    # Start a new conversation
    initial_prompt = input("Enter your initial prompt to start the conversation: ")
    final_result = framework.start_conversation(initial_prompt)

    print("\nFinal result:")
    print(yaml.dump(final_result, default_flow_style=False))
