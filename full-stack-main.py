import yaml
from dotenv import load_dotenv


from multiagent_framework.MultiAgentFramework import MultiAgentFramework

# Load environment variables
load_dotenv()

if __name__ == "__main__":
    # Load the main configuration
    with open('FullStack/config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # Create the framework
    framework = MultiAgentFramework('./FullStack/')

    # Start a new conversation
    initial_prompt = input("Enter your initial prompt to start the conversation: ")
    final_result = framework.start_conversation(initial_prompt)

    print("\nFinal result:")
    print(yaml.dump(final_result, default_flow_style=False))