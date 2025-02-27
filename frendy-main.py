# File: main.py
import argparse
import yaml
import sys
from dotenv import load_dotenv
from multiagent_framework.MultiAgentFramework import MultiAgentFramework, LogLevel


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Multi-Agent Framework")
    parser.add_argument("--verbosity", choices=["user", "system", "debug"], default="user",
                        help="Set the verbosity level")
    args = parser.parse_args()

    # Map verbosity string to LogLevel enum
    verbosity_map = {
        "user": LogLevel.USER,
        "system": LogLevel.SYSTEM,
        "debug": LogLevel.DEBUG
    }

    # Set verbosity to DEBUG if running in debug mode
    if __debug__:
        print("Running in debug mode. Setting verbosity to DEBUG.")
        verbosity = LogLevel.DEBUG
    else:
        verbosity = verbosity_map[args.verbosity]

    # Load environment variables
    load_dotenv()

    # Get the singleton instance of the framework
    framework = MultiAgentFramework('./MyProject/', verbosity=verbosity)

    # Start a new conversation
    initial_prompt = input("Enter your initial prompt to start the conversation: ")
    final_result = framework.run_system(initial_prompt)

    print("\nFinal result:")
    print(yaml.dump(final_result, default_flow_style=False))


if __name__ == "__main__":
    main()
