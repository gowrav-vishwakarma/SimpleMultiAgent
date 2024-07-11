# File: main.py
# python ./main.py run ./Frendy --verbosity user
import sys
from multiagent_framework.multiagent_cli import main as cli_main

if __name__ == "__main__":
    # Pass command-line arguments to the CLI main function
    sys.exit(cli_main())