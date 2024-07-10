#!/usr/bin/env python3
# File: multiagent_cli.py (This will be part of your library)

import argparse
import os
import shutil
import yaml
from dotenv import load_dotenv
from MultiAgent.MultiAgentFramework import MultiAgentFramework

def create_new_project(project_name):
    template_path = os.path.join(os.path.dirname(__file__), 'templates', 'project')
    project_path = os.path.abspath(project_name)

    if os.path.exists(project_path):
        print(f"Error: The project '{project_name}' already exists.")
        return

    shutil.copytree(template_path, project_path)
    print(f"New project '{project_name}' created successfully.")

def create_new_component(project_name, component_type, component_name):
    project_path = os.path.abspath(project_name)
    if not os.path.exists(project_path):
        print(f"Error: The project '{project_name}' does not exist.")
        return

    template_path = os.path.join(os.path.dirname(__file__), 'templates', f"{component_type.lower()}.yaml")
    component_path = os.path.join(project_path, f"{component_type}s", f"{component_name}.yaml")

    if os.path.exists(component_path):
        print(f"Error: The {component_type} '{component_name}' already exists in the project.")
        return

    shutil.copy(template_path, component_path)
    print(f"New {component_type} '{component_name}' created successfully in project '{project_name}'.")

def run_conversation(project_path):
    load_dotenv()
    framework = MultiAgentFramework(project_path, debug_mode=True)
    initial_prompt = input("Enter your initial prompt to start the conversation: ")
    final_result = framework.start_conversation(initial_prompt)
    print("\nFinal result:")
    print(yaml.dump(final_result, default_flow_style=False))

def main():
    parser = argparse.ArgumentParser(description="MultiAgent Framework CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # New project command
    new_project_parser = subparsers.add_parser("new", help="Create a new project")
    new_project_parser.add_argument("project_name", help="Name of the new project")

    # New component command
    new_component_parser = subparsers.add_parser("add", help="Add a new component to an existing project")
    new_component_parser.add_argument("project_name", help="Name of the existing project")
    new_component_parser.add_argument("component_type", choices=["Agent", "Tool", "Example"], help="Type of component to add")
    new_component_parser.add_argument("component_name", help="Name of the new component")

    # Run command
    run_parser = subparsers.add_parser("run", help="Run a conversation in an existing project")
    run_parser.add_argument("project_path", help="Path to the project")

    args = parser.parse_args()

    if args.command == "new":
        create_new_project(args.project_name)
    elif args.command == "add":
        create_new_component(args.project_name, args.component_type, args.component_name)
    elif args.command == "run":
        run_conversation(args.project_path)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()