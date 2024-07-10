# File: project_root/Frendy/Tools/GoogleSearch.py

import sys
import os
import yaml
from MultiAgent.MultiAgentFramework import MultiAgentFramework
from MultiAgent.ToolsLibrary.GoogleSearch import GoogleSearch

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)


def main(input_data):
    """
    Main function to search Google and extract content, use like this {"GoogleSearch": {"query": "__query_here__"}}
    """
    framework = MultiAgentFramework(project_root + '/Frendy/')
    query = input_data.get('query')
    llm_type = 'ollama'

    if query is None:
        raise ValueError("The 'query' field is required in the input data.")

    config = framework.config

    return GoogleSearch(query, config, llm_type)