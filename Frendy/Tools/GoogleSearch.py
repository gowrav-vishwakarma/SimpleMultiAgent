# File: project_root/Frendy/Tools/GoogleSearch.py
import sys
import os
import requests
from bs4 import BeautifulSoup
import urllib.parse

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

from MultiAgent.MultiAgentFramework import MultiAgentFramework, LLMManager

def GoogleSearch(query, config, llm_type):
    # Encode the query for URL
    encoded_query = urllib.parse.quote(query)
    # Search for the query on Google
    url = f"https://www.google.com/search?q={encoded_query}"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    response = requests.get(url, headers=headers)
    # Parse the HTML content
    soup = BeautifulSoup(response.text, 'html.parser')
    # Extract search result details
    organic_results = []
    for i, el in enumerate(soup.select(".g")):
        title_el = el.select_one("h3")
        link_el = el.select_one(".yuRUbf > a")
        desc_el = el.select_one(".VwiC3b")
        if title_el and link_el and desc_el:
            organic_results.append({
                "title": title_el.text,
                "link": link_el["href"],
                "description": desc_el.text,
                "rank": i + 1
            })
    # Create a prompt for the LLM to extract content
    results_text = "\n".join([
        f"{i + 1}. {result['title']} ({result['link']})"
        for i, result in enumerate(organic_results)
    ])
    prompt = f"""
    You are a web content extractor. For each of the following search results, visit the URL and extract the main textual content, summarizing it in about 2-3 sentences. Ignore any ads, navigation menus, or unrelated content. Here are the search results for the query "{query}":
    {results_text}
    Please provide the extracted content for each result in the following format:
    1. [Title]
    Summary: [2-3 sentence summary of main content]
    2. [Title]
    Summary: [2-3 sentence summary of main content]
    ... and so on for all results.
    """
    # Use the LLM to extract and summarize content
    llm_manager = LLMManager(config)
    extracted_content = llm_manager.call_llm({'type': llm_type}, prompt)
    return extracted_content

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

# Example usage:
if __name__ == "__main__":
    # This part is for testing the tool directly
    query = "python tutorial"
    framework = MultiAgentFramework(project_root + '/Frendy/')
    config = framework.config
    llm_type = 'ollama'  # or 'openai'
    results = GoogleSearch(query, config, llm_type)
    print(results)

    # This part tests the main function as it would be called by the framework
    input_data = {"query": "python tutorial"}
    results = main(input_data)
    print(results)