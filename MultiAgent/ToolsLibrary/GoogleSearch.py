# File: project_root/MultiAgent/ToolsLibrary/GoogleSearch.py

import requests
from bs4 import BeautifulSoup
import urllib.parse

from MultiAgent.MultiAgentFramework import LLMManager

def GoogleSearch(query, llm_config, llm_type):
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
    llm_manager = LLMManager(llm_config)
    extracted_content = llm_manager.call_llm({'type': llm_type}, prompt)

    return extracted_content

# Example usage:
if __name__ == "__main__":
    query = "python tutorial"
    llm_config = {
        'llm': {
            'openai': {
                'api_key': 'your_openai_api_key',
                'default_model': 'gpt-3.5-turbo'
            },
            'ollama': {
                'api_base': 'http://localhost:11434',
                'default_model': 'openhermes:latest'
            }
        }
    }
    llm_type = 'openai'  # or 'ollama'
    results = GoogleSearch(query, llm_config, llm_type)
    print(results)
