# File: project_root/Frendy/Tools/GoogleSearch.py
import sys
import os
import logging
from googlesearch import search
import requests
from bs4 import BeautifulSoup
import html2text

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)
from MultiAgent.MultiAgentFramework import MultiAgentFramework, LLMManager

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def extract_content(url):
    try:
        response = requests.get(url, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')

        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()

        # Convert HTML to plain text
        h = html2text.HTML2Text()
        h.ignore_links = True
        h.ignore_images = True
        text = h.handle(str(soup))

        # Get the title
        title = soup.title.string if soup.title else "No title found"

        # Truncate the content if it's too long
        max_length = 1000
        if len(text) > max_length:
            text = text[:max_length] + "..."

        return title, text
    except Exception as e:
        logger.warning(f"Error fetching content from {url}: {str(e)}")
        return "Error fetching content", f"Could not fetch content from {url}: {str(e)}"

def GoogleSearch(query, config, llm_type, num_results=5):
    logger.info(f"Starting Google Search for query: '{query}'")

    # Perform the search
    search_results = list(search(query, num_results=num_results))

    logger.info(f"Retrieved {len(search_results)} search results")

    # Extract content from each result
    organic_results = []
    for i, url in enumerate(search_results):
        title, content = extract_content(url)
        organic_results.append({
            "title": title,
            "link": url,
            "content": content,
            "rank": i + 1
        })

    logger.info(f"Extracted content from {len(organic_results)} search results")

    # Create a prompt for the LLM to summarize content
    results_text = "\n\n".join([
        f"{result['rank']}. {result['title']} ({result['link']})\nContent: {result['content']}"
        for result in organic_results
    ])
    prompt = f"""
    You are a web content summarizer. For each of the following search results, provide a brief summary of the content. Here are the search results for the query "{query}":

    {results_text}

    Please provide the summary for each result in the following format:
    1. [Title]
    Summary: [2-3 sentence summary of the content]
    2. [Title]
    Summary: [2-3 sentence summary of the content]
    ... and so on for all results.
    """

    logger.info(f"Sending prompt to LLM for content summarization using LLM type: {llm_type}")

    # Use the LLM to summarize content
    llm_manager = LLMManager(config)
    summarized_content = llm_manager.call_llm({'type': llm_type}, prompt)

    logger.info("Received summarized content from LLM")

    return summarized_content

def main(input_data):
    """
    Main function to search Google and summarize content, use like this {"GoogleSearch": {"query": "__query_here__"}}
    """
    logger.info("Starting main function of GoogleSearch tool")

    framework = MultiAgentFramework(project_root + '/Frendy/')
    query = input_data.get('query')
    llm_type = 'ollama'
    num_results = input_data.get('num_results', 5)

    if query is None:
        logger.error("The 'query' field is missing in the input data")
        raise ValueError("The 'query' field is required in the input data.")

    logger.info(f"Executing Google Search for query: '{query}' using LLM type: {llm_type}")

    config = framework.config
    result = GoogleSearch(query, config, llm_type, num_results)

    logger.info("Google Search completed successfully")

    return result

# Example usage:
if __name__ == "__main__":
    logger.info("Running GoogleSearch tool in standalone mode")

    # This part is for testing the tool directly
    query = "python tutorial"
    framework = MultiAgentFramework(project_root + '/Frendy/')
    config = framework.config
    llm_type = 'ollama'  # or 'openai'

    logger.info(f"Testing GoogleSearch function with query: '{query}'")
    results = GoogleSearch(query, config, llm_type)
    print("Results from direct GoogleSearch function call:")
    print(results)

    # This part tests the main function as it would be called by the framework
    input_data = {"query": "python tutorial", "num_results": 3}
    logger.info(f"Testing main function with input data: {input_data}")
    results = main(input_data)
    print("\nResults from main function call:")
    print(results)

    logger.info("GoogleSearch tool test completed")