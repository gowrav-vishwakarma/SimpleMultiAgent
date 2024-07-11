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
from multiagent_framework.MultiAgentFramework import MultiAgentFramework

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


def GoogleSearch(query, num_results=5):
    logger.info(f"Starting Google Search for query: '{query}'")

    # Perform the search
    search_results = list(search(query, num_results=num_results))

    logger.info(f"Retrieved {len(search_results)} search results")

    # Extract content from each result
    results = []
    for i, url in enumerate(search_results):
        title, content = extract_content(url)
        results.append({
            "title": title,
            "link": url,
            "content": content,
            "rank": i + 1
        })

    logger.info(f"Extracted content from {len(results)} search results")
    return results


def main(input_data, framework, current_agent):
    """
    Main function to search Google and return raw content, use like this {"GoogleSearch": {"query": "__query_here__"}}
    """
    logger.info("Starting main function of GoogleSearch tool")
    query = input_data.get('query')
    num_results = input_data.get('num_results', 5)
    if query is None:
        logger.error("The 'query' field is missing in the input data")
        raise ValueError("The 'query' field is required in the input data.")
    logger.info(f"Executing Google Search for query: '{query}'")
    result = GoogleSearch(query, num_results)
    logger.info("Google Search completed successfully")
    return result


# Example usage:
if __name__ == "__main__":
    # This part is for testing the tool directly
    query = "python tutorial"
    results = GoogleSearch(query)
    print("Results from direct GoogleSearch function call:")
    print(results)

    # This part tests the main function as it would be called by the framework
    input_data = {"query": "python tutorial", "num_results": 3}
    results = main(input_data)
    print("\nResults from main function call:")
    print(results)
