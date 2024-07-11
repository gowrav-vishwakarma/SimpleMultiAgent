def main(input_data, framework, current_agent):
    """
    Retrieve information from the agent's RAG system based on a query and optional tags.
    Use like this: {"RetrieveInfoRAG": {"query": "search query", "tags": ["optional_tag1", "optional_tag2"]}}
    """
    query = input_data.get('query')
    tags = input_data.get('tags', [])

    if not query:
        return "Error: No query provided for retrieval."

    if not current_agent.rag_system:
        return "Error: RAG system is not enabled for this agent."

    results = current_agent.rag_system.retrieve_info(query, tags)
    return results