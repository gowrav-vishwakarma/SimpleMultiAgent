def main(input_data, framework, current_agent):
    """
    Save information to the agent's RAG system with tags.
    Use like this: {"SaveInfoRAG": {"content": "text to save", "tags": ["tag1", "tag2"]}}
    """
    content = input_data.get('content')
    tags = input_data.get('tags', [])

    if not content:
        return "Error: No content provided to save."

    if not current_agent.rag_system:
        return "Error: RAG system is not enabled for this agent."

    doc_id = current_agent.rag_system.save_info(content, tags)
    return f"Information saved successfully with ID: {doc_id}"