# MultiAgent Framework

MultiAgent Framework is a powerful and flexible system for creating and managing multi-agent conversations and workflows. It provides a robust CLI for easy project management and a comprehensive framework for developing complex agent-based systems.

## Table of Contents

1. [Installation](#installation)
2. [CLI Usage](#cli-usage)
   - [Creating a New Project](#creating-a-new-project)
   - [Adding Components](#adding-components)
   - [Running a Conversation](#running-a-conversation)
3. [Framework Usage](#framework-usage)
   - [Project Structure](#project-structure)
   - [Configuring Agents](#configuring-agents)
   - [Creating Tools](#creating-tools)
   - [Defining Examples](#defining-examples)
4. [Configuration](#configuration)
   - [Main Configuration File](#main-configuration-file)
   - [Agent Configuration](#agent-configuration)
5. [Advanced Features](#advanced-features)
   - [Tool Extraction Methods](#tool-extraction-methods)
   - [Pre and Post Prompts](#pre-and-post-prompts)
   - [LLM Integration](#llm-integration)
   - [RAG (Retrieval-Augmented Generation)](#rag-retrieval-augmented-generation)
6. [Contributing](#contributing)
7. [License](#license)

## Installation

To install the MultiAgent Framework, use pip:

```bash
pip install multiagent-framework
```

## CLI Usage

The MultiAgent Framework comes with a powerful CLI tool for managing your projects.

### Creating a New Project

To create a new project, use the following command:

```bash
python -m multiagent_framework.multiagent_cli new MyProject
```

This will create a new directory `MyProject` with the basic structure and configuration files needed for a MultiAgent project.

### Adding Components

You can add new components (Agents, Tools, or Examples) to an existing project using the `add` command:

```bash
python -m multiagent_framework.multiagent_cli add MyProject Agent MyNewAgent
python -m multiagent_framework.multiagent_cli add MyProject Tool MyNewTool
python -m multiagent_framework.multiagent_cli add MyProject Example MyNewExample
```

### Running a Conversation

To start a conversation in an existing project:

```bash
python -m multiagent_framework.multiagent_cli run ./MyProject --verbosity user
```

This command will initialize the framework with your project's configuration and prompt you for an initial input to start the conversation. You can set the verbosity level to user, system, or debug.

## Framework Usage

### Project Structure

A typical MultiAgent project has the following structure:

```
MyProject/
├── Agents/
│   ├── Agent1.yaml
│   └── Agent2.yaml
├── Tools/
│   ├── Tool1.py
│   └── Tool2.py
├── Examples/
│   ├── Example1.txt
│   └── Example2.txt
├── RoleKnowledge/
│   └── role_knowledge.json
├── chroma_db/
└── config.yaml
```

### Configuring Agents

Agents are defined in YAML files within the `Agents/` directory. Here's an example:

```yaml
name: Executive Assistant
role: Managing communication and coordination between team members, stakeholders, and clients.
prompt: >
  You are an experienced Executive Assistant. Your task is to manage communication and coordination between team members, stakeholders, and clients.
  Other agents you can collaborate with:
  $otherAgents
  Tools at your disposal:
  $tools
  When given a task, think through the problem step-by-step, consider the roles and capabilities of other agents, and use the available tools when necessary. Provide detailed explanations of your thought process and decisions.
tools:
  - GoogleSearch  # List of tools this agent can use
pre_prompt: true  # Whether to use the global pre_prompt as prefix
post_prompt: true  # Whether to use the global post_prompt as suffix
agentConnections:
  - SummarizerAgent  # Other agents this agent can interact with
color: "#FFA07A"  # Color for console output
llm_config:  # Language Model configuration
  type: ollama
  model: phi3:latest
  temperature: 0.1
  max_tokens: 1000
  stream: true
rag_config:  # Retrieval-Augmented Generation configuration
  enabled: true
  vector_db:
    type: "chromadb"
    path: "./chroma_db"
  embedding_model:
    type: "default"
  chunk_size: 1000
  chunk_overlap: 200
  default_retriever:
    search_type: "similarity"
    search_kwargs:
      k: 5
```

### Creating Tools

Tools are Python scripts located in the `Tools/` directory. Each tool should have a `main` function that the framework will call. For example:

```python
def main(input_data, framework, current_agent):
    # Tool logic here
    return result
```

### Defining Examples

Examples are text files in the `Examples/` directory. They can be referenced in agent prompts using the `#ExampleName` syntax.

## Configuration

### Main Configuration File

The `config.yaml` file in the project root directory contains the main configuration for the framework. It includes settings for the framework, LLM integration, agents, tools, and RAG system.

Here's an example configuration:

```yaml
framework:
  base_path: ./
  default_agent: InitialAgent
  pre_prompt: >
    # Global pre-prompt text prefixed with each agent's prompt, override in agent to disable
  post_prompt: >
    # Global post-prompt text appended to each agent's response, override in agent to disable
  tool_extract_methods:
    - name: json_format
      regexp: 'USE_TOOL:\s*(\{(?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*\})'
      parse_method: json
      tool_name_extractor: '"(\w+)"'
      params_extractor: ':\s*(\{.*?\})(?=\s*\})'  # Changed this line
    - name: named_with_json
      regexp: 'USE_TOOL:\s*(\w+)\s+with\s+(\{(?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*\})'
      parse_method: json_with_name
      tool_name_extractor: '^(\w+)'
      params_extractor: '(\{.*\})$'
    - name: named_with_key_value
      regexp: 'USE_TOOL:\s*(\w+)\s+with\s+(.+)'
      parse_method: key_value_with_name
      tool_name_extractor: '^(\w+)'
      params_extractor: '(?<=with\s)(.+)$'
  rag:
    enabled: true
    vector_db:
      type: "chromadb"
      path: "./chroma_db"
    embedding_model:
      type: "default"  # ChromaDB uses its own default embedding model
    chunk_size: 1000
    chunk_overlap: 200
    default_retriever:
      search_type: "similarity"
      search_kwargs:
        k: 5
    custom_rag_manager: null  # Set this to the import path of a custom RAG manager class if needed
llm:
  openai:
    api_key: ${OPENAI_API_KEY}
    default_model: gpt-3.5-turbo
  ollama:
    api_base: http://localhost:11434
    default_model: phi3:latest
    stream: true
agents:
  - DeveloperAgent
  - DesignerAgent
  - ProductManagerAgent
tools_path: ./Tools
role_knowledge_path: ./RoleKnowledge
logging:
  level: INFO
  file: framework.log
```


### Agent Configuration

Each agent is configured in its own YAML file within the `Agents/` directory. The configuration includes the agent's name, role, prompt, tools, LLM settings, and RAG configuration.

## Advanced Features

### Tool Extraction Methods

The framework supports multiple methods for extracting tool usage from agent responses:

1. JSON Format
2. Named with JSON
3. Named with Key-Value Pairs

These methods are configured in the `tool_extract_methods` section of the main configuration file.

### Pre and Post Prompts

The framework supports pre-prompts and post-prompts for each agent, which can be enabled or disabled in the agent's configuration file. These prompts provide additional context and instructions to the agent before and after processing the main input.

### LLM Integration

The framework supports multiple Language Model providers, including OpenAI and Ollama. You can configure the LLM settings in the main configuration file and override them for individual agents if needed.

### RAG (Retrieval-Augmented Generation)

The MultiAgent Framework incorporates a powerful Retrieval-Augmented Generation (RAG) system that enhances the agents' capabilities by providing relevant information from a vector database. This feature allows agents to access and utilize a large knowledge base efficiently.

#### RAG Configuration

The RAG system can be configured globally in the main `config.yaml` file and can be overridden or customized for individual agents in their respective configuration files.

Global RAG configuration example:

```yaml
framework:
  rag:
    enabled: true
    vector_db:
      type: "chromadb"
      path: "./chroma_db"
    embedding_model:
      type: "default"
    chunk_size: 1000
    chunk_overlap: 200
    default_retriever:
      search_type: "similarity"
      search_kwargs:
        k: 5
    custom_rag_manager: null  # Optional: Path to a custom RAG manager class
```

Agent-specific RAG configuration example:

```yaml
rag_config:
  enabled: true
  vector_db:
    type: "chromadb"
    path: "./agent_specific_db"
  chunk_size: 500
  chunk_overlap: 100
```
#### RAG Functionality
The RAG system provides two main functions:

#### Saving Information: 
Agents can save information to the vector database for future retrieval.
#### Retrieving Information: 
Agents can query the vector database to retrieve relevant information based on a given query.

#### Built-in RAG Tools
The framework provides two default tools for interacting with the RAG system:

SaveInfoRAG: Allows agents to save information to the RAG system.

```Usage: {"SaveInfoRAG": {"content": "text to save", "tags": ["tag1", "tag2"]}}```

RetrieveInfoRAG: Enables agents to retrieve information from the RAG system.

```Usage: {"RetrieveInfoRAG": {"query": "search query", "tags": ["optional_tag1", "optional_tag2"]}}```

#### Custom RAG Manager
You can implement a custom RAG manager to use alternative vector databases or add specialized functionality:

1. Create a custom RAG manager class that inherits from RAGManager.
2. Implement the save_info and retrieve_info methods.
3. Specify the path to your custom RAG manager in the configuration:

```yaml
framework:
  rag:
    custom_rag_manager: "your_module.YourCustomRAGManager"
```

#### RAG Integration in Agent Processing
The RAG system is automatically integrated into the agent processing pipeline:

Before processing an agent's input, the framework checks if the agent has a RAG system enabled.
If enabled, it retrieves relevant information based on the input.
The retrieved information is then prepended to the agent's input, providing additional context for the agent's decision-making process.

This integration allows agents to leverage the knowledge stored in the vector database without explicit calls to the RAG system, enhancing their ability to provide informed responses and make better decisions.
Benefits of RAG in the MultiAgent Framework

- Enhanced Knowledge Access: Agents can access a vast amount of information beyond their initial training data. 
- Improved Decision-Making: By retrieving relevant context, agents can make more informed decisions and provide more accurate responses. 
- Dynamic Knowledge Base: The ability to save new information allows the system to grow and adapt over time. 
- Flexible Integration: The RAG system can be easily customized or replaced to suit specific project needs.

By leveraging the RAG system, the MultiAgent Framework provides a powerful tool for building more intelligent and adaptive multi-agent systems.

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for more details.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

This updated README provides a comprehensive overview of your MultiAgent Framework, including its installation, CLI usage, framework usage, configuration options, and advanced features. It incorporates the latest changes and features from your code, such as the updated CLI commands, the new verbosity options, and the detailed configuration examples for both the main config and agent config files.

The README now also includes more detailed explanations of the YAML configurations, helping users understand how to set up and customize their agents and the overall framework. It also highlights the flexibility of the framework in terms of LLM integration, tool extraction methods, and the RAG system.

You can copy this entire README and use it as the new README.md file for your project. It should provide users with a clear understanding of how to use and configure your MultiAgent Framework.