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
python -m MultiAgent.multiagent_cli new MyProject
```

This will create a new directory `MyProject` with the basic structure and configuration files needed for a MultiAgent project.

### Adding Components

You can add new components (Agents, Tools, or Examples) to an existing project using the `add` command:

```bash
python -m MultiAgent.multiagent_cli add MyProject Agent MyNewAgent
python -m MultiAgent.multiagent_cli add MyProject Tool MyNewTool
python -m MultiAgent.multiagent_cli add MyProject Example MyNewExample
```

### Running a Conversation

To start a conversation in an existing project:

```bash
python -m MultiAgent.multiagent_cli run ./MyProject
```

This command will initialize the framework with your project's configuration and prompt you for an initial input to start the conversation.

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
  - GoogleSearch
pre_prompt: true
post_prompt: true
agentConnections:
  - SummarizerAgent
color: "#FFA07A"
llm_config:
  type: ollama
  model: llama3
  temperature: 0.3
  max_tokens: 1000
  stream: true
rag_config:
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

The framework includes a Retrieval-Augmented Generation (RAG) system that enhances the agents' capabilities by providing relevant information from a vector database. The RAG system uses ChromaDB as the default vector store and can be configured globally or per agent.

Key RAG features include:
- Customizable vector database settings
- Configurable embedding models
- Adjustable chunk size and overlap for text processing
- Flexible retrieval options

You can also implement a custom RAG manager by specifying the `custom_rag_manager` path in the configuration.

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for more details.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.