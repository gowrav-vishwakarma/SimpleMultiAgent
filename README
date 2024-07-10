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
llm_config:
  type: ollama
  model: phi3:latest
  temperature: 0.3
  max_tokens: 1000
  stream: true
```

### Creating Tools

Tools are Python scripts located in the `Tools/` directory. Each tool should have a `main` function that the framework will call. For example:

```python
def main(params):
    # Tool logic here
    return result
```

### Defining Examples

Examples are text files in the `Examples/` directory. They can be referenced in agent prompts using the `#ExampleName` syntax.

## Configuration

### Main Configuration File

The `config.yaml` file in the project root directory contains the main configuration for the framework:

```yaml
framework:
  base_path: ./
  default_agent: InitialAgent
  pre_prompt: >
    You are an experienced Executive Assistant. Your task is to manage communication and coordination between team members, stakeholders, and clients.

    Other agents you can collaborate with:
    $otherAgents

    Tools at your disposal:
    $tools

    When given a task, think through the problem step-by-step, consider the roles and capabilities of other agents, and use the available tools when necessary. Provide detailed explanations of your thought process and decisions.
  post_prompt: >
    Role: {agent.role}
    Previous Context: {agent.memory}
    Knowledge: {agent.role_knowledge}
    Current Input: {input_data}
    Task: Process the input based on your role, knowledge, and the instructions above.
    If there are any tool results, analyze and incorporate them into your response.
    Provide your thoughts and any necessary JSON updates.
    Suggest the next action or agent to handle the task.
    Format your response as follows:
    THOUGHTS: [Your reasoning process, including analysis of any tool results]
    USE_TOOL: [Optional: Specify a tool to use, {"tool_name": {"parameter": "value", "parameter": "value"} } ]
    NEXT_ACTION: [Suggest the next action or agent, e.g., NEXT_AGENT: DeveloperAgent or FINISH if the task is complete]
    Begin your response now:
  tool_extract_methods:
    - name: json_format
      regexp: 'USE_TOOL:\s*(\{(?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*\})'
      parse_method: json
      tool_name_extractor: '"(\w+)"'
      params_extractor: ':\s*(\{.*?\})(?=\s*\})'
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

Each agent is configured in its own YAML file within the `Agents/` directory. The configuration includes the agent's name, role, prompt, tools, and LLM settings.

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

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for more details.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
