import importlib
import inspect
import logging
import os
import json
import yaml
from typing import Dict, List, Any, Callable, Optional
from enum import Enum
import re
import openai
from ollama import Client as OllamaClient
import colorama
from colorama import Fore, Back, Style
import random

colorama.init(autoreset=True)


class PhaseType(Enum):
    THOUGHT = "THOUGHT"
    DRY_RUN = "DRY_RUN"
    FIND_BOTTLENECK = "FIND_BOTTLENECK"
    CORRECT = "CORRECT"


class LLMConfig:
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        self._apply_env_variables()

    def _apply_env_variables(self):
        def replace_env_vars(item):
            if isinstance(item, dict):
                return {k: replace_env_vars(v) for k, v in item.items()}
            elif isinstance(item, str) and item.startswith('${') and item.endswith('}'):
                env_var = item[2:-1]
                return os.getenv(env_var, item)
            return item

        self.config = replace_env_vars(self.config)

    def get_config(self):
        return self.config


class LLMManager:
    def __init__(self, config: dict):
        self.config = config
        self.openai_client = None
        self.ollama_client = None

        if 'openai' in self.config['llm']:
            self.openai_client = openai.OpenAI(api_key=self.config['llm']['openai']['api_key'])
        if 'ollama' in self.config['llm']:
            self.ollama_client = OllamaClient(host=self.config['llm']['ollama']['api_base'])

    def call_llm(self, llm_config: dict, prompt: str) -> str:
        llm_type = llm_config.get('type', '').lower()

        if llm_type == 'openai' and self.openai_client:
            return self._call_openai(llm_config, prompt)
        elif llm_type == 'ollama' and self.ollama_client:
            return self._call_ollama(llm_config, prompt)
        else:
            raise ValueError(f"Unsupported or unconfigured LLM type: {llm_type}")

    def _call_openai(self, config: dict, prompt: str) -> str:
        response = self.openai_client.chat.completions.create(
            model=config.get('model', self.config['llm']['openai']['default_model']),
            messages=[{"role": "user", "content": prompt}],
            temperature=config.get('temperature', 0.7),
            max_tokens=config.get('max_tokens', 1000)
        )
        return response.choices[0].message.content

    def _call_ollama(self, config: dict, prompt: str) -> str:
        print(f"Calling Ollama with prompt: {prompt}")
        response = self.ollama_client.chat(
            model=config.get('model', self.config['llm']['ollama']['default_model']),
            messages=[{"role": "user", "content": prompt}],
            stream=False
        )
        return response['message']['content']


class Agent:
    def __init__(self, name: str, prompt: str, role: str, llm_config: Dict, use_pre_prompt: bool = True,
                 use_post_prompt: bool = True, tools: List[Callable] = None):
        self.name = name
        self.prompt = prompt
        self.role = role
        self.llm_config = llm_config
        self.use_pre_prompt = use_pre_prompt
        self.use_post_prompt = use_post_prompt
        self.tools = tools or []
        self.memory = []
        self.thought_process = []
        self.role_knowledge = {}
        self.agent_connections = []

    def add_tool(self, tool: Callable):
        self.tools.append(tool)

    def add_memory(self, item):
        self.memory.append(item)

    def add_thought(self, thought: str):
        self.thought_process.append(thought)

    def set_role_knowledge(self, knowledge: Dict):
        self.role_knowledge = knowledge


class JSONManager:
    def __init__(self, base_path: str):
        self.base_path = base_path
        self.json_files = {}

    def load_json(self, file_path: str) -> Dict:
        full_path = os.path.join(self.base_path, file_path)
        if os.path.exists(full_path):
            with open(full_path, 'r') as f:
                return json.load(f)
        return {}

    def save_json(self, file_path: str, data: Dict):
        full_path = os.path.join(self.base_path, file_path)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        with open(full_path, 'w') as f:
            json.dump(data, f, indent=2)

    def update_json(self, file_path: str, update_data: Dict):
        current_data = self.load_json(file_path)
        self._deep_update(current_data, update_data)
        self.save_json(file_path, current_data)

    def _deep_update(self, d: Dict, u: Dict):
        for k, v in u.items():
            if isinstance(v, dict):
                d[k] = self._deep_update(d.get(k, {}), v)
            else:
                d[k] = v
        return d


class MultiAgentFramework:
    _instance = None

    def __new__(cls, base_path: str, debug_mode: bool = False):
        if cls._instance is None:
            cls._instance = super(MultiAgentFramework, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, base_path: str, debug_mode: bool = False):
        if self._initialized:
            return
        self.base_path = base_path
        self.config = LLMConfig(os.path.join(base_path, "config.yaml")).get_config()
        self.agents: Dict[str, Agent] = {}
        self.tools: Dict[str, Dict[str, Any]] = {}
        self.json_manager = JSONManager(base_path)
        self.llm_manager = LLMManager(self.config)
        self.examples: Dict[str, str] = {}
        self.debug_mode = debug_mode
        self._setup_logging()
        self.agent_colors = {}
        self.loaded_tool_names = set()  # Add this line
        self._initialize_agent_colors()
        self._initialized = True
        self.load_components()


    def _setup_logging(self):
        # Set up root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.WARNING)  # Set to WARNING to suppress most library logs

        # Create our custom logger
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG if self.debug_mode else logging.INFO)

        # Create console handler with formatting
        console_handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)

        # Add console handler to our logger
        self.logger.addHandler(console_handler)

        # Disable propagation to root logger
        self.logger.propagate = False

        # Suppress specific loggers
        logging.getLogger("openai").setLevel(logging.WARNING)
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("urllib3").setLevel(logging.WARNING)

    def _initialize_agent_colors(self):
        available_colors = [Fore.RED, Fore.GREEN, Fore.YELLOW, Fore.BLUE, Fore.MAGENTA, Fore.CYAN]
        for agent_name, agent in self.agents.items():
            if 'color' in agent.llm_config:
                self.agent_colors[agent_name] = getattr(Fore, agent.llm_config['color'].upper(), Fore.WHITE)
            else:
                self.agent_colors[agent_name] = random.choice(available_colors)
                available_colors.remove(self.agent_colors[agent_name])
                if not available_colors:
                    available_colors = [Fore.RED, Fore.GREEN, Fore.YELLOW, Fore.BLUE, Fore.MAGENTA, Fore.CYAN]

    def _debug_print(self, agent_name: str, message: str, message_type: str = "INFO"):
        if self.debug_mode:
            color = self.agent_colors.get(agent_name, Fore.WHITE)
            print(f"{color}[{agent_name}] {Style.BRIGHT}{message_type}: {Style.NORMAL}{message}{Style.RESET_ALL}")

    def _load_config(self):
        config_path = os.path.join(self.base_path, "config.yaml")

        with open(config_path, 'r') as f:
            config_str = f.read()

        # Replace environment variables
        config_str = re.sub(r'\$\{([^}^{]+)\}', lambda m: os.environ.get(m.group(1), m.group(0)), config_str)

        # Parse the YAML after environment variable substitution
        config = yaml.safe_load(config_str)

        return config

    def start_conversation(self, initial_prompt: str) -> Dict[str, Any]:
        """
        Start a new conversation with the multi-agent system.

        :param initial_prompt: The initial prompt or query to start the conversation.
        :return: The final result of the conversation.
        """
        print(f"Starting new conversation with prompt: {initial_prompt}")

        current_agent = self._determine_starting_agent(initial_prompt)
        current_data = initial_prompt

        while True:
            print(f"\nProcessing with agent: {current_agent.name}")
            result = self._process_agent(current_agent, current_data)
            current_data = result.get('output', current_data)

            print(f"Agent {current_agent.name} output: {current_data}")

            next_agent = self._determine_next_agent(result)
            if next_agent is None:
                print("\nConversation finished.")
                break
            current_agent = next_agent

        return result

    def load_components(self):
        try:
            self.load_tools()
            self.load_agents()
            self.load_examples()  # New method to load example files
            self.validate_and_update_agent_connections()
            self.update_agent_prompts()
            self.load_role_knowledge()
        except ValueError as e:
            print(f"Error during component loading: {str(e)}")
            raise SystemExit(1)

    def load_examples(self):
        examples_path = os.path.join(self.base_path, "Examples")
        for root, _, files in os.walk(examples_path):
            for file in files:
                if file.endswith(".txt"):
                    example_name = os.path.splitext(file)[0]
                    with open(os.path.join(root, file), 'r') as f:
                        self.examples[example_name] = f.read().strip()

    def load_tools(self):
        # Load custom tools
        custom_tools_path = os.path.join(self.base_path, "Tools")
        self._load_tools_from_directory(custom_tools_path, is_default=False)

    def _load_tools_from_directory(self, directory_path, is_default=False):
        for root, _, files in os.walk(directory_path):
            for file in files:
                if file.endswith(".py"):
                    tool_name = os.path.splitext(file)[0]
                    if tool_name not in self.loaded_tool_names:
                        module_path = os.path.join(root, file)

                        try:
                            spec = importlib.util.spec_from_file_location(tool_name, module_path)
                            module = importlib.util.module_from_spec(spec)
                            spec.loader.exec_module(module)

                            if hasattr(module, 'main'):
                                full_tool_name = f"default_{tool_name}" if is_default else tool_name
                                self.tools[full_tool_name] = {
                                    "function": module.main,
                                    "description": module.main.__doc__ or "No description available",
                                    "is_default": is_default
                                }
                                self.loaded_tool_names.add(tool_name)
                                print(f"Loaded tool: {full_tool_name} ({'default' if is_default else 'custom'})")
                        except Exception as e:
                            print(f"Error loading tool {tool_name}: {str(e)}")

    def load_agents(self):
        agents_path = os.path.join(self.base_path, "Agents")
        for root, _, files in os.walk(agents_path):
            for file in files:
                if file.endswith(".yaml"):
                    agent_name = os.path.splitext(file)[0]
                    with open(os.path.join(root, file), 'r') as f:
                        agent_config = yaml.safe_load(f)
                    prompt = agent_config.get('prompt', '')
                    role = agent_config.get('role', '')
                    tool_names = agent_config.get('tools', [])
                    llm_config = agent_config.get('llm_config', {})
                    agent_connections = agent_config.get('agentConnections', [])
                    use_pre_prompt = agent_config.get('pre_prompt', True)
                    use_post_prompt = agent_config.get('post_prompt', True)
                    agent = Agent(agent_name, prompt, role, llm_config, use_pre_prompt, use_post_prompt)
                    agent.agent_connections = agent_connections
                    agent.tools = tool_names  # Store tool names for validation
                    self.agents[agent_name] = agent

    def validate_and_update_agent_connections(self):
        undefined_agents = set()
        undefined_tools = set()

        for agent_name, agent in self.agents.items():
            # Validate agent connections
            for connection in agent.agent_connections:
                if connection not in self.agents:
                    undefined_agents.add(connection)

            # Validate tools
            for tool in agent.tools:
                if isinstance(tool, str):
                    if tool not in self.tools:
                        undefined_tools.add(tool)
                elif isinstance(tool, dict):
                    tool_name = tool.get('name')
                    if tool_name not in self.tools:
                        undefined_tools.add(tool_name)

        # If any undefined agents or tools are found, raise an error
        if undefined_agents or undefined_tools:
            error_message = "Validation failed. The following components are undefined:\n"
            if undefined_agents:
                error_message += f"Agents: {', '.join(undefined_agents)}\n"
            if undefined_tools:
                error_message += f"Tools: {', '.join(undefined_tools)}\n"
            raise ValueError(error_message)

        # If validation passes, update agent tools with actual tool functions
        for agent in self.agents.values():
            updated_tools = []
            for tool in agent.tools:
                if isinstance(tool, str):
                    if tool in self.tools:
                        updated_tools.append(self.tools[tool])
                elif isinstance(tool, dict):
                    tool_name = tool.get('name')
                    if tool_name in self.tools:
                        updated_tools.append(self.tools[tool_name])
            agent.tools = updated_tools

    def update_agent_prompts(self):
        other_agents_info = {name: agent.role for name, agent in self.agents.items()}

        def get_tool_info(tool):
            description = tool.get('description', 'No description available')
            if 'params' in tool:
                params = tool['params']
            else:
                # Get the function signature if 'params' is not available
                func = tool['function']
                signature = inspect.signature(func)
                params = str(signature)
            return f"({description}, {params})"

        tools_info = {name: get_tool_info(tool) for name, tool in self.tools.items()}

        for agent in self.agents.values():
            updated_prompt = agent.prompt.replace("$otherAgents", json.dumps(other_agents_info, indent=2))
            updated_prompt = updated_prompt.replace("$tools", json.dumps(tools_info, indent=2))

            # Replace #ExampleFile placeholders
            for example_name, example_content in self.examples.items():
                placeholder = f"#{example_name}"
                if placeholder in updated_prompt:
                    updated_prompt = updated_prompt.replace(placeholder, example_content)
                else:
                    print(f"Warning: Example file '{example_name}' not used in any prompt.")

            # Check for any remaining #ExampleFile placeholders
            remaining_placeholders = re.findall(r'#(\w+)', updated_prompt)
            if remaining_placeholders:
                print(f"Warning: The following example files were not found: {', '.join(remaining_placeholders)}")

            agent.prompt = updated_prompt

    def load_role_knowledge(self):
        knowledge_path = os.path.join(self.base_path, "RoleKnowledge")
        for root, _, files in os.walk(knowledge_path):
            for file in files:
                if file.endswith(".json"):
                    role = os.path.splitext(file)[0]
                    with open(os.path.join(root, file), 'r') as f:
                        role_knowledge = json.load(f)
                    for agent in self.agents.values():
                        if agent.role == role:
                            agent.set_role_knowledge(role_knowledge)

    def run_system(self, initial_input: str):
        self._debug_print("SYSTEM", f"Starting system with initial input: {initial_input}", "START")
        current_agent = self._determine_starting_agent(initial_input)
        current_data = initial_input
        conversation_step = 1

        while True:
            self._print_step_header(conversation_step, current_agent)
            result = self._process_agent(current_agent, current_data)
            self._print_step_result(result)

            current_data = result.get('output', current_data)
            next_agent = self._determine_next_agent(result)

            if next_agent is None:
                self._debug_print("SYSTEM", "Conversation finished.", "END")
                break

            current_agent = next_agent
            conversation_step += 1

            human_input = input(
                f"\n{Fore.WHITE}{Back.BLUE}Proceed to next agent ({current_agent.name})? Press Enter to continue or type 'stop' to end: {Style.RESET_ALL}")
            if human_input.lower() == 'stop':
                self._debug_print("SYSTEM", "Conversation finished by user request.", "END")
                break

        return current_data

    def _print_step_header(self, step: int, agent: Agent):
        color = self.agent_colors.get(agent.name, Fore.WHITE)
        print(f"\n{color}{'=' * 50}")
        print(f"{color}Step {step}: {agent.name} ({agent.role})")
        print(f"{color}{'=' * 50}{Style.RESET_ALL}")

    def _print_step_result(self, result: Dict[str, Any]):
        color = self.agent_colors.get(result['agent'], Fore.WHITE)
        print(f"\n{color}--- Agent Output ---")
        print(f"{color}Output: {result['output']}")
        print(f"\n{color}--- Agent Thoughts ---")
        for thought in result['thoughts']:
            print(f"{color}- {thought}")
        print(f"\n{color}Next Action: {result['next_action']}{Style.RESET_ALL}")

    def _determine_starting_agent(self, initial_input: str) -> Agent:
        # Logic to determine the starting agent based on initial input
        # For simplicity, we'll start with a default agent
        return self.agents.get("InitialAgent", next(iter(self.agents.values())))

    def _process_agent(self, agent: Agent, input_data: Any) -> Dict[str, Any]:
        self._debug_print(agent.name, f"Processing agent", "START")
        self._debug_print(agent.name, f"Input data: {input_data}", "INPUT")

        llm_input = self._prepare_llm_input(agent, input_data)
        self._debug_print(agent.name, f"LLM input:\n{llm_input}", "LLM_INPUT")

        llm_output = self.llm_manager.call_llm(agent.llm_config, llm_input)
        self._debug_print(agent.name, f"LLM output:\n{llm_output}", "LLM_OUTPUT")

        thoughts = self._extract_thoughts(llm_output)
        json_updates = self._extract_json_updates(llm_output)
        next_action = self._extract_next_action(llm_output)

        for thought in thoughts:
            agent.add_thought(thought)
            self._debug_print(agent.name, f"Thought: {thought}", "THOUGHT")

        tool_to_use = self._extract_tool_to_use(llm_output)
        if tool_to_use:
            self._debug_print(agent.name, f"Using tool: {tool_to_use['name']}", "TOOL")
            tool_result = self._execute_tool(tool_to_use, self.tools)
            if isinstance(tool_result, str) and tool_result.startswith("Error:"):
                agent.add_thought(f"Failed to use tool {tool_to_use['name']}: {tool_result}")
                self._debug_print(agent.name, f"Tool execution failed: {tool_result}", "TOOL_ERROR")
            else:
                agent.add_thought(f"Used tool {tool_to_use['name']}: {tool_result}")
                self._debug_print(agent.name, f"Tool result: {tool_result}", "TOOL_RESULT")
                input_data = tool_result

        for file_path, update_data in json_updates.items():
            self.json_manager.update_json(file_path, update_data)
            self._debug_print(agent.name, f"Updated JSON file: {file_path}", "JSON_UPDATE")

        if "HUMAN_INTERVENTION" in llm_output or not next_action:
            human_input = self._get_human_input(agent.name, input_data)
            agent.add_thought(f"Received human input: {human_input}")
            self._debug_print(agent.name, f"Human input: {human_input}", "HUMAN_INPUT")
            input_data = human_input

        result = {
            'agent': agent.name,
            'role': agent.role,
            'output': llm_output,
            'thoughts': agent.thought_process,
            'next_action': next_action
        }
        agent.add_memory(result)
        self._debug_print(agent.name, f"Processing complete. Next action: {next_action}", "END")
        return result

    def _execute_tool(self, tool_to_use: Dict[str, Any], available_tools: Dict[str, Dict[str, Any]]) -> Any:
        tool_name = next(iter(tool_to_use))
        tool_params = tool_to_use[tool_name]

        if tool_name in available_tools:
            tool = available_tools[tool_name]['function']
            try:
                return tool(tool_params)
            except Exception as e:
                self.logger.error(f"Error executing tool {tool_name}: {str(e)}")
                return f"Error: {str(e)}"
        else:
            self.logger.warning(f"Tool {tool_name} not found in available tools.")
            return f"Error: Tool {tool_name} not found"

    def _parse_json_tool(self, json_str: str) -> Dict[str, Any]:
        try:
            tool_dict = json.loads(json_str)
            if len(tool_dict) == 1 and isinstance(next(iter(tool_dict.values())), dict):
                return tool_dict
            else:
                self.logger.warning(f"Unexpected JSON format: {json_str}")
                return None
        except json.JSONDecodeError as e:
            self.logger.warning(f"Failed to parse JSON: {json_str}")
            self.logger.debug(f"JSON decode error: {str(e)}")
            self.logger.debug(f"Problematic JSON string: {json_str}")
            return None
    def _parse_json_with_name_tool(self, tool_name: str, json_str: str) -> Dict[str, Any]:
        try:
            params = json.loads(json_str)
            return {tool_name: params}
        except json.JSONDecodeError:
            self.logger.warning(f"Failed to parse JSON params: {json_str}")
            return None

    def _parse_key_value_with_name_tool(self, tool_name: str, params_str: str) -> Dict[str, Any]:
        params = self._parse_key_value_pairs(params_str)
        return {tool_name: params}

    def _parse_key_value_pairs(self, params_str: str) -> Dict[str, Any]:
        params = {}
        pairs = re.findall(r'(?:[^\s,"]|"(?:\\.|[^"])*")+', params_str)
        for pair in pairs:
            match = re.match(r'(\w+)\s*[:=]\s*(.+)', pair.strip())
            if match:
                key, value = match.groups()
                value = value.strip('"')
                params[key] = value
        return params

    def _extract_tool_to_use(self, llm_output: str) -> Optional[Dict[str, Any]]:
        extract_methods = self.config['framework'].get('tool_extract_methods', [])

        for method in extract_methods:
            self.logger.debug(f"Attempting to extract tool using method: {method['name']}")
            regexp = method['regexp']
            parse_method = method['parse_method']

            match = re.search(regexp, llm_output, re.DOTALL | re.IGNORECASE)
            if match:
                result = None
                if parse_method == 'json':
                    result = self._parse_json_tool(match.group(1))
                elif parse_method == 'json_with_name':
                    result = self._parse_json_with_name_tool(match.group(1), match.group(2))
                elif parse_method == 'key_value_with_name':
                    result = self._parse_key_value_with_name_tool(match.group(1), match.group(2))

                if result:
                    self.logger.info(f"Tool successfully extracted using method: {method['name']}")
                    return result
                else:
                    self.logger.debug(f"Method {method['name']} matched but failed to parse.")

            self.logger.warning("No matching tool extraction method found.")
        return None

    def _extract_next_action(self, llm_output: str) -> Dict[str, Any]:
        next_agent_match = re.search(r'NEXT_AGENT:\s*(\w+)', llm_output, re.IGNORECASE)
        if next_agent_match:
            next_agent = next_agent_match.group(1)
            if next_agent.lower() != "none":
                return {"type": "NEXT_AGENT", "agent": next_agent}
        if "FINISH" in llm_output.upper():
            return {"type": "FINISH"}
        return {"type": "CONTINUE"}

    def _prepare_llm_input(self, agent: Agent, input_data: Any) -> str:
        pre_prompt = self.config.get('framework', {}).get('pre_prompt', '') if agent.use_pre_prompt else ''
        post_prompt = self.config.get('framework', {}).get('post_prompt', '') if agent.use_post_prompt else ''

        def get_tool_info(tool):
            description = tool.get('description', 'No description available')
            if 'params' in tool:
                params = tool['params']
            else:
                # Get the function signature if 'params' is not available
                func = tool['function']
                signature = inspect.signature(func)
                params = str(signature)
            return f"({description}, {params})"

        # Prepare the replacement dictionary
        replacements = {
            "$otherAgents": json.dumps({name: a.role for name, a in self.agents.items() if a != agent}, indent=2),
            "$tools": json.dumps({name: get_tool_info(tool) for name, tool in self.tools.items()}, indent=2),
            "{agent.name}": agent.name,
            "{agent.role}": agent.role,
            "{agent.memory}": json.dumps(agent.memory[-5:] if agent.memory else [], indent=2),
            "{agent.role_knowledge}": json.dumps(agent.role_knowledge, indent=2),
            "{input_data}": str(input_data)
        }

        # Function to replace placeholders in a string
        def replace_placeholders(text):
            for key, value in replacements.items():
                text = text.replace(key, str(value))
            return text

        # Apply replacements to pre_prompt, agent.prompt, and post_prompt
        pre_prompt = replace_placeholders(pre_prompt)
        agent_prompt = replace_placeholders(agent.prompt)
        post_prompt = replace_placeholders(post_prompt)

        # Construct the final prompt
        prompt = f"""
        {pre_prompt}
        {agent_prompt}
        {post_prompt}
        """

        return prompt

    def _extract_thoughts(self, llm_output: str) -> List[str]:
        thoughts_section = re.search(r'THOUGHTS:(.*?)(?:JSON_UPDATES:|NEXT_ACTION:|$)', llm_output, re.DOTALL)
        if thoughts_section:
            return [thought.strip() for thought in thoughts_section.group(1).split('\n') if thought.strip()]
        return []

    def _extract_json_updates(self, llm_output: str) -> Dict[str, Dict]:
        updates = {}
        json_pattern = r'UPDATE_JSON\(([^)]+)\):\s*(\{[\s\S]+?\})'
        matches = re.findall(json_pattern, llm_output, re.DOTALL)
        for file_path, json_str in matches:
            try:
                # Remove any leading/trailing whitespace
                json_str = json_str.strip()

                # Replace any single quotes with double quotes
                json_str = json_str.replace("'", '"')

                # Handle potential line breaks and formatting issues
                json_str = re.sub(r'\s+', ' ', json_str)

                # Parse the JSON string
                json_data = json.loads(json_str)
                updates[file_path.strip()] = json_data
            except json.JSONDecodeError as e:
                print(f"Error parsing JSON for {file_path}: {str(e)}")
                print(f"Problematic JSON string: {json_str}")
        return updates

    def _determine_next_agent(self, result: Dict[str, Any]) -> Agent:
        next_action = result.get('next_action', {})
        if next_action.get('type') == "NEXT_AGENT":
            next_agent_name = next_action.get('agent')
            next_agent = self.agents.get(next_agent_name)
            if next_agent:
                return next_agent
            else:
                print(f"Warning: Specified next agent '{next_agent_name}' does not exist.")
                return self._handle_human_intervention(result)
        elif next_action.get('type') == "FINISH":
            return None
        else:
            return self._handle_human_intervention(result)

    def _handle_human_intervention(self, result: Dict[str, Any]) -> Agent:
        print("\n--- Human Intervention Required ---")
        print(f"Current Agent: {result.get('agent')}")
        print(f"Current Output: {result.get('output')}")
        print("\nAvailable Agents:")
        for idx, (agent_name, agent) in enumerate(self.agents.items(), 1):
            print(f"{idx}. {agent_name} ({agent.role})")

        while True:
            try:
                choice = int(input(
                    "\nPlease select the next agent number, 0 to finish the conversation, or -1 to continue with the current agent: "))
                if choice == 0:
                    return None  # Finish the conversation
                elif choice == -1:
                    return self.agents[result.get('agent')]  # Continue with current agent
                elif 1 <= choice <= len(self.agents):
                    return list(self.agents.values())[choice - 1]
                else:
                    print("Invalid choice. Please try again.")
            except ValueError:
                print("Please enter a valid number.")

    def _get_human_input(self, agent_name: str, current_data: Any) -> Any:
        print(f"\n--- Human Input Required for Agent: {agent_name} ---")
        print(f"Current Data: {current_data}")
        return input("Please provide your input or press Enter to continue: ")
