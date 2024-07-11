from typing import Dict, List, Any, Callable, Optional, Tuple

from multiagent_framework.rag_system import RAGSystem


class Agent:
    def __init__(self, name: str, prompt: str, role: str, llm_config: Dict, use_pre_prompt: bool = True,
                 use_post_prompt: bool = True, tools: List[Callable] = None, rag_config: Dict = None):
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
        self.rag_config = rag_config
        self.rag_system = None  # Will be initialized when needed

    def initialize_rag_system(self, global_rag_config: Dict):
        if self.rag_config is None:
            self.rag_config = global_rag_config
        else:
            # Merge agent-specific config with global config, prioritizing agent-specific settings
            merged_config = global_rag_config.copy()
            merged_config.update(self.rag_config)
            self.rag_config = merged_config

        if self.rag_config.get('enabled', False):
            self.rag_system = RAGSystem(self.rag_config)

    def add_tool(self, tool: Callable):
        self.tools.append(tool)

    def add_memory(self, item):
        self.memory.append(item)

    def add_thought(self, thought: str):
        self.thought_process.append(thought)

    def set_role_knowledge(self, knowledge: Dict):
        self.role_knowledge = knowledge

