import json
from typing import Dict, List, Any, Callable, Optional

from multiagent_framework.rag_system import RAGSystem


class Agent:
    def __init__(self, name: str, system_prompt: str, validation_prompt: str, role: str, llm_config: Dict,
                 tools: List[Callable] = None, rag_config: Dict = None, extracts: Dict = None, transition_prompts: Dict = None):
        self.raw_system_prompt = None
        self.validation_prompt = validation_prompt
        self.name = name
        self.system_prompt = system_prompt
        self.role = role
        self.llm_config = llm_config
        self.tools = tools or []
        self.conversation_history = [{"role": "system", "content": system_prompt}]
        self.thought_process = []
        self.role_knowledge = {}
        self.agent_connections = []
        self.rag_config = rag_config
        self.rag_system = None
        self.extracts = extracts or {}
        self.transition_prompts = transition_prompts or {}

    def initialize_rag_system(self, global_rag_config: Dict):
        if self.rag_config is None:
            self.rag_config = global_rag_config
        else:
            merged_config = global_rag_config.copy()
            merged_config.update(self.rag_config)
            self.rag_config = merged_config

        if self.rag_config.get('enabled', False):
            self.rag_system = RAGSystem(self.rag_config)

    def add_to_history(self, role: str, content: str):
        self.conversation_history.append({"role": role, "content": content})

    def add_thought(self, thought: str):
        self.thought_process.append(thought)
        self.add_to_history("thought", thought)

    def set_role_knowledge(self, knowledge: Dict):
        self.role_knowledge = knowledge

    def get_transition_prompt(self, next_agent: str) -> str:
        return self.transition_prompts.get(next_agent, self.transition_prompts.get('DEFAULT', ''))

    def get_history_as_string(self) -> str:
        return "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in self.conversation_history])


    def summarize_history(self, max_tokens: int = 1000):
        # Implement a method to summarize conversation history
        # This is a placeholder implementation
        total_tokens = sum(len(msg['content'].split()) for msg in self.conversation_history)
        if total_tokens > max_tokens:
            summary = f"Summarized {len(self.conversation_history)} messages..."
            self.conversation_history = [
                                            self.conversation_history[0],  # Keep the system message
                                            {"role": "system", "content": summary}
                                        ] + self.conversation_history[-3:]  # Keep the last 3 messages
