import openai
from ollama import Client as OllamaClient


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

        # log the prompt like output if verbosity is debug


        if llm_type == 'openai' and self.openai_client:
            return self._call_openai(llm_config, prompt)
        elif llm_type == 'ollama' and self.ollama_client:
            return self._call_ollama(llm_config, prompt)
        else:
            raise ValueError(f"Unsupported or unconfigured LLM type: {llm_type}")

    def _call_openai(self, config: dict, prompt: str) -> str:
        print(f"Calling OpenAi")
        response = self.openai_client.chat.completions.create(
            model=config.get('model', self.config['llm']['openai']['default_model']),
            messages=[{"role": "user", "content": prompt}],
            temperature=config.get('temperature', 0.7),
            max_tokens=config.get('max_tokens', 1000)
        )
        return response.choices[0].message.content

    def _call_ollama(self, config: dict, prompt: str) -> str:
        # print(f"Calling Ollama")

        # Extract Ollama-specific parameters
        model = config.get('model', self.config['llm']['ollama']['default_model'])
        temperature = config.get('temperature', 0.7)  # Default to 0.7 if not specified
        stream = config.get('stream', self.config['llm']['ollama'].get('stream', False))  # Default to False if not specified anywhere

        # Prepare the options dictionary
        options = {
            "temperature": temperature,
            # You can add other Ollama-specific parameters here, such as:
            # "top_p": config.get('top_p', 1.0),
            # "top_k": config.get('top_k', 40),
        }

        response = self.ollama_client.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            stream=stream,
            options=options  # Pass the options to the chat method
        )

        if stream:
            # Handle streaming response
            full_response = ""
            for chunk in response:
                if 'message' in chunk and 'content' in chunk['message']:
                    full_response += chunk['message']['content']
                    print(chunk['message']['content'], end='', flush=True)  # Print each chunk as it arrives
            print()  # Print a newline at the end
            return full_response
        else:
            # Handle non-streaming response
            return response['message']['content']
