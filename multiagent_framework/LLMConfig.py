import os

import yaml


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

