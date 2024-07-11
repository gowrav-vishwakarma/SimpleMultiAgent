import json
import os
from typing import Dict


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

