
import json
from typing import List, Dict, Any

class ReActNotebook:
    def __init__(self):
        self.trajectory: List[Dict[str, Any]] = []

    def add_thought(self, thought: str):
        self.trajectory.append({"type": "thought", "content": thought})

    def add_action(self, action: str, inputs: Dict[str, Any]):
        self.trajectory.append({"type": "action", "action": action, "inputs": inputs})

    def add_observation(self, observation: Any):
        self.trajectory.append({"type": "observation", "content": observation})

    def add_code(self, code: str):
        self.trajectory.append({"type": "code", "content": code})

    def add_error(self, error: str):
        self.trajectory.append({"type": "error", "content": error})

    def add_self_correction(self, correction: str):
        self.trajectory.append({"type": "self_correction", "content": correction})

    def to_json(self) -> str:
        return json.dumps(self.trajectory, indent=2)

    def from_json(self, json_str: str):
        self.trajectory = json.loads(json_str)
