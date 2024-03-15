import random
from typing import Dict

from action import Action
from state import State

class Policy(dict):
    def __init__(self) -> None:
        pass

    def __str__(self) -> str:
        lines = []
        for s in self.keys():
            lines.append(f"{s}\n")
            for a in self[s].keys():
                lines.append(f"  {a}: {self[s][a]}")
            lines.append("\n")
        return "".join(lines)

    def set_action_probs(self, state: State, action_probs: Dict[Action, float]):
        self[state] = action_probs
    
    def get_action_prob(self, state: State, action: Action) -> float:
        action_probs: Dict[Action] = self.get(state)
        return action_probs[action]

    def get_action(self, state):
        action_probs: Dict[Action] = self.get(state)
        return random.choice(list(action_probs.keys()), weights=list(action_probs.values()))[0]
        