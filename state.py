from typing import List

from constant import *

from action import Action

class State:
    x: int
    y: int
    value: float
    action_space: List[Action]
    def __init__(self, x: int, y: int) -> None:
        self.x, self.y = x, y
        self.action_space = [
            Action(dx=0, dy=0), # O
            Action(dx=0, dy=-1), # L
            Action(dx=0, dy=1), # R
            Action(dx=-1, dy=0), # U
            Action(dx=1, dy=0), # D
        ]
        
    def __str__(self) -> str:
        return f"({self.x}, {self.y})"
        
    def __eq__(self, __o: object) -> bool:
        return self.x == __o.x and self.y == __o.y

    @property
    def id(self) -> int:
        return self.x * n + self.y
        
    def __hash__(self) -> int:
        return self.id