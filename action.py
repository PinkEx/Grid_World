class Action:
    dx: int
    dy: int
    def __init__(self, dx: int = None, dy: int = None) -> None:
        self.dx, self.dy = dx, dy

    def __str__(self) -> str:
        return f"({self.dx}, {self.dy})"
        
    def __eq__(self, __o: object) -> bool:
        return self.dx == __o.dx and self.dy == __o.dy

    def __hash__(self) -> int:
        if (self.dx, self.dy) == (0, -1):
            return 1
        elif (self.dx, self.dy) == (0, 1):
            return 2
        elif (self.dx, self.dy) == (-1, 0):
            return 3
        elif (self.dx, self.dy) == (1, 0):
            return 4
        else: return 0