class Reward:
    value: float
    def __init__(self, value: float) -> None:
        self.value = value
    
    def __float__(self) -> float:
        return self.value
    
    def __str__(self) -> str:
        return str(self.value)
    
    def __add__(self, __o: object) -> object:
        if hasattr(__o, "value"):
            return Reward(self.value + __o.value)
        else:
            return Reward(self.value + __o)
    
    def __iadd__(self, __o: object) -> object:
        if hasattr(__o, "value"):
            self.value += __o.value
        else:
            self.value += __o
        return self
        
    def __mul__(self, __o: object) -> object:
        if hasattr(__o, "value"):
            return Reward(self.value * __o.value)
        else:
            return Reward(self.value * __o)

    def __imul__(self, __o: object) -> object:
        if hasattr(__o, "value"):
            self.value *= __o.value
        else:
            self.value *= __o
        return self