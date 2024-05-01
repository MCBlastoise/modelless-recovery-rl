class Agent:
    def __init__(self, initial_pdm):
        self.pdm = initial_pdm
    
    def next_action(self):
        pass

    def is_safe(self, coords: tuple[int, int]) -> bool:
        pass