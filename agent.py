import numpy as np

class Agent:
    DANGER_THRESHOLD = 0.2
    
    def __init__(self, initial_pdm):
        self.pdm = initial_pdm
        self.explored = np.zeros(initial_pdm.shape)
    
    def get_next_coordinates(self):
        ROWS, COLS = self.pdm.shape
        next_coordinates = []

        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                new_r, new_c = self.r + dr, self.c + dc
                if (dr, dc) == (0, 0) or not (0 <= new_r < ROWS and 0 <= new_c < COLS):
                    continue
                next_coordinates.append( (new_r, new_c) )
        
        return next_coordinates

    def get_next_action(self):
        next_coords = self.get_next_coordinates()

    def task_policy(self):
        raise NotImplementedError

    def is_safe(self, coords: tuple[int, int]) -> bool:
        pass