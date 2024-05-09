import numpy as np

class Agent:
    DANGER_THRESHOLD = 0.2
    
    def __init__(self, initial_pdm, initial_x=0, initial_y=0):
        self.pdm = initial_pdm # numpy array
        self.x = initial_x
        self.y = initial_y
        self.epsilon = 0.5
    
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
        possible_next_coords = self.get_next_coordinates()

        task_action = self.task_policy(possible_next_coords)

        # Decide if safe, if not get recovery action

        raise NotImplementedError

        recovery_action = self.recovery_step(possible_next_coords)

    def task_policy(self, possible_next_coords):
        raise NotImplementedError

    def is_safe(self, coords: tuple[int, int]) -> bool:
        """
        returns boolean describing whether the given coord is above the epsilon safety value
        """
        return self.pdm[i, j] > self.epsilon

    def recovery_step(self, possible_next_coords) -> tuple:
        """
        returns recovery action as a tuple, defined as the safest place to go from the current location
        """
        safest = None
        safest_prob = 1

        for next_y, next_x in possible_next_coords:
            pnext_yob_violation = self.pdm[next_y, next_x]
            if prob_violation < safest_prob or safest is None:
                safest = (next_y, next_x)
                safest_prob = prob_violation 

        return safest