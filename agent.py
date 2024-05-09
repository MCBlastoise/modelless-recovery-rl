import numpy as np

class Agent:
    def __init__(self, initial_pdm, initial_r=0, initial_c=0):
        self.pdm = initial_pdm # numpy array
        self.r = initial_r
        self.c = initial_c
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
        if self.is_safe(task_action):
            return task_action

        recovery_action = self.recovery_step(possible_next_coords)
        return recovery_action

    def task_policy(self, possible_next_coords):
        step = possible_next_coords[0]
        for coord in possible_next_coords:
            if not self.explored[*coord]:
                step = coord
                break
        return step

    def get_probability_obstacle(self, coords: tuple[int, int]):
        return self.pdm[*coords]

    def is_safe(self, coords: tuple[int, int]) -> bool:
        """
        returns boolean describing whether the given coord is above the epsilon safety value
        """
        return self.pdm[*coords] > self.epsilon

    def recovery_step(self, possible_next_coords) -> tuple:
        """
        returns recovery action as a tuple, defined as the safest place to go from the current location
        """
        safest = None
        safest_prob = 1

        for next_coord in possible_next_coords:
            pnext_yob_violation = self.pdm[*next_coord]
            if pnext_yob_violation < safest_prob or safest is None:
                safest = next_coord
                safest_prob = pnext_yob_violation 

        return safest

    def get_position():
        return self.x, self.y