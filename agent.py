import numpy as np
import random
from enum import Enum

# PlanState = Enum('PlanState', ['Planning', 'NotPlanning'])

class Agent:
    EPSILON = 0.7
    PDM_UPDATE_DELTA = 0.2
    
    def __init__(self, initial_pdm, initial_coords = (0, 0)):
        self.pdm = initial_pdm # numpy array
        self.pos = initial_coords

        self.explored = np.zeros(self.pdm.shape)
        self.update_explored(initial_coords)

        self.trajectory = None
        # self.plan_state = False
        trajectory = self.get_new_trajectory()
        print(trajectory)

    def get_new_trajectory(self):
        queue = [ (self.pos,) ]
        seen = set()
        goal_func = lambda c: not self.explored[*c]
        while queue:
            path = queue.pop(0)
            coords = path[-1]
            neighbors = self.possible_steps(coords)
            for neighbor in neighbors:
                new_path = path + (neighbor,)
                if goal_func(neighbor):
                    return new_path
                if neighbor not in seen:
                    seen.add(neighbor)
                    queue.append(new_path)

    def get_next_action(self):
        """
        returns action that agent should execute - either task policy or recovery policy
        """
        possible_next_coords = self.get_next_coordinates()

        # print("coord options", possible_next_coords)
        task_action = self.task_policy(possible_next_coords)

        # Decide if safe, if not get recovery action
        if self.is_safe(task_action):
            print("Took task action", task_action)
            return task_action

        recovery_action = self.recovery_step(possible_next_coords)
        print("Took recovery", recovery_action)

        return recovery_action

    def task_policy(self, possible_next_coords):
        """
        policy to poll for desired action
        """
        step = possible_next_coords[0]
        unexplored_coords, explored_coords = [], []

        for coord in possible_next_coords:
            if not self.explored[*coord]:
                unexplored_coords.append(coord)
            else:
                explored_coords.append(coord)
        
        if unexplored_coords:
            step = random.choice(unexplored_coords)
        elif explored_coords:
            step = random.choice(explored_coords)
        
        return step

    def recovery_step(self, possible_next_coords) -> tuple:
        """
        policy to poll for recovery action
        returns recovery action as a tuple, defined as the safest place to go from the current location
        """
        safest = []
        safest_prob = 1

        for next_coord in possible_next_coords:
            pnext_yob_violation = self.pdm[*next_coord]
            if pnext_yob_violation <= safest_prob or len(safest) == 0:
                if pnext_yob_violation == safest_prob:
                    safest.append(next_coord)
                else:
                    safest = [next_coord]
                    safest_prob = pnext_yob_violation 

        return random.choice(safest)

    def possible_steps(self, pos):
        ROWS, COLS = self.pdm.shape
        next_coordinates = []

        r, c = pos
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                new_r, new_c = r + dr, c + dc
                if (dr, dc) == (0, 0) or not (0 <= new_r < ROWS and 0 <= new_c < COLS):
                    continue
                next_coordinates.append( (new_r, new_c) )
        
        return next_coordinates

    def get_next_coordinates(self):
        return self.possible_steps(self.pos)

    def update_position(self, coords):
        self.pos = coords
        self.update_explored(coords)

    def update_explored(self, coords):
        self.explored[*coords] = 1

    def update_pdm(self, coords: tuple[int, int], obstacle: bool):
        delta = self.PDM_UPDATE_DELTA if obstacle else -self.PDM_UPDATE_DELTA
        self.pdm[*coords] = max(min(self.pdm[*coords] + delta, 1.0), 0.0)

    def get_probability_obstacle(self, coords: tuple[int, int]):
        return self.pdm[*coords]

    def is_safe(self, coords: tuple[int, int]) -> bool:
        """
        returns boolean describing whether the given coord is above the epsilon safety value
        """
        return self.pdm[*coords] < self.EPSILON