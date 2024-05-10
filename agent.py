import numpy as np
import random
from enum import Enum

# PlanState = Enum('PlanState', ['Planning', 'NotPlanning'])

class Agent:
    EPSILON = 0.7
    PDM_UPDATE_DELTA = 0.2
    PATH_DANGER_WINDOW = 4
    
    def __init__(self, initial_pdm, initial_coords = (0, 0)):
        self.pdm = np.copy(initial_pdm) # numpy array
        self.pos = initial_coords

        self.explored = np.zeros(self.pdm.shape)
        self.update_explored(initial_coords)

        self.trajectory = None
        self.trajectory_step = None

        self.previous_positions = [None] * self.PATH_DANGER_WINDOW

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
                    return list(new_path[1:])
                if neighbor not in seen:
                    seen.add(neighbor)
                    queue.append(new_path)

    def get_next_action(self):
        """
        returns action that agent should execute - either task policy or recovery policy
        """

        task_action = self.task_policy()

        # Decide if safe, if not get recovery action
        if self.is_safe(task_action):
            # print("Took task action", task_action)
            self.trajectory_step += 1
            return task_action

        # Failure, so invalidate trajectory
        self.invalidate_trajectory()

        possible_next_coords = self.get_next_coordinates()
        recovery_action = self.recovery_step(possible_next_coords)
        # print("Took recovery", recovery_action)

        return recovery_action

    def task_policy(self):
        """
        policy to poll for desired action
        """

        if self.trajectory is None or self.trajectory_step >= len(self.trajectory):
            self.trajectory = self.get_new_trajectory()
            self.trajectory_step = 0

        task_action = self.trajectory[self.trajectory_step]
        return task_action

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
        
        random.shuffle(next_coordinates)
        return next_coordinates

    def get_next_coordinates(self):
        return self.possible_steps(self.pos)

    def update_danger_zone(self):
        valid_coords = [ coord for coord in self.previous_positions if coord is not None ]
        for coord in valid_coords:
            print("Updating coord", coord, "with danger")
            self.update_pdm(coord, obstacle=True)

    def invalidate_trajectory(self):
        self.trajectory = None
        self.trajectory_step = None

    def reset_for_failure(self, coords):
        self.update_danger_zone()
        self.trajectory, self.trajectory_step = None, None
        self.update_position(coords)

    def successful_move(self, coords):
        # raise NotImplementedError
        self.update_position(coords)
        self.update_pdm(self.pos, False)

    def update_position(self, coords):
        self.pos = coords
        self.update_explored(coords)

        self.previous_positions.pop(0)
        self.previous_positions.append(self.pos)

        # print(self.previous_positions)

    def update_explored(self, coords):
        self.explored[*coords] = 1

    def update_pdm(self, coords: tuple[int, int], obstacle: bool):
        # delta = self.PDM_UPDATE_DELTA if obstacle else -self.PDM_UPDATE_DELTA
        # self.pdm[*coords] = max(min(self.pdm[*coords] + delta, 1.0), 0.0)
        self.pdm[*coords] = int(obstacle)

    def get_probability_obstacle(self, coords: tuple[int, int]):
        return self.pdm[*coords]

    def is_safe(self, coords: tuple[int, int]) -> bool:
        """
        returns boolean describing whether the given coord is above the epsilon safety value
        """
        return self.pdm[*coords] < self.EPSILON