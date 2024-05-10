import numpy as np
import random
from enum import Enum

class Agent:
    EPSILON = 0.7
    PDM_POS_DELTA = 0.3
    PDM_NEG_DELTA = -0.1
    PATH_DANGER_WINDOW = 4
    DANGER_RADIUS = 2
    
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
                    final_path = list(new_path[1:])
                    # print(final_path)
                    return final_path
                if neighbor not in seen:
                    seen.add(neighbor)
                    queue.append(new_path)

    def get_next_action(self):
        """
        returns action that agent should execute - either task policy or recovery policy
        """

        # Decide if safe
        if self.is_safe(self.pos):
            task_action = self.task_policy()
            self.trajectory_step += 1
            return task_action

        # If not, invalidate trajectory and take recovery action

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
        for dr in range(-1, 2):
            for dc in range(-1, 2):
                new_r, new_c = r + dr, c + dc
                if (dr, dc) == (0, 0) or not (0 <= new_r < ROWS and 0 <= new_c < COLS):
                    continue
                next_coordinates.append( (new_r, new_c) )
        
        # if random.random() > 0.85:
        random.shuffle(next_coordinates)
        return next_coordinates

    def get_next_coordinates(self):
        return self.possible_steps(self.pos)

    def update_danger_zone(self):
        self.update_zone_with_danger()
        self.update_path_with_danger()

    def update_path_with_danger(self):
        valid_coords = [ (ix, coord) for ix, coord in enumerate(self.previous_positions, start=1) if coord is not None ]
        for ix, coord in valid_coords:
            # print("Updating coord", coord, "with danger")
            scale_factor = 1 / (ix ** 2)
            self.update_pdm(coord, obstacle=True, scale_factor=scale_factor)
    
    def update_zone_with_danger(self):
        ROWS, COLS = self.pdm.shape
        r, c = self.pos

        danger_zone_coords = []
        for dr in range(-self.DANGER_RADIUS, self.DANGER_RADIUS + 1):
            for dc in range(-self.DANGER_RADIUS, self.DANGER_RADIUS + 1):
                danger_r, danger_c = r + dr, c + dc
                if (0 <= danger_r < ROWS and 0 <= danger_c < COLS):
                    danger_zone_coords.append( (danger_r, danger_c) )
        
        for danger_coord in danger_zone_coords:
            (x1, y1), (x2, y2) = self.pos, danger_coord
            distance = ((y2 - y1) ** 2 + (x2 - x1) ** 2) ** (1/2)
            scale_factor = 1 / (distance ** 2) if distance != 0 else 1
            self.update_pdm(danger_coord, obstacle=True, scale_factor=scale_factor)

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

    def update_pdm(self, coords: tuple[int, int], obstacle: bool, scale_factor=1):
        delta = self.PDM_POS_DELTA if obstacle else self.PDM_NEG_DELTA
        scaled_delta = delta * scale_factor
        new_pdm = self.pdm[*coords] + scaled_delta
        clipped_pdm = max(min(new_pdm, 1.0), 0.0)
        self.pdm[*coords] = clipped_pdm

    def get_probability_obstacle(self, coords: tuple[int, int]):
        return self.pdm[*coords]

    def is_safe(self, coords: tuple[int, int]) -> bool:
        """
        returns boolean describing whether the given coord is above the epsilon safety value
        """
        return self.pdm[*coords] < self.EPSILON