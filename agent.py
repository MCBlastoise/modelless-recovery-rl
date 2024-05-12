import numpy as np
import random
from enum import Enum

class Agent:
    EPSILON = 0.7
    PDM_POS_DELTA = 0.3
    PDM_NEG_DELTA = -0.1
    PATH_DANGER_WINDOW = 4
    DANGER_RADIUS = 2

    COMMUNICATION_THRESHOLD = 7
    
    def __init__(self, initial_pdm, goal_percentage, initial_coords = (0, 0)):
        self.pdm = np.copy(initial_pdm) # numpy array
        self.pos = initial_coords

        self.goal_satisfied = False # Flips when environment informs agent that they have explored enough

        self.explored = np.zeros(self.pdm.shape)
        self.update_explored(initial_coords)

        self.trajectory = None
        self.trajectory_step = None

        self.previous_positions = [None] * self.PATH_DANGER_WINDOW

        self.other_agents = []

    def inform_goal_completed(self):
        print("goal reached")
        self.goal_satisfied = True

    def share_other_agents(self, other_agents):
        self.other_agents = other_agents

    def other_agent_access(self, other_agent):
        my_pos, other_pos = self.pos, other_agent.pos
        x1, y1 = my_pos
        x2, y2 = other_pos
        distance = ((y2 - y1) ** 2 + (x2 - x1) ** 2) ** (1/2)

        if distance <= self.COMMUNICATION_THRESHOLD:
            return (other_agent.pos, other_agent.pdm)
        else:
            return None
    
    def get_available_others(self):
        other_agents_data = []
        for other_agent in self.other_agents:
            pos = self.other_agent_access(other_agent)
            if pos is not None:
                other_agents_data.append(pos)
        return other_agents_data

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

        possible_next_coords = self.get_next_coordinates()

        # Decide if safe
        if self.is_safe(self.pos):
            task_action = self.task_policy(possible_next_coords)
            self.trajectory_step = self.trajectory_step + 1 if self.trajectory_step is not None else self.trajectory_step
            return task_action

        # If not, invalidate trajectory and take recovery action

        # Failure, so invalidate trajectory
        self.invalidate_trajectory()
        recovery_action = self.recovery_step(possible_next_coords)
        # print("Took recovery", recovery_action)

        return recovery_action

    def task_policy(self, possible_next_coords):
        """
        policy to poll for desired action
        """

        if self.goal_satisfied:
            task_action = self.get_random_action(possible_next_coords)
        else:
            if self.trajectory is None or self.trajectory_step >= len(self.trajectory):
                self.trajectory = self.get_new_trajectory()
                self.trajectory_step = 0
            task_action = self.trajectory[self.trajectory_step]
        return task_action

    def get_random_action(self, possible_next_coords):
        danger_sorted = sorted(possible_next_coords, key=lambda c: self.pdm[*c])
        safer_choices = danger_sorted[:len(danger_sorted) // 2]
        return random.choice(safer_choices)

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
        # random.shuffle(next_coordinates)
        danger_sorted_coords = sorted(next_coordinates, key=lambda c: self.pdm[*c])
        return danger_sorted_coords

    def get_next_coordinates(self):
        return self.possible_steps(self.pos)

    def update_current_danger(self):
        self.update_zone_with_danger(self.pos, initial_scale_factor=1.5)
        # self.update_path_with_danger()

    def update_path_with_danger(self):
        valid_coords = [ (ix, coord) for ix, coord in enumerate(self.previous_positions, start=1) if coord is not None ]
        for ix, coord in valid_coords:
            # print("Updating coord", coord, "with danger")
            scale_factor = 1 / (ix ** 2)
            self.update_pdm(coord, obstacle=True, scale_factor=scale_factor)
    
    def update_zone_with_danger(self, coords, initial_scale_factor=1, obstacle=True):
        ROWS, COLS = self.pdm.shape
        r, c = coords

        danger_zone_coords = []
        for dr in range(-self.DANGER_RADIUS, self.DANGER_RADIUS + 1):
            for dc in range(-self.DANGER_RADIUS, self.DANGER_RADIUS + 1):
                danger_r, danger_c = r + dr, c + dc
                if (0 <= danger_r < ROWS and 0 <= danger_c < COLS):
                    danger_zone_coords.append( (danger_r, danger_c) )
        
        for danger_coord in danger_zone_coords:
            (x1, y1), (x2, y2) = self.pos, danger_coord
            distance = ((y2 - y1) ** 2 + (x2 - x1) ** 2) ** (1/2)
            scale_factor = initial_scale_factor * (1 / (distance ** 2) if distance != 0 else 1)
            self.update_pdm(danger_coord, obstacle=obstacle, scale_factor=scale_factor)

    def invalidate_trajectory(self):
        self.trajectory = None
        self.trajectory_step = None

    def take_step(self, coords, success: bool):
        if success:
            self.successful_move(coords)
        else:
            self.reset_for_failure(coords)
        
        self.update_others_danger()
        # self.get_fraction_explored()
        # print(communicable_poses)

    def update_others_danger(self):
        communicable_poses = self.get_available_others()
        for other_pos, other_pdm in communicable_poses:
            self.incorporate_other_pdm(other_pdm)
            self.update_zone_with_danger(coords=other_pos, initial_scale_factor=0.75)

    def reset_for_failure(self, coords):
        self.update_current_danger()
        self.trajectory, self.trajectory_step = None, None
        self.update_position(coords)

    def successful_move(self, coords):
        # raise NotImplementedError
        self.update_position(coords)
        # self.update_pdm(self.pos, False)
        self.update_zone_with_danger(coords=self.pos, initial_scale_factor=0.25, obstacle=False)

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

        safety = self.pdm[*coords]
        if safety < self.EPSILON:
            return True
        else:
            scaled_safety = safety ** (1 / 2)
            UNSAFE_CAP = 0.95
            capped_safety = min(scaled_safety, UNSAFE_CAP)
            random_val = random.random()
            return capped_safety < random_val
    
    def get_fraction_explored(self):
        frac = np.sum(self.explored) / self.explored.size
        print(self, frac)
        return frac
    
    def incorporate_other_pdm(self, other_pdm):
        ROWS, COLS = self.pdm.shape
        for row in range(ROWS):
            for col in range(COLS):
                my_prob, other_prob = self.pdm[row, col], other_pdm[row, col]
                average_prob = (my_prob + other_prob) / 2
                self.pdm[row, col] = average_prob