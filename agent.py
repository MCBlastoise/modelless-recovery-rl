import numpy as np
import random
from enum import Enum

class Agent:
    EPSILON = 0.6

    COMMUNICATION_THRESHOLD = 8

    OBSTACLE_DANGER_RADIUS = 2
    # DYNAMIC_DANGER_RADIUS = 2
    SAFE_RADIUS = 1

    DYNAMIC_DETECTION_RADIUS = 1

    PDM_UNSAFE_DELTA = 0.4
    PDM_SAFE_DELTA = -0.2
    PATH_DANGER_WINDOW = 4
    
    CRASH_SCALE_FACTOR = 2.0
    DYNAMIC_SCALE_FACTOR = 0.7

    SUCCESSFUL_MOVE_FACTOR = 0.25

    NEIGHBOR_PDM_FRAC = 0.25

    UNSAFE_CAP = 0.9

    # HOTSPOT_DURATION = 2

    PREVIOUS_GOAL_WINDOW = 5
    PREVIOUS_GOAL_RADIUS = 1
    
    
    def __init__(self, initial_pdm, initial_coords = (0, 0)):
        self.pdm = np.copy(initial_pdm) # numpy array
        self.pos = initial_coords

        self.goal_satisfied = False # Flips when environment informs agent that they have explored enough

        self.explored = np.zeros(self.pdm.shape)
        self.update_explored(initial_coords)

        self.trajectory = None
        self.trajectory_step = None

        self.previous_positions = [None] * self.PATH_DANGER_WINDOW

        self.previous_goals = [None] * self.PREVIOUS_GOAL_WINDOW

        self.other_agents = []

        self.hotspots = set()

    def inform_goal_completed(self):
        # print("goal reached")
        self.goal_satisfied = True

    def share_other_agents(self, other_agents):
        self.other_agents = other_agents

    def other_agent_access(self, other_agent):
        my_pos, other_pos = self.pos, other_agent.pos
        x1, y1 = my_pos
        x2, y2 = other_pos
        distance = ((y2 - y1) ** 2 + (x2 - x1) ** 2) ** (1/2)

        if distance <= self.COMMUNICATION_THRESHOLD:
            return (other_agent.pos, other_agent.pdm, other_agent.explored)
        else:
            return None
    
    def get_available_others(self):
        other_agents_data = []
        for other_agent in self.other_agents:
            data = self.other_agent_access(other_agent)
            if data is not None:
                other_agents_data.append(data)
        return other_agents_data

    def add_new_goal(self, goal_coords):
        self.previous_goals.pop(0)
        self.previous_goals.append(goal_coords)

    def close_to_previous_goals(self, coords):
        valid_prev_goals = [ coord for coord in self.previous_goals if coord is not None ]
        danger_set = set()
        for prev_goal in valid_prev_goals:
            danger_coords = self.get_coords_in_radius(prev_goal, radius=self.PREVIOUS_GOAL_RADIUS)
            danger_set |= set(danger_coords)
        return coords in danger_set

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
                if goal_func(neighbor) and not self.close_to_previous_goals(neighbor):
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
        task_action = self.task_policy(possible_next_coords)

        # Decide if safe
        if self.is_safe(task_action):
            self.trajectory_step = self.trajectory_step + 1 if self.trajectory_step is not None else self.trajectory_step
            return task_action

        # If not, invalidate trajectory and take recovery action

        # Failure, so invalidate trajectory
        if self.trajectory is not None:
            prev_goal = self.trajectory[-1]
            self.add_new_goal(prev_goal)
        
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
        safe_actions = []
        for coord in possible_next_coords:
            if self.is_safe(coord):
                safe_actions.append(coord)
        if len(safe_actions) > 0:
            return random.choice(safe_actions)
        return min(possible_next_coords, key=lambda x: self.pdm[*x])

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

    def update_path_with_danger(self):
        valid_coords = [ (ix, coord) for ix, coord in enumerate(self.previous_positions, start=1) if coord is not None ]
        for ix, coord in valid_coords:
            # print("Updating coord", coord, "with danger")
            scale_factor = 1 / (ix ** 2)
            self.update_pdm(coord, obstacle=True, scale_factor=scale_factor)
    
    def get_coords_in_radius(self, coords, radius):
        ROWS, COLS = self.pdm.shape
        r, c = coords

        danger_zone_coords = []
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                danger_r, danger_c = r + dr, c + dc
                if (0 <= danger_r < ROWS and 0 <= danger_c < COLS):
                    danger_zone_coords.append( (danger_r, danger_c) )
        
        return danger_zone_coords

    def update_zone(self, coords, initial_scale_factor=1, obstacle=True, dynamic=False):
        # ROWS, COLS = self.pdm.shape
        # r, c = coords

        if obstacle:
            danger_radius = self.DYNAMIC_DANGER_RADIUS if dynamic else self.OBSTACLE_DANGER_RADIUS
        else:
            danger_radius = self.SAFE_RADIUS

        danger_zone_coords = self.get_coords_in_radius(coords=coords, radius=danger_radius)

        # print(danger_radius)

        # danger_zone_coords = []
        # for dr in range(-danger_radius, danger_radius + 1):
        #     for dc in range(-danger_radius, danger_radius + 1):
        #         danger_r, danger_c = r + dr, c + dc
        #         if (0 <= danger_r < ROWS and 0 <= danger_c < COLS):
        #             danger_zone_coords.append( (danger_r, danger_c) )
        
        for danger_coord in danger_zone_coords:
            (x1, y1), (x2, y2) = self.pos, danger_coord
            distance = ((y2 - y1) ** 2 + (x2 - x1) ** 2) ** (1/2)
            scale_factor = initial_scale_factor * (1 / (distance ** 2) if distance != 0 else 1)
            self.update_pdm(danger_coord, obstacle=obstacle, scale_factor=scale_factor)

    def invalidate_trajectory(self):
        self.trajectory = None
        self.trajectory_step = None

    def take_step(self, coords, success: bool, neutral=False):
        if success:
            self.successful_move(coords, neutral)
        else:
            self.reset_for_failure(coords)
        
        self.update_hotspots()
        self.update_others_danger()

        print(id(self), self.previous_goals)
        
        # self.get_fraction_explored()
        # print(communicable_poses)

    def average_poses(self, coord1, coord2):
        x1, y1 = coord1
        x2, y2 = coord2
        average_coord = (x1 + x2) // 2, (y1 + y2) // 2
        return average_coord

    def update_others_danger(self):
        communicable_poses = self.get_available_others()
        for other_pos, other_pdm, other_explored in communicable_poses:
            self.incorporate_other_pdm(other_pdm)
            self.incorporate_other_explored(other_explored)

            # dynamic_collision_spot = self.average_poses(self.pos, other_pos)
            # self.update_zone(coords=dynamic_collision_spot, initial_scale_factor=self.DYNAMIC_SCALE_FACTOR, dynamic=True)
            # self.hotspots.add(dynamic_collision_spot)

            # self.update_zone(coords=dynamic_collision_spot, initial_scale_factor=self.DYNAMIC_SCALE_FACTOR)

    def update_hotspots(self):
        for coord in self.hotspots:
            # print("hotspot", coord)
            self.update_zone(coords=coord, initial_scale_factor=self.DYNAMIC_SCALE_FACTOR, obstacle=False, dynamic=True)
        self.hotspots = set()

    def reset_for_failure(self, coords):
        self.update_zone(self.pos, initial_scale_factor=self.CRASH_SCALE_FACTOR)
        self.trajectory, self.trajectory_step = None, None
        self.update_position(coords)

    def successful_move(self, coords, neutral=False):
        # raise NotImplementedError
        self.update_position(coords)
        # self.update_pdm(self.pos, False)
        if not neutral:
            self.update_zone(coords=self.pos, initial_scale_factor=self.SUCCESSFUL_MOVE_FACTOR, obstacle=False)

    def update_position(self, coords):
        self.pos = coords
        self.update_explored(coords)

        self.previous_positions.pop(0)
        self.previous_positions.append(self.pos)

        # print(self.previous_positions)

    def update_explored(self, coords):
        self.explored[*coords] = 1

    def update_pdm(self, coords: tuple[int, int], obstacle: bool, scale_factor=1):
        # if self.pdm[*coords] >= self.EPSILON:
        #     return
        delta = self.PDM_UNSAFE_DELTA if obstacle else self.PDM_SAFE_DELTA
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

        danger_set = set()
        communicable_poses = self.get_available_others()
        for other_pos, _, _ in communicable_poses:
            spot_in_between = self.average_poses(coord1=coords, coord2=other_pos)
            danger_coords = self.get_coords_in_radius(coords=spot_in_between, radius=self.DYNAMIC_DETECTION_RADIUS)
            danger_set |= set(danger_coords)

        if coords in danger_set:
            return False

        safety = self.pdm[*coords]
        if safety < self.EPSILON:
            return True
        else:
            scaled_safety = safety ** (1 / 2)
            capped_safety = min(scaled_safety, self.UNSAFE_CAP)
            random_val = random.random()
            return capped_safety < random_val
    
    def incorporate_other_pdm(self, other_pdm):
        ROWS, COLS = self.pdm.shape
        for row in range(ROWS):
            for col in range(COLS):
                my_prob, other_prob = self.pdm[row, col], other_pdm[row, col]
                highest_prob = max(my_prob, other_prob)
                if highest_prob >= self.EPSILON:
                    new_prob = highest_prob
                else:
                    new_prob = my_prob
                self.pdm[row, col] = new_prob
    
    def incorporate_other_explored(self, other_explored):
        ROWS, COLS = self.explored.shape
        for row in range(ROWS):
            for col in range(COLS):
                my_explored_val, other_explored_val = self.explored[row, col], other_explored[row, col]
                cohesive_explored = max(my_explored_val, other_explored_val)
                self.explored[row, col] = cohesive_explored