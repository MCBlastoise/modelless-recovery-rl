import numpy as np
from PIL import Image, ImageOps
from agent import Agent
import random
from scipy.ndimage import gaussian_filter

class Environment:
    def __init__(self, occupancy_data: str | np.ndarray, num_agents, width, height, completion_percentage):
        self.width = width
        self.height = height

        if type(occupancy_data) is str:
            self.occupancy_grid = self.read_from_file(occupancy_data)
        else:
            self.occupancy_grid = occupancy_data

        self.agents = []
        for _ in range(num_agents):
            agent = Agent(initial_pdm=np.full((self.height,self.width), 0.4), initial_coords=self.get_random_position())
            self.agents.append(agent)

        for ix, agent in enumerate(self.agents):
            other_agents = self.agents[:ix] + self.agents[ix + 1:]
            agent.share_other_agents(other_agents)

        self.success = 0
        self.fail = 1

        self.completion_percentage = completion_percentage

    def get_cohesive_explored_map(self):
        map_shape = (self.height, self.width)
        self.cohesive_map = np.zeros(shape=map_shape)
        for row in range(self.height):
            for col in range(self.width):
                for agent in self.agents:
                    if agent.explored[row, col]:
                        self.cohesive_map[row, col] = 1
                        break
        return self.cohesive_map

    def get_random_position(self):
        while True:
            pos = (random.randint(0, self.height-1), random.randint(0, self.width-1))
            if self.is_occupied(pos):
                continue
            return pos

    def read_from_file(self, image_filename):
        img = ImageOps.grayscale(Image.open(image_filename))
        img = img.resize((self.width, self.height))
        pixels = np.array(img.getdata())

        for i in range(len(pixels)):
            if pixels[i] < 128:
                pixels[i] = 0
            else:
                pixels[i] = 1

        pixels = pixels.reshape((self.width,self.height))
        return pixels
        

    def is_occupied(self, coords: tuple[int, int]) -> bool:
        return bool(self.occupancy_grid[*coords])

    def update_pos(self):
        """
        update every agent's position with either the next action or recovery action
        update every agent's pdm to reflect what they encounter in this step
        """

        agent_actions = [agent.get_next_action() for agent in self.agents]
        for ix, (agent, action) in enumerate(zip(self.agents, agent_actions)):
            # action = agent.get_next_action()

            is_collision = False
            for jx, other_action in enumerate(agent_actions):
                if ix == jx:
                    continue
                if action == other_action:
                    is_collision = True

            if self.is_occupied(action): # tried to do an action that results in constraint violation
                self.fail += 1
                print("Agent made a mistake, resetting to random position")
                agent.take_step(coords=self.get_random_position(), success=False)
                # agent.reset_for_failure()
            elif is_collision:
                self.fail += 1
                agent.take_step(coords=self.get_random_position(), success=True, neutral=True)
            else: # all good all safe
                agent.take_step(coords=action, success=True)
                # agent.successful_move(action)
                # agent.update_explored(action)
                self.success += 1
            
            # print(agent.other_agents)
        cohesive_map = self.get_cohesive_explored_map()
        if not self.explored_enough(cohesive_map):
            return
        
        for agent in self.agents:
            print("goal completed")
            agent.inform_goal_completed()
        
    def explored_enough(self, cohesive_map):
        map_no_obstacles_size = np.count_nonzero(self.occupancy_grid == 0)
        explored_frac = np.sum(cohesive_map) / map_no_obstacles_size
        # print(explored_frac)
        return explored_frac >= self.completion_percentage
