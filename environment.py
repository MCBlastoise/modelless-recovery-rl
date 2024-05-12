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


    def get_random_position(self):
        return (random.randint(0, self.height-1), random.randint(0, self.width-1))

    def read_from_file(self, image_filename):
        img = ImageOps.grayscale(Image.open(image_filename))
        img = img.resize((self.width, self.height))
        pixels = np.array(img.getdata())

        for i in range(len(pixels)):
            if pixels[i] < 128:
                pixels[i] = 0
            else:
                pixels[i] = 1

        pixels = pixels.reshape((100,100))
        return pixels
        

    def is_occupied(self, coords: tuple[int, int]) -> bool:
        return bool(self.occupancy_grid[*coords])

    def update_pos(self):
        """
        update every agent's position with either the next action or recovery action
        update every agent's pdm to reflect what they encounter in this step
        """

        for agent in self.agents:
            action = agent.get_next_action()

            if self.is_occupied(action): # tried to do an action that results in constraint violation
                self.fail += 1
                # print("Agent made a mistake, resetting to random position")
                agent.take_step(coords=self.get_random_position(), success=False)
                # agent.reset_for_failure()
            else: # all good all safe
                agent.take_step(coords=action, success=True)
                # agent.successful_move(action)
                # agent.update_explored(action)
                self.success += 1
            
            # print(agent.other_agents)
        


    def fully_explored(self, agent):
        xor_result = np.bitwise_xor(agent.explored.astype(int), self.occupancy_grid.astype(int))
        visited = np.count_nonzero(xor_result)
        return visited > self.completion_percentage*self.occupancy_grid.size 
