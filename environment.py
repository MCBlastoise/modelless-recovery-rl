import numpy as np
from PIL import Image, ImageOps
from agent import Agent
import random
from scipy.ndimage import gaussian_filter

class Environment:
    def __init__(self, occupancy_data: str | np.ndarray, num_agents, width, height):
        self.width = width
        self.height = height

        if type(occupancy_data) is str:
            self.occupancy_grid = self.read_from_file(occupancy_data)
        else:
            self.occupancy_grid = occupancy_data

        self.agents = []
        for _ in range(num_agents):
            agent = Agent(initial_pdm=np.full((self.height,self.width), 0.4), initial_coords=(random.randint(0, self.height-1), random.randint(0, self.width-1)))
            self.agents.append(agent)


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

        for i in range(len(self.agents)):
            agent = self.agents[i]
            action = agent.get_next_action()

            if self.is_occupied(action): # tried to do an action that results in constraint violation
                # update pdm to mark current spot as more unsafe
                agent.update_pdm(agent.pos, True)
                # update pdm to mark next spot as more unsafe
                agent.update_pdm(action, True)
                # restart simulation

                print("Agent made a mistake, resetting to random position")
                agent.update_position((random.randint(0, self.height - 1), random.randint(0, self.width - 1)))
            else: # all good all safe
                # update pdm to know that this spot and next spot are safe
                agent.update_pdm(agent.pos, False)
                agent.update_pdm(action, False)
                agent.update_position(action)
                # agent.update_explored(action)

            if self.fully_explored(agent):
                self.agents.remove(agent)

    def fully_explored(self, agent):
        xor_result = np.bitwise_xor(agent.explored.astype(int), self.occupancy_grid.astype(int))
        num_zeros = xor_result.size - np.count_nonzero(xor_result)
        print(num_zeros)
        return num_zeros == 0
