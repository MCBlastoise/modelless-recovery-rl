import numpy as np
from PIL import Image, ImageOps
from agent import Agent
import random

class Environment:
    def __init__(self, occupancy_data: str | np.ndarray, num_agents, width, height):
        self.width = width
        self.height = height

        if type(occupancy_data) is str:
            self.occupancy_grid = self.read_from_file(occupancy_data)
        else:
            self.occupancy_grid = occupancy_data

        self.agents = []
        for i in range(num_agents):
            self.agents.append(Agent(np.full((height,width), 0.3)))


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
                # update pdm to mark current spot as more unsafe
                agent.update_pdm(agent.pos, True)
                # update pdm to mark next spot as more unsafe
                agent.update_pdm(action, True)
                # restart simulation
                agent.update_position((random.randint(0, self.height), random.randint(0, self.width)))
            else: # all good all safe
                # update pdm to know that this spot and next spot are safe
                agent.update_pdm(agent.pos, False)
                agent.update_pdm(action, False)
                agent.update_position(action)
                # agent.update_explored(action)