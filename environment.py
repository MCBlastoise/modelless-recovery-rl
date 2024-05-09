import numpy as np
from PIL import Image, ImageOps
from agent import Agent

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
            self.agents.append(Agent(np.zeros((height,width))))


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

            if is_occupied(action):
                # tried to make an action that results in constraint violation
                # update pdm
                # restart simulation
                pass
            else: # all good all safe
                # update pdm to know that spot was good
                agent.x, agent.y = action

e = Environment("handop.jpg", 1, 100, 100)