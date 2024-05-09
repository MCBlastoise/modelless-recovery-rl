import numpy as np
from PIL import Image

class Environment:
    def __init__(self, occupancy_data: str | np.ndarray, num_agents):
        if type(occupancy_data) is str:
            self.occupancy_grid = Environment.read_from_file(occupancy_data)
        else:
            self.occupancy_grid = occupancy_data

        self.agents = []
        for i in range(num_agents):
            self.agents.append(Agent(np.zeros((100,100))))


    def read_from_file(image_filename):
        img = Image.open(image_filename)
        pixels = list(img.getdata())

        

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
            else: # all good all safe
                # update pdm to know that spot was good
                agent.x, agent.y = action
