from environment import Environment
from visualization import RobotVisualization
from agent import Agent
import numpy as np

occupancy_data = np.zeros((30, 30))
occupancy_data[20, :10] = 1
occupancy_data[10, 20:] = 1
environment = Environment(occupancy_data, 1, 30, 30)
anim = RobotVisualization(environment)
while True:
    # print("Before environment update")
    environment.update_pos()
    # print("Environment update")
    anim.update(environment)
    # print("After animation update")
# anim.done()