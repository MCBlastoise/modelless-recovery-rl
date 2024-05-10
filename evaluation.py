from environment import Environment
from visualization import RobotVisualization
from agent import Agent
import numpy as np
import matplotlib as plt

occupancy_data = np.zeros((10, 10))
occupancy_data[3, :7] = 1
occupancy_data[7, 2:] = 1
environment = Environment(occupancy_data, 1, 10, 10)
anim = RobotVisualization(environment)
ratios = []
while len(environment.agents) > 0:
    # print("Before environment update")
    environment.update_pos()
    # print("Environment update")
    anim.update(environment)
    # print("After animation update")
    # ratios.append(environment.success/environment.fail)

x_vals = [i for i in range(len(ratios))]
print(ratios)
plt.plot(x_vals, ratios)
# anim.done()