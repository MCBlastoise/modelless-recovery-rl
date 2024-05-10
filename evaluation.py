from environment import Environment
from visualization import RobotVisualization
from agent import Agent
import numpy as np
import matplotlib.pyplot as plt
import cv2

occupancy_data = np.zeros((30, 30))
occupancy_data[20:25, :10] = 1
occupancy_data[5:10, 20:] = 1
# occupancy_data[:10, 5:15] = 1
# occupancy_data[20:, 15:25] = 1
environment = Environment(occupancy_data, 1, 30, 30, 0.9)
anim = RobotVisualization(environment)
ratios = []
for i in range(10000):

    # print("Before environment update")
    environment.update_pos()

    if len(environment.agents) == 0:
        print("agents ran out")
        break
    # print("Environment update")
    anim.update(environment)
    # print("After animation update")
    ratios.append(environment.success/environment.fail)
    print(i)


x_vals = [i for i in range(len(ratios))]
print(ratios)
plt.plot(x_vals, ratios)
plt.xlabel('Iterations')
plt.ylabel('Ratio of Successes to Failures')
plt.title('Evaluation of Recovery RL on 30x30')
plt.show()
# anim.done()