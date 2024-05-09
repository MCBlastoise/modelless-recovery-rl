from environment import Environment
from visualization import RobotVisualization
from agent import Agent

occupancy_data = np.zeros((100,100))
occupancy_data[30, :40] = 1
occupancy_data[60, 60:] = 1
environment = Environment(occupancy_data, 1)
anim = RobotVisualization(environment)
while True:
    environment.update_pos()
    anim.update(environment)
anim.done()