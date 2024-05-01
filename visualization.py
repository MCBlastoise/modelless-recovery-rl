import math
import time
from tkinter import *
from environment import Environment
from agent import Agent

class RobotVisualization:
    """
    Visualization of a Robot simulation.
    """

    def __init__(self, environment, delay=0.2):
        "Initializes a visualization with the specified parameters."
        # Number of seconds to pause after each frame
        self.delay = delay
        width = environment.occupancy_grid.shape[1]
        height = environment.occupancy_grid.shape[0]
        self.max_dim = max(width, height)
        self.width = width
        self.height = height

        # Initialize a drawing surface
        self.master = Tk()
        self.w = Canvas(self.master, width=800, height=800)
        self.w.pack()
        self.master.update()

        # Draw a backing and lines
        x1, y1 = self._map_coords(0, 0)
        x2, y2 = self._map_coords(width, height)
        self.w.create_rectangle(x1, y1, x2, y2, fill="white")

        # Draw gray squares for dusty tiles
        self.tiles = {}
        for i in range(width):
            for j in range(height):
                x1, y1 = self._map_coords(i, j)
                x2, y2 = self._map_coords(i + 1, j + 1)
                if (i, j) not in self.tiles:
                    self.tiles[(i, j)] = self.w.create_rectangle(
                        x1, y1, x2, y2, fill="black"
                    )
                else: # not really sure what this is for
                    self.tiles[(i, j)] = self.w.create_rectangle(
                        x1, y1, x2, y2, fill="red"
                    )

        # Draw gridlines - i think maybe we don't want to draw gridlines
        # for i in range(width + 1):
        #     x1, y1 = self._map_coords(i, 0)
        #     x2, y2 = self._map_coords(i, height)
        #     self.w.create_line(x1, y1, x2, y2)
        # for i in range(height + 1):
        #     x1, y1 = self._map_coords(0, i)
        #     x2, y2 = self._map_coords(width, i)
        #     self.w.create_line(x1, y1, x2, y2)

        self.robots = None
        self.time = 0

        # Bring window to front and focus
        self.master.attributes(
            "-topmost", True
        )  # Brings simulation window to front upon creation
        self.master.focus_force()  # Makes simulation window active window
        self.master.attributes(
            "-topmost", False
        )  # Allows you to bring other windows to front

        self.master.update()

    def _map_coords(self, x, y):
        "Maps grid positions to window positions (in pixels)."
        return (
            250 + 450 * ((x - self.width / 2.0) / self.max_dim),
            250 + 450 * ((self.height / 2.0 - y) / self.max_dim) + 1,
        )

    def update(self, environment):
        "Redraws the visualization with the specified map and robot state."

        # Delete all unfurnished tiles
        for tile in self.tiles.items():
            self.w.delete(tile)

        # Redraw tiles
        self.tiles = {}
        for i in range(self.width):
            for j in range(self.height):
                x1, y1 = self._map_coords(i, j)
                x2, y2 = self._map_coords(i + 1, j + 1)

                # figure out color
                probability = environment.agent.get_probability(i, j)
                obstacle = environment.is_occupied(i, j)
                if obstacle:
                    color = 255
                else:
                    color = int(probability * 255) # TODO: this prob needs to be refined
                r = color
                g = color
                b = color
                rgb = r, g, b
                Hex = "#%02x%02x%02x" % rgb
                self.tiles[(i, j)] = self.w.create_rectangle(
                    x1, y1, x2, y2, fill=str(Hex)
                )

        # Delete all existing robots.
        if self.robots:
            for robot in self.robots:
                self.w.delete(robot)
                self.master.update_idletasks()
        # Draw new robots
        self.robots = []
        for robot in environment.agents:
            pos = robot.get_position()
            x, y = pos.get_x(), pos.get_y()
            x1, y1 = self._map_coords(x - 0.08, y - 0.08)
            x2, y2 = self._map_coords(x + 0.08, y + 0.08)
            self.robots.append(self.w.create_oval(x1, y1, x2, y2, fill="black"))

        # Update text
        self.time += 1
        self.master.update()
        time.sleep(self.delay)

    def done(self):
        "Indicate that the animation is done so that we allow the user to close the window."
        mainloop()


def test_robot_movement():
    environment = Environment()
    agents = environment.agents
    anim = RobotVisualization(environment)
    while True:
        environment.update_pos()
        anim.update(environment)
    anim.done()

test_robot_movement()