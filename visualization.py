import math
import time
from tkinter import *
from environment import Environment
from agent import Agent
import numpy as np

class RobotVisualization:
    """
    Visualization of a Robot simulation.
    """

    def __init__(self, environment, delay=0.00):
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
        self.w = Canvas(self.master, width=1000, height=500)
        self.w.pack()
        self.master.update()

        # Draw a backing and lines
        x1, y1 = self._map_coords(0, 0)
        x2, y2 = self._map_coords(width, height)
        self.w.create_rectangle(x1, y1, x2, y2, fill="white", outline="white")

        # Draw black/white tiles 
        self.tiles = set()
        for i in range(width):
            for j in range(height):
                x1, y1 = self._map_coords(i, j)
                x2, y2 = self._map_coords(i + 1, j + 1)
                if not environment.is_occupied((i, j)):
                    self.tiles.add(self.w.create_rectangle(
                        x1, y1, x2, y2, fill="black", outline="black"
                    ))
                else:
                    self.tiles.add(self.w.create_rectangle(
                        x1, y1, x2, y2, fill="white", outline="white"
                    ))

        # Draw black/white tiles 
        self.tiles = set()
        for i in range(width):
            for j in range(height):
                x1, y1 = self._map_coords(i, j)
                x2, y2 = self._map_coords(i + 1, j + 1)
                if not environment.is_occupied((i, j)):
                    self.tiles.add(self.w.create_rectangle(
                        x1 + 500, y1, x2 + 500, y2, fill="black", outline="black"
                    ))
                else:
                    self.tiles.add(self.w.create_rectangle(
                        x1 + 500, y1, x2 + 500, y2, fill="white", outline="white"
                    ))

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

        # print("Pre-delete")

        # print(len(self.w.find_all()))

        # print(len(self.w))

        # Delete all unfurnished tiles
        for obj in self.tiles:
            self.w.delete(obj)

        # print("Pre-draw")

        # Redraw tiles
        self.tiles = set()
        # first map

        for i in range(self.width):
            for j in range(self.height):
                x1, y1 = self._map_coords(i, j)
                x2, y2 = self._map_coords(i + 1, j + 1)

                # figure out color
                probability = environment.agents[0].get_probability_obstacle((i, j))
                obstacle = environment.is_occupied((i, j))
                if obstacle and probability < 0.5:
                    r = 255
                    g = 255
                    b = 255
                else:
                    color = int(probability * 255)
                    r = color
                    g = 255 - color
                    b = 0
                
                rgb = r, g, b
                Hex = "#%02x%02x%02x" % rgb
                self.tiles.add(self.w.create_rectangle(
                    x1, y1, x2, y2, fill="black", outline="black"
                ))
                self.tiles.add(self.w.create_rectangle(
                    x1, y1, x2, y2, fill=str(Hex), outline=str(Hex)
                ))

        #second map
        if len(environment.agents) > 1:
            for i in range(self.width):
                for j in range(self.height):
                    x1, y1 = self._map_coords(i, j)
                    x2, y2 = self._map_coords(i + 1, j + 1)

                    # figure out color
                    probability = environment.agents[1].get_probability_obstacle((i, j))
                    obstacle = environment.is_occupied((i, j))
                    if obstacle:
                        r = 255
                        g = 255
                        b = 255
                    else:
                        color = int(probability * 255)
                        r = color
                        g = 255 - color
                        b = 0
                    
                    rgb = r, g, b
                    Hex = "#%02x%02x%02x" % rgb
                    self.tiles.add(self.w.create_rectangle(
                        x1+500, y1, x2+500, y2, fill="black", outline="black"
                    ))
                    self.tiles.add(self.w.create_rectangle(
                        x1+500, y1, x2+500, y2, fill=str(Hex), outline=str(Hex)
                    ))

        # print("Pre-robot deleting")
        # Delete all existing robots.
        if self.robots:
            for robot in self.robots:
                self.w.delete(robot)
                self.master.update_idletasks()
        
        # print("Pre-robot draw")
        
        # Draw new robots
        self.robots = []
        for robot_idx, robot in enumerate(environment.agents):
            x, y = robot.pos
            x1, y1 = self._map_coords(x - 0.6, y - 0.6)
            x2, y2 = self._map_coords(x + 0.6, y + 0.6)
            if robot_idx == 0:
                self.robots.append(self.w.create_oval(x1, y1, x2, y2, fill="blue", outline="blue"))
                if len(environment.agents) == 2:
                    self.robots.append(self.w.create_oval(x1+500, y1, x2+500, y2, fill="gray", outline="gray"))
            else:
                self.robots.append(self.w.create_oval(x1+500, y1, x2+500, y2, fill="blue", outline="blue"))
                self.robots.append(self.w.create_oval(x1, y1, x2, y2, fill="gray", outline="gray"))
            

        self.master.update()
        if self.delay != 0:
            time.sleep(self.delay)

    def done(self):
        "Indicate that the animation is done so that we allow the user to close the window."
        mainloop()
