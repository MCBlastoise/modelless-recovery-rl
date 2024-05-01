import numpy as np

class Environment:
    def __init__(self, occupancy_data: str | np.ndarray):
        if type(occupancy_data) is str:
            self.occupancy_grid = Environment.read_from_file(occupancy_data)
        else:
            self.occupancy_grid = occupancy_data


    def read_from_file(image_filename):
        pass

    def is_occupied(self, coords: tuple[int, int]) -> bool:
        return bool(self.occupancy_grid[*coords])