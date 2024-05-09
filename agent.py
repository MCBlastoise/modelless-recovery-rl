class Agent:
    def __init__(self, initial_pdm):
        self.pdm = initial_pdm # numpy array
        self.x = 0
        self.y = 0
        self.epsilon = 0.5
    
    def next_action(self):
        pass

    def is_safe(self, coords: tuple[int, int]) -> bool:
        """
        returns boolean describing whether the given coord is above the epsilon safety value
        """
        return self.pdm[i, j] > self.epsilon

    def recovery_step(self) -> tuple:
        """
        returns recovery action as a tuple, defined as the safest place to go from the current location
        """
        safest = None
        safest_prob = 1

        for r in range(-1, 2):
            for c in range(-1, 2):
                if r==0 and c==0:
                    continue
                else:
                    next_x = c + self.x
                    next_y = r + self.y
                    if next_y >= 0 and next_y < self.pdm.shape[0] and next_x >= 0 and next_x < self.pdm.shape[1]:
                        pnext_yob_violation = self.pdm[next_y, next_x]
                        if prob_violation < safest_prob or safest is None:
                            safest = (next_y, next_x)
                            safest_prob = prob_violation 

        return safest