#
# Behavior Tree framework for A1 Behavior trees assignment.
# CS 131 - Artificial Intelligence
#
# Version 2.0.0 - copyright (c) 2023 Santini Fabrizio. All rights reserved.
#

import bt_library as btl
from random import random
from bt.globals import ROBOT_LOCATION

class CleanFloor(btl.Task):
    """
    Implementation of the Task "Clean Floor".
    """
    def run(self, blackboard: btl.Blackboard) -> btl.ResultEnum:
        self.print_message("Clean Floor")
        moved = False
        rob_loc = blackboard.get_in_environment(ROBOT_LOCATION, [0,0])
        # Goes 1 step in a random direction at each time step to clean the floor
        while not moved:
            rand = random()
            # move left
            if rand < 0.25 and rob_loc[0] > 0:
                rob_loc[0] -= 1
                moved = True
            # move right
            elif rand < 0.5 and rob_loc[0] < 9:
                rob_loc[0] += 1
                moved = True
            # move down
            elif rand < 0.75 and rob_loc[1] > 0:
                rob_loc[1] -= 1
                moved = True
            # move up
            elif rob_loc[1] < 9:
                rob_loc[1] += 1
                moved = True
        blackboard.set_in_environment(ROBOT_LOCATION, rob_loc)
        if random() < .05:
            # Nothing left to clean
            return self.report_failed(blackboard)
        else:
            # Still cleaning floor
            return self.report_succeeded(blackboard)

