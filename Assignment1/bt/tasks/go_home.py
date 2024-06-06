#
# Behavior Tree framework for A1 Behavior trees assignment.
# CS 131 - Artificial Intelligence
#
# Version 2.0.0 - copyright (c) 2023 Santini Fabrizio. All rights reserved.
#

import bt_library as btl
from ..globals import HOME_PATH, ROBOT_LOCATION, SPOT_CLEANING, GENERAL_CLEANING


class GoHome(btl.Task):
    """
    Implementation of the Task "Go Home".
    """
    def run(self, blackboard: btl.Blackboard) -> btl.ResultEnum:
        self.print_message("Going home")
        home_path = blackboard.get_in_environment(HOME_PATH, [])
        robot_loc = blackboard.get_in_environment(ROBOT_LOCATION, [0,0])
        instruction = home_path.pop(0)
        if instruction == 'Up':
            robot_loc[1] += 1
        elif instruction == 'Down':
            robot_loc[1] -= 1
        elif instruction == 'Left':
            robot_loc[0] -= 1
        elif instruction == 'Right':
            robot_loc[0] += 1
        
        blackboard.set_in_environment(HOME_PATH, home_path)
        blackboard.set_in_environment(ROBOT_LOCATION, robot_loc)
        if not home_path:
            return self.report_succeeded(blackboard)
        else:
            return self.report_running(blackboard)

