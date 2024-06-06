#
# Behavior Tree framework for A1 Behavior trees assignment.
# CS 131 - Artificial Intelligence
#
# Version 2.0.0 - copyright (c) 2023 Santini Fabrizio. All rights reserved.
#

import bt_library as btl
from ..globals import HOME_PATH, ROBOT_LOCATION, HOME_LOCATION


class FindHome(btl.Task):
    """
    Implementation of the Task "Find Home".
    """
    def run(self, blackboard: btl.Blackboard) -> btl.ResultEnum:

        self.print_message("Looking for a home")
        home_path = []
        init_robot_loc = blackboard.get_in_environment(ROBOT_LOCATION, [0, 0])
        robot_loc = init_robot_loc.copy()
        home_loc = blackboard.get_in_environment(HOME_LOCATION, [5, 5])
        
        while robot_loc != home_loc:
            if robot_loc[0] < home_loc[0]:
                home_path.append("Right")
                robot_loc[0] += 1
            elif robot_loc[0] > home_loc[0]:
                home_path.append("Left")
                robot_loc[0] -= 1
            if robot_loc[1] < home_loc[1]:
                home_path.append("Up")
                robot_loc[1] += 1
            elif robot_loc[1] > home_loc[1]:
                home_path.append("Down")
                robot_loc[1] -= 1
                
        blackboard.set_in_environment(HOME_PATH, home_path)
        blackboard.set_in_environment(ROBOT_LOCATION, init_robot_loc)


        return self.report_succeeded(blackboard)
