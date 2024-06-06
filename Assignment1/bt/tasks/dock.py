#
# Behavior Tree framework for A1 Behavior trees assignment.
# CS 131 - Artificial Intelligence
#
# Version 2.0.0 - copyright (c) 2023 Santini Fabrizio. All rights reserved.
#

import bt_library as btl
from ..globals import BATTERY_LEVEL

class Dock(btl.Task):
    """
    Implementation of the Task "Dock".
    """
    def run(self, blackboard: btl.Blackboard) -> btl.ResultEnum:
        self.print_message("Docking")
        # Assume docking recharges battery
        blackboard.set_in_environment(BATTERY_LEVEL, 100)
        return self.report_succeeded(blackboard)
