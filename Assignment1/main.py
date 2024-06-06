#
# Behavior Tree framework for A1 Behavior trees assignment.
# CS 131 - Artificial Intelligence
#
# version 2.0.0 - copyright (c) 2023 Santini Fabrizio. All rights reserved.
#

import bt_library as btl

from bt.behavior_tree import tree_root
from bt.globals import BATTERY_LEVEL, GENERAL_CLEANING, SPOT_CLEANING, DUSTY_SPOT_SENSOR, HOME_PATH, ROBOT_LOCATION, HOME_LOCATION
from random import random


# Main body of assignment - sets default values for environment, and runs simulation
def main():
    # Set default values for enviornment
    current_blackboard = btl.Blackboard()
    current_blackboard.set_in_environment(BATTERY_LEVEL, 29)
    current_blackboard.set_in_environment(SPOT_CLEANING, False)
    current_blackboard.set_in_environment(GENERAL_CLEANING, True)
    current_blackboard.set_in_environment(DUSTY_SPOT_SENSOR, False)
    # Array of strings - being either "Right", "Left", "Up", or "Down"
    current_blackboard.set_in_environment(HOME_PATH, [])
    # Home and Robot locations are x,y coordinates on a 9x9 Grid
    current_blackboard.set_in_environment(HOME_LOCATION, [5, 5])
    current_blackboard.set_in_environment(ROBOT_LOCATION, [0, 0])

    done = False
    while not done:
        # Each cycle in this while-loop is equivalent to 1 second time

        # Battery decreases by 1% each second 
        current_blackboard.set_in_environment(BATTERY_LEVEL, current_blackboard.get_in_environment(BATTERY_LEVEL, 0) - 1)

        # Simulate the response of the dusty spot sensor
        if random() < 0.1:
            current_blackboard.set_in_environment(DUSTY_SPOT_SENSOR, True)
        else:
            # Only want to consider it for current square - no "memory" to previous dusty spots
            current_blackboard.set_in_environment(DUSTY_SPOT_SENSOR, False)
        # Evaluate the tree
        done = run(current_blackboard)
        # Print extra information - the current location of the robot
        print(f'current location: {current_blackboard.get_in_environment(ROBOT_LOCATION, None)}')

# Evaluates the tree and receives user input
def run(blackboard: btl.Blackboard) -> bool:
    result = tree_root.run(blackboard)
    #   - Simulate user input commands
    if result != btl.ResultEnum.RUNNING and blackboard.get_in_environment(SPOT_CLEANING, False) == False and blackboard.get_in_environment(GENERAL_CLEANING, False) == False:
        try:
            cmd = input('The Robot is finshed its current mode - would you like to specify another? Enter "spot" for spot cleaning, "general" for general cleaning, or "quit" to quit:\n')
        except KeyboardInterrupt:
            cmd = "quit"
        if cmd == "spot":
            blackboard.set_in_environment(SPOT_CLEANING, True)
        elif cmd == "general":
            blackboard.set_in_environment(GENERAL_CLEANING, True)
        elif cmd == "quit":
            print("Goodbye!")
            return True
        else:
            cmd = input('Your command was not valid, please try again. Valid commands are "spot", "general", and "quit"\n')
    return False

if __name__ == "__main__":
    main()
