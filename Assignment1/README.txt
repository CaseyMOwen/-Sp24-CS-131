Casey Owen, cowen03
CS131
Assignment 1
Vacuum Cleaning Robot's Roamings

Requirements to Run:
-Ensure the "random" library is installed
-Run main.py

Assumptions:

-Each call happens at one second intervals

-The battery drains at 1% per second

-The battery is instantly fully charged upon docking

-the robot exists on a 9x9 grid, and can move one step per second

-The robot cannot move diagonally

-The robot location is given by variable ROBOT_LOCATION, which by default starts as 0,0

-The dock is specified as HOME_LOCATION, which by default is at 5,5

-The home location is not special in any other way - it can still be a space to be cleaned

-It takes 1 second to dock once the robot arrives at the dock
-When the robot is performing task "Clean Floor" it will move 1 step in a random direction every time the task is called, to simulate moving to a new area to clean 

-Implementation of HOME_PATH was changed - it now is a list of strings (rather than a single string with spaces), with possible values "Left", "Right", "Up" and "Down", which give the path the robot should take to get home when evaluated from index 0 to the end

-Sequence was implemented to handle running states the same as selection (not priority) - if a node returns running, then it will pick up where it left off on the next cycle and try that node again, rather than start from the left-most node every time. I believe this is the "correct" way to do it but I'm stating it as an assumption since I don't think it was explicitly stated in class.

-Will ask the user for an input every time it reaches a state where both GENERAL_CLEANING and SPOT_CLEANING are false - that is, it has finished it's current instructions. The user can either give the robot a new instruction, or finish the program.

-There is a 10% chance that any individual square is detected as a dusty spot

-The robot has no memory for dusty spots - it will not go back to clean dusty spots found in previous timesteps

-Clean Floor continues until failure - when there is nothing left to clean.  Each individual timestep has a 5% chance of entering a state where there is nothing left to clean.

-If the robot is in the middle of a task, and goes home because its battery is too low, it will pick up the task where it left off after docking. This is probably not good behavior for a real vaccuum robot - it would keep cleaning a dusty spot, even though it is now in a new location (and the spot is somewhere else). An alternative strategy might be to navigate back to the previous spot and pick up cleaning there, but this would require a different behahviour tree than specified in the assignment with new tasks. I also considered clearing the GENERAL_CLEANING and SPOT_CLEANING commands when the battery runs out, but this would violate the specification that those commands can only change state when they are completed.