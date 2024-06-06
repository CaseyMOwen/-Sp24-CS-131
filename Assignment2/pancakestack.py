# Casey Owen
# CS131
# Assignment 2, the Pancake Problem
'''
A PancakeStack object represents a pancake stack for the Pancake Problem. Includes useful methods such as flipping the stack at a specified location and pretty printing the stack. 
'''

import random

class PancakeStack():

    # If stack list is specified, it the stack is initialized deterministically
    # If random flips is specified, the stack is initialized as 1-10, then that many flips called in random locations 
    def __init__(self, stack_list:list[int] = None, random_flips:int = None) -> None:
        '''
        Purpose: 
            PancakeStack Constructor for a stack of 10 pancakes - can be used one of three ways, with either stack_list specified, random_flips specified, or neither. If stack_list is specified, the object is created deterministically. If random_flips is specified, a stack is generated randomly that is at most random_flips steps from being in order. Otherwise, it is created completely randomly.
        Inputs: 
            -stack_list: The list to use when the stack is created deterministically. Must be a list of 10 numbers, from 1-10.
            -random_flips: The number of steps away the solution is, at most, in a random stack
        Outputs: none
        '''
        # First indexes are top of stack, last are bottom
        if stack_list is not None:
            self.stack = stack_list
        elif random_flips is not None:
            self.stack = list(range(1, 11))
            for i in range(random_flips):
                self.flip(random.randint(0,9))
        else:
            self.stack = list(range(1, 11))
            random.shuffle(self.stack)
        
        # # Use a joined string of the stack list to uniquely identify it
        # self.stack_id = ",".join(str(i) for i in self.stack)
    
    def print_state(self, step:int, flipnext:int = None) -> None:
        '''
        Purpose: 
            Pretty-prints the pancakestack, including information about the step the solution is on, and how many pancakes will be flipped next.
        Inputs: 
            -step: The step the solution is on.
            -flipnext: How many pancakes will be flipped next.
        Outputs: none
        '''
        if step == None:
            print(("\nDone! " + " ").ljust(30,'-'))
        elif step == 0:
            print(("\nInitial State: " + " ").ljust(30,'-'))
        else:
            print(("\nStep " + str(step) + ": ").ljust(30,'-'))
        for i, cake_size in enumerate(self.stack):
            xs = "x" * cake_size
            centered = xs.center(20, ' ') + "(" + str(cake_size) + ")"
            print(centered)
            if i + 1 == flipnext and step != 0:
                print("   " + "-"*13 + "     <--- Flip Here")
            
    
    def flip(self, num:int) -> None:
        '''
        Purpose: 
            Flips a given number of pancakes, from the top
        Inputs: 
            -num: The number of pancakes to flip off the top of the stack.
        Outputs: none
        '''
        top = self.stack[:num]
        bottom = self.stack[num:]
        top.reverse()
        self.stack = top + bottom

    def get_stack_id(self) -> str:
        '''
        Purpose: 
            Gets a hashable unique identifier for a given pancake stack configuration
        Inputs: none
        Outputs: 
            stack_id: a string unique identifer for a given pancake stack configuration
        '''
        return ",".join(str(i) for i in self.stack)

    def is_in_order(self) -> bool:
        '''
        Purpose: 
            Checks if the pancake stack is in order
        Inputs: none
        Outputs: 
            result: a boolean indicating if the stack is in order
        '''
        if self.stack == list(range(1, 11)):
            return True
        else:
            return False

    def get_stack_list(self) -> list[int]:
        '''
        Purpose: 
            Getter function of the list representation of the pancake stack.
        Inputs: none
        Outputs: 
            stack_list: The list representation of the pancake stack.
        '''
        return self.stack