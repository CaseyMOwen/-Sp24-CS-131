# Casey Owen
# CS131
# Assignment 2, the Pancake Problem
'''
Implementation of the SearchNode Class - a node in the search space of the Pancake Problem. Contains information about the state of the pancake stack, and backward and total costs.
'''

from pancakestack import PancakeStack

class SearchNode():
    def __init__(self, stack: PancakeStack, fn: int, gn: int, parent, num_flipped: int) -> None:
        '''
        Purpose: 
            SearchNode Constructor for a node used for searching for solutions to the Pancake Problem.
        Inputs: 
            -stack: The PancakeStack of the node.
            -fn: The total cost function of the node.
            -gn: The backward cost function of the node - how many flips it took from the root to get to this node
            -parent: The parent SearchNode
            -num_flipped: The number of pancakes flipped to get from the parent node to this node
        Outputs: none
        '''
        self.stack = stack
        self.fn = fn
        self.parent = parent
        self.num_flipped = num_flipped
        self.gn = gn

    def get_parent(self):
        '''
        Purpose: 
            Getter function for the parent of a SearchNode
        Inputs: none
        Outputs: 
            The parent SearchNode of this node.
        '''
        return self.parent

    def get_gn(self) -> int:
        '''
        Purpose: 
            Getter function for the backward cost - how many flips it took to get from the root to this node.
        Inputs: none
        Outputs: 
            The backward cost of this node.
        '''
        return self.gn

    def get_num_flipped(self) -> int:
        '''
        Purpose: 
            Getter function for num_flipped - the number of pancakes flipped to get from the parent node to this node
        Inputs: none
        Outputs: 
            num_flipped
        '''
        return self.num_flipped

    def __lt__(self, other):
        '''
        Purpose: 
            Sets the comparison operator for '<' to be based on fn. This is useful for sorting the nodes by total cost in search algorithms.
        Inputs: 
            -other: The SearchNode being compared to
        Outputs: none
        '''
        return self.fn < other.fn
    
    def get_stack(self):
        '''
        Purpose: 
            Getter function for the nodes PancakeStack
        Inputs: none
        Outputs: 
            The node's PancakeStack
        '''
        return self.stack
    
    # def get_fn(self):
    #     '''
    #     Purpose: 
    #         Getter function for the total cost - backward cost + forward cost
    #     Inputs: none
    #     Outputs: 
    #         The total cost of this node.
    #     '''
    #     return self.fn