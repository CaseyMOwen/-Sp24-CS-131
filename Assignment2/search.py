# Casey Owen
# CS131
# Assignment 2, the Pancake Problem
'''
Searches for the solution to the Pancake Problem, using either A* with the gap heurstic, or Uniform Cost Search (UCS).
'''

from pancakestack import PancakeStack
from frontier import Frontier
from searchnode import SearchNode
import copy

class Search():
    def find_solution(self, pancake_stack:PancakeStack, heuristic = None) -> tuple[list[int], tuple[int, int]]:
        '''
        Purpose: 
            Uses the A*/UCS algorithm to find a solution to the pancake problem given a particular PancakeStack
        Inputs: 
            -pancake_stack: The pancake stack for which to find the solution
            -heursitic: The heurstic function to use to solve the problem. If called as 'gap', will use the gap heuristic with an A* algorithm. Otherwise, no heursitic is used, which reduces the algorithm to Uniform Cost Search (UCS)
        Outputs:       
            -solution:  A list of integers representing the solution of the problem - the number of pancakes that should be flipped at each step, in order
            -additional_stats: A tuple of two integers, the total number of nodes visited, and the cumulative number of nodes added to the frontier to find the solution
        '''
        root = SearchNode(pancake_stack, 0, 0, None, None)
        frontier = Frontier(root)
        while not frontier.is_empty():
            node = frontier.pop()
            if node.get_stack().is_in_order():
                solution = self.get_path_to_root(node)
                return solution, (frontier.get_nodes_visited(), frontier.get_cum_frontier_size())
            children = self.expand_node(node, heuristic)
            for child in children:
                # Push all, frontier class deals with priority queue/duplicates
                frontier.push(child)
            frontier.mark_visited(node)
        return None, (0, 0)
    
    def expand_node(self, node: SearchNode, heuristic: str) -> list[SearchNode]:
        '''
        Purpose: 
            Finds all possible children of a given node, given all possible actions that can be taken from that state, returns them as a list of nodes with properties consistent with the algorithm invariant
        Inputs: 
            -node: The node for which to find the children of
            -heursitic: The heurstic function in use to use to solve the problem. This will be used to update the node childrens hn value.
        Outputs:       
            -children:  A list of nodes representing the original nodes children.
        '''
        node_stack = node.get_stack()
        children = []
        # 10 children - all possible places it can be flipped
        for num_to_flip in range(1, 11):
            new_stack = copy.deepcopy(node_stack)
            new_stack.flip(num=num_to_flip)
            # Cost (gn) is number of flips to get to this state
            new_gn = node.get_gn() + 1
            if heuristic == 'gap':
                new_hn = self.gap_heuristic(new_stack)
            else: #heurstic = None
                # hn = 0 is equivalent to UCS
                new_hn = 0
            new_fn = new_gn + new_hn
            new_node = SearchNode(new_stack, new_fn, new_gn, node, num_to_flip)
            children.append(new_node)
        return children
    
    def get_path_to_root(self, node: SearchNode) -> list[int]:
        '''
        Purpose: 
            Get the path to the root of the search tree from a given node as a list of integers indicating what number of pancakes was flipped to get from the parent to the child
        Inputs: 
            -node: The node for which to find the path to the root from
        Outputs:       
            -path:  A list of integers indicating what number of pancakes was flipped to get from the parent to the child
        '''
        if node.get_parent() == None:
            return []
        else:
            path = self.get_path_to_root(node.get_parent())
            path.append(node.get_num_flipped())
            return path

    def gap_heuristic(self, stack: PancakeStack) -> int:
        '''
        Purpose: 
            Calculates the gap heurstic value for a given pancake stack (Helmert 2010)
        Inputs: 
            -stack: The PancakeStack for which to calculate the gap heuristic
        Outputs:       
            -result: The resulting heurstic value
        '''
        stack_list = stack.get_stack_list()
        result = 0
        num_cakes = len(stack_list)
        for i, cake_size in enumerate(stack_list):
            if i == num_cakes - 1:
                return result
            elif abs(cake_size - stack_list[i+1]) > 1:
                result += 1


