# Casey Owen
# CS131
# Assignment 2, the Pancake Problem
'''
Implementation of the Frontier Class - the frontier of the search space of the Pancake Problem. Handles logic of pushing/popping nodes onto the frontier, and keeping track of when nodes have been visited or not.
'''

from searchnode import SearchNode
import heapq


class Frontier():
    def __init__(self, root: SearchNode) -> None:
        '''
        Purpose: 
            Frontier Constructor for a search frontier used for searching for solutions to the Pancake Problem. Keeps track of what nodes have been visited.
        Inputs: 
            -root: The root node of the search tree.
        Outputs: none
        '''
        # Keeps two copies of frontier - a priority queue to quickly check for priority, and a dictionary to quickly check membership
        self.frontier_pq = [root]
        heapq.heapify(self.frontier_pq)
        self.frontier_dict = {root.get_stack().get_stack_id(): root}
        self.visited = set()
        self.cum_frontier_size = 0

    def push(self, node: SearchNode) -> None:
        '''
        Purpose: 
            Pushes a node onto the frontier. If the node has already been visited, nothing happens. If the stack configuration exists on the frontier already, but at a higher fn than the the current node, the higher fn node is replaced. Otherwise if the existing node has a lower fn then it is not replaced and the node to be pushed is discarded.
        Inputs: 
            -node: The node to be pushed.
        Outputs: none
        '''
        stack_id = node.get_stack().get_stack_id()
        if stack_id in self.visited:
            return
        elif stack_id in self.frontier_dict:
            other_node = self.frontier_dict[stack_id]
            # Compares fns
            if other_node < node:
                return
            else:
                self.frontier_dict[stack_id] = node
                # Replacing by value in the list is a slow operation, but this case should not happen too often - two search tree branches converging to the same stack configuration from different directions, AND the one found later is better
                self.frontier_pq[self.frontier_pq.index(other_node)] = node
                heapq.heapify(self.frontier_pq)
                self.cum_frontier_size += 1
        else: # Not visited or in frontier
            heapq.heappush(self.frontier_pq, node)
            self.frontier_dict[stack_id] = node
            self.cum_frontier_size += 1

    def pop(self) -> SearchNode:
        '''
        Purpose: 
            Pops the highest priority node from the frontier and returns it
        Inputs: none
        Outputs: 
            -node: The node being popped.
        '''
        node = heapq.heappop(self.frontier_pq)
        stack_id = node.get_stack().get_stack_id()
        del self.frontier_dict[stack_id]
        return node


    def mark_visited(self, node: SearchNode) -> None:
        '''
        Purpose: 
            Marks a node as having already been visted by the search problem
        Inputs: 
            -node: The node to be marked visted
        Outputs: none 
        '''
        stack_id = node.get_stack().get_stack_id()
        self.visited.add(stack_id)

    def get_nodes_visited(self) -> int:
        '''
        Purpose: 
            Getter function for the total number of nodes that have been visited.
        Inputs: none
        Outputs: 
            num_nodes_visited: The number of nodes that have been visited 
        '''
        return len(self.visited)
    
    def get_cum_frontier_size(self) -> int:
        '''
        Purpose: 
            Getter function for the total number of unique nodes that have ever been added to the frontier.
        Inputs: none
        Outputs: 
            cum_frontier_size: The total number of unique nodes that have ever been added to the frontier. 
        '''
        return self.cum_frontier_size
    
    def is_empty(self) -> bool:
        '''
        Purpose: 
            Checks if the frontier is currently empty
        Inputs: none
        Outputs: 
            is_empty: A boolean indicating if the frontier is currently empty 
        '''
        return not self.frontier_pq 