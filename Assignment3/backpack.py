# Casey Owen
# CS131
# Assignment 3, the Knapsack Problem
'''
Implementation of the Backpack Class - a representation of the backpack in the Knapsack Problem. Contains information about the backpack and genome, and defines the fitness function, and
'''
class Backpack():
    # Genome is a list of 1's and 0's checking if that box is in the backpack or not
    # Available boxes is list of tuples - first value is weight, second is value
    
    def __init__(self, genome:list[int], avail_boxes:list[tuple[int]], 
    capacity:int) -> None:
        '''
        Purpose: 
            Constructor for the Backpack class used for representing possible backpacks (solutions) to the Knapsack Problem.
        Inputs: 
            -genome: A list of 1's and 0's that represent whether that box is present in the backpack. Has the same lenght as avail_boxes
            -avail_boxes: A list of tuples of the available boxes to be put in each the backpack. The tuple has two integers - the first represents the weight of the box, and the second is the importance value of the box
        Outputs: none
        '''
        self.avail_boxes = avail_boxes
        self.genome = genome
        self.capacity = capacity
        if len(genome) != len(avail_boxes):
            raise ValueError("The genome must be the same length as avail_boxes")

    @property
    def weight(self) -> int:
        '''
        Purpose: 
            Getter function for the weight property of the backpack. Weight represents the sum of weights of all boxes that are in the backpack
        Inputs: none
        Outputs: The weight value of the backpack
        '''
        return sum(self.avail_boxes[idx][0] for idx, contains_box in enumerate(self.genome) if contains_box)

    @property
    def importance(self) -> int:
        '''
        Purpose: 
            Getter function for the importance property of the backpack. Weight represents the sum of importances of all boxes that are in the backpack
        Inputs: none
        Outputs: The importance value of the backpack
        '''
        return sum(self.avail_boxes[idx][1] for idx, contains_box in enumerate(self.genome) if contains_box)
    
    @property
    def fitness(self) -> int:
        '''
        Purpose: 
            Getter function for the fitness property of the backpack. Fitness is a function which is higher for backpacks with higher importance values, and rewards backpacks with space left underneath the capacity, and punishes those that have gone overcapacity.
        Inputs: none
        Outputs: The fitness value of the backpack
        '''
        # If still have plenty of space left, give small bonus for each amount of extra weight left, room to put things in that mutations might fill
        if self.weight < (.8*self.capacity):
            return self.importance + .1*((.8*self.capacity) - self.weight)
        elif self.weight <= self.capacity:
            return self.importance
        else:
            # Small discrete puncishment (to ensure final best solution is very unlikely to be above weight), with gradual slope down as weight exceeds capacity, so as to still give credit to genomes that are close to the solution, but slightly above. Using a min of 0 so that there are no negative values and they can be used for weighting more easily
            return max(0, self.importance - .05*self.capacity - (0.25*(self.weight - self.capacity)))

    @property
    def genome(self) -> list[int]:
        '''
        Purpose: 
            Getter function for the genome of the backpack.
        Inputs: none
        Outputs: The genome of the backpack
        '''
        return self._genome
    
    @genome.setter
    def genome(self, value):
        '''
        Purpose: 
            Setter function for the genome of the backpack.
        Inputs: 
            value: The new genome for the backpack
        Outputs: none.
        '''
        if len(value) != len(self.avail_boxes):
            raise ValueError("The length of the genome to set must match the length of the available boxes")
        self._genome = value

    def __lt__(self, other) -> bool:
        '''
        Purpose: 
            Sets the comparison operator for '<' to be based on fitness. This is useful for sorting the nodes by fitness in the search algorithms.
        Inputs: 
            -other: The Backpack being compared to
        Outputs: 
            -result: boolean result of the comparison
        '''
        return self.fitness < other.fitness

    def print_backpack(self) -> None:
        '''
        Purpose: 
            Prints the backpack, including information about the genome, weight, importance value, and fitness.
        Inputs: none
        Outputs: none
        '''
        print(f"Genome: {self.genome}")
        contained_boxes = [box for idx, box in enumerate(self.avail_boxes) if self.genome[idx]]
        print(f"Boxes in backpack: {contained_boxes}")
        print(f"Weight: {self.weight}")
        print(f"Importance value: {self.importance}")
        print(f"Fitness: {self.fitness}")