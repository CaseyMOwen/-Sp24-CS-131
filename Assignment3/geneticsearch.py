# Casey Owen
# CS131
# Assignment 3, the Knapsack Problem
'''
Searches for the solution to the Knapsack Problem, using a genetic algorithm, with fringe operations including mutation (single or multi point) and crossover (single or multi point).
'''

from backpack import Backpack
import random
import copy
import numpy

class GeneticSearch():
    def __init__(self, avail_boxes:list[tuple[int]], population_size:int, capacity: int) -> None:
        '''
        Purpose: 
            Constructor for the GeneticSearch class used for searching for solutions to the Knapsack Problem.
        Inputs: 
            -avail_boxes: A list of tuples of the available boxes to be put in each the backpack. The tuple has two integers - the first represents the weight of the box, and the second is the importance value of the box
            -population_size: The size of the population to use in the genetic algorithm, which stays constant throughout the process
            -capacity: The weight capacity of each backpack
        Outputs: none
        '''
        self.avail_boxes = avail_boxes
        self.population = [None]*population_size
        self.capacity = capacity
        self.population_size = population_size
        self.init_population()

    def init_population(self) -> None:
        '''
        Purpose: 
            Initializes the population of backpacks. Each initial backpack has a genome where each individual element has a 50% chance of being a 1, and a 50% chance of being a 0
        Inputs: none
        Outputs: none
        '''
        genome_length = len(self.avail_boxes)
        for i in range(len(self.population)):
            new_genome = random.choices((0,1),k=genome_length)
            self.population[i] = Backpack(new_genome, self.avail_boxes, self.capacity)

    def search(self, max_iter, max_plateau_steps) -> tuple[Backpack, list[list[int]]]:
        '''
        Purpose: 
            Searches for solution a to the Knapsack Problem. Exits after max_iter iterations, or exits early if the best solution does not improve for max_plateau_steps consecutive iterations.
        Inputs: 
            -num_iter: The number of iterations to go through to solve the problem 
        Outputs: 
            -best_backpack: The resulting best backpack found
            -pop_fitness_by_iter: A list of lists of the population fitness of all members of the population, one entry for each iteration
        '''
        iter = 0
        pop_fitness_by_iter = []
        plateau_steps = 0
        best_fitness = 0
        while iter < max_iter:
            self.cull(0.5)
            pop_fitness = self.pop_fitness()
            new_population = []
            while len(new_population) < self.population_size:
                for child in self.weighted_reproduce(pop_fitness):
                    mutated_child = self.random_mutate(child)
                    new_population.append(mutated_child)
                    if len(new_population) == self.population_size:
                        break
            self.population = new_population
            pop_fitness_by_iter.append(self.pop_fitness())
            iter += 1
            new_best_fitness = max(self.population).fitness
            if new_best_fitness == best_fitness:
                plateau_steps += 1
            else:
                plateau_steps = 0
            best_fitness = new_best_fitness
            if plateau_steps >= max_plateau_steps:
                break
        return max(self.population), pop_fitness_by_iter
    
    def pop_fitness(self) -> list[int]:
        '''
        Purpose: 
            Generates a list of the fitnesses of a population.
        Inputs: none
        Outputs: 
            -pop_fitness: a list of fitness values for each member of the population, where the value at index i represents the fitness of backpack at index i of self.population
        '''
        return [backpack.fitness for backpack in self.population]

    def weighted_reproduce(self, pop_fitness: list[int]) -> tuple[Backpack, Backpack]:
        '''
        Purpose: 
            Reproduces two parents, drawn from the population weighted by their fitness without replacement, and crosses them over two create two children. The location number of places they are crossed over is randomized.
        Inputs: 
            -pop_fitness: a list of fitness values for each member of the population, where the value at index i represents the fitness of backpack at index i of self.population
        Outputs: 
            -child1: The first created backpack
            -child2: The second created backpack 
        '''
        parents = numpy.random.choice(a=self.population, size=2, replace=False, p=[x/sum(pop_fitness) for x in pop_fitness])
        num_crossover_points = random.choices((3,2,1,0), weights=(1,2,4,8), k=1)[0]
        genome_length = len(parents[0].genome)
        points = random.sample(list(range(genome_length)), k=num_crossover_points)
        return self.crossover(parents[0], parents[1], points)
    
    def random_mutate(self, backpack: Backpack) -> Backpack:
        '''
        Purpose: 
            Mutate a backpack with a probability of 0.05. If the backpack is chosen to be mutated, the number of mutations is chosen randomly, with a bias towards fewer mutations. 
        Inputs: 
            -backpack: The backpack to possible mutate
        Outputs: 
            -result: The resulting backpack. If the backpack is not mutated, the original backpack is returned.
        '''
        if random.random() < .05:
            num_mutations = random.choices((5,4,3,2,1), weights=(1,2,4,8,16), k=1)[0]
            return self.mutate(backpack, num_mutations)
        else:
            return backpack

    def mutate(self, backpack:Backpack, num_genes:int) -> Backpack:
        '''
        Purpose: 
            Flips a given number of bits within a backpack genome. The locations of the bits to flip are chosen randomly.
        Inputs: 
            -backpack: The backpack to mutate
            -num_genes: The number of genes to mutate
        Outputs: 
            -result: The resulting backpack
        '''
        if num_genes > len(backpack.genome):
            raise ValueError("Cannot change more genes than the genome contains")
        mutate_order = list(range(len(backpack.genome)))
        random.shuffle(mutate_order)
        mutated_backpack = copy.deepcopy(backpack)
        for mutated_count, idx in enumerate(mutate_order):
            if mutated_count < num_genes:
                mutated_backpack.genome[idx] = 1 - mutated_backpack.genome[idx]
        return mutated_backpack
    
    # Points - list of points where each sequence
    def crossover(self, backpack1:Backpack, backpack2:Backpack, points:list
                  [int]) -> tuple[Backpack, Backpack]:
        '''
        Purpose: 
            Crosses over two backpack genomes to create two new children backpacks at where genome sequences were swapped at  given locations.
        Inputs: 
            -backpack1: The first parent backpack
            -backpack2: The second parent backpack
            -points: a list of integers indicating indices where the genome should be swapped. The points represent where alternating sequences of "keep" vs. "swap" should begin and end.
        Outputs: 
            -backpack3: The first child backpack
            -backpack4: The second child backpack
        '''
        # No crossover case
        if not points:
            return backpack1, backpack2
        elif len(backpack1.genome) != len(backpack2.genome):
            raise ValueError("Both backpacks must have genomes of equal length")
        elif (len(points) + 1) >= len(backpack1.genome):
            raise ValueError("You cannot crossover more than [genome length - 1] number of points")
        elif max(points) >= len(backpack1.genome):
            raise ValueError("the maximum point to crossover must be less than the length of the genome")
        points.sort()
        # Needs to end at end of genome
        if points[-1] != len(backpack1.genome):
            points.append(len(backpack1.genome))
        backpack3_genome = []
        backpack4_genome = []
        seq_start = 0
        for i, point in enumerate(points):
            if i % 2 == 0:
                backpack3_genome += backpack2.genome[seq_start:point]
                backpack4_genome += backpack1.genome[seq_start:point]
            else:
                backpack3_genome += backpack1.genome[seq_start:point]
                backpack4_genome += backpack2.genome[seq_start:point] 
            seq_start = point
        backpack3 = Backpack(backpack3_genome, self.avail_boxes, self.capacity)
        backpack4 = Backpack(backpack4_genome, self.avail_boxes, self.capacity)
        return backpack3, backpack4

    def cull(self, pct: float) -> None:
        '''
        Purpose: 
            Culls the population of backpacks by a given %. The top remaining backpacks are chosen by fitness.
        Inputs: 
            -pct: The percent of the population to keep
        Outputs: none 
        '''
        num_left = int(len(self.population)*pct)
        self.population = sorted(self.population)[num_left:]

        
