# Casey Owen
# CS131
# Assignment 3, the Knapsack Problem
'''
Implementation of the Knapsack Problem Assignment, main.py. Runs the main
program, prints the best backpack and saves the resulting fitness graph to the current directory.
'''
from backpack import Backpack
from geneticsearch import GeneticSearch
import numpy
import matplotlib.pyplot as plt

def main() -> None:
    '''
    Purpose: 
        Sets up the program, initializes the state of the problem, calls the search function in Genetirc, and plots the results.
    Inputs: none
    Outputs: none
    '''
    boxes = [(20,6), (30,5), (60,8), (90,7), (50,6), (70,9), (30,4), (30,5), (70,4), (20,9), (20,2), (60,1)]
    capacity = 250
    pop_size = 100
    max_iter = 500
    max_plateau_steps = 100
    plot_filepath = 'results_plot.png'

    gs = GeneticSearch(boxes, population_size=pop_size, capacity=capacity)
    best_backpack, pop_fitness_by_iter = gs.search(max_iter=max_iter, max_plateau_steps=max_plateau_steps)

    print(f'The Best backpack after {len(pop_fitness_by_iter)} iterations and a population size of {pop_size} is:\n')
    best_backpack.print_backpack()
    print(f'\nSee the plot saved in the file "{plot_filepath}" for information on convergence to this solution over each iteration')
    plot_percentiles(pop_fitness_by_iter, plot_filepath)
    
def plot_percentiles(pop_fitness_by_iter:list[list[int]], filepath:str):
    '''
    Purpose: 
        Plots the fitness of individuals within the population at various percentiles - 0, 25, 50, 75, and 100 against the iteration number of the solution 
    Inputs: 
        -pop_fitness_by_iter: A list of lists of the population fitness of all members of the population, one entry for each iteration
        -filepath: The filepath to save the resulting plot to
    Outputs: none
    '''
    percentiles = []
    for fitness_state in pop_fitness_by_iter:
        percentiles.append(numpy.percentile(fitness_state, [100, 75, 50, 25, 0]))
    plt.plot(percentiles)
    plt.xlabel('Iteration')
    plt.ylabel('Population Fitness')
    plt.title('Population Fitness by Iteration')
    plt.legend(['Maximum', '75th Percentile', 'Median', '25th Percentile', 'Minimum'])
    plt.savefig(filepath)

if __name__ == "__main__":
    main()