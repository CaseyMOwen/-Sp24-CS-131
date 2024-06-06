# Casey Owen
# CS131
# Assignment 2, the Pancake Problem
'''
Implementation of the Pancake Problem Assignment, main.py. Runs the main
program, accepts user input, and prints output to terminal.
'''

from search import Search
from pancakestack import PancakeStack

def main():
    '''
    Purpose: 
        Sets up the program, and calls the run function until the program is complete
    Inputs: none
    Outputs: none
    '''
    print("Welcome to the Pancake Problem Solver!")
    print("This tool allows you to solve the pancake problem using either the UCS algorithm or the A* algorithm using the gap heurstic.")
    print("If you choose A*, a completely random stack of 10 pancakes will be generated to be solved. \nHowever, if you choose UCS, due to the significant runtime of the algorithm, the random stack will be limited to at most 5 flips away from a solution.")
    done = False
    while not done:
        done = run()

def print_sol(initial_stack: PancakeStack, solution: list[int], additional_stats: tuple[int, int]) -> None:
    '''
    Purpose: 
        Prints the solution of the problem to the console in a user friendly format
    Inputs: 
        -initial_stack: The initial pancake stack that the problem solved
        -solution:  A list of integers representing the solution of the problem - the number of pancakes that should be flipped at each step, in order
        -additional_stats: A tuple of two integers, the total number of nodes visited, and the total number of nodes added to the frontier to find the solution
    Outputs: none
    '''
    print(f'The solution requires {len(solution)} steps: {solution}.')
    print('This represents the number of pancakes that should be flipped at each step.')
    print('To find the solution:')
    print(f'The total number of nodes visited is: {additional_stats[0]}')
    print(f'The total number of nodes added to the frontier is: {additional_stats[1]}')
    stack = initial_stack
    stack.print_state(0)
    stack.print_state(1, solution[0])
    num_steps = len(solution)
    for i, flip in enumerate(solution):
        stack.flip(flip)
        if i + 1 < num_steps:
            stack.print_state(i+2, solution[i+1])
    stack.print_state(None)

def run() -> bool:
    '''
    Purpose: 
        Runs the primary program logic, accepts user input and handles commands
    Inputs: none
    Outputs: 
        -done: a boolean indicating if the program is done running
    '''
    try:
        cmd = input("Please enter either 'UCS', 'A*', or  'quit' now:")
    except KeyboardInterrupt:
        cmd = "quit"
    print('\n\n')
    if cmd == "UCS":
        stack = PancakeStack(random_flips=5)
        sol, addnl_stats = Search().find_solution(stack, heuristic=None)
    elif cmd == "A*":
        stack = PancakeStack()
        sol, addnl_stats = Search().find_solution(stack, heuristic='gap')
    elif cmd == "quit":
        print("Goodbye!")
        return True
    else:
        print('Your command was not valid, please try again.')
        return False

    if sol == None:
        print("Could not find a solution")
    elif sol == []:
        print("Wow, that randomly generated stack was already in order! Try again for a non-trivial problem.")
    else:
        print_sol(stack, sol, addnl_stats)
    print("\nIf you'd like to run again, you may enter a new command.")
    return False


if __name__ == "__main__":
    main()