# Casey Owen
# CS131
# Assignment 4, A Sudoku Solver
'''
Implementation of the Sudoku Solver Assignment, main.py. Runs the main
program, accepts user input, and prints output to terminal.
'''

from solver import *
import time

def main() -> None:
    '''
    Purpose: 
        Sets up the program, initializes the state of the problem, calls the search function in Solver, and prints the results.
    Inputs: none
    Outputs: none
    '''
    print("\nWelcome to the Sudoku Puzzle Solver!")
    print("This tool allows you to solve one of two pre-set sudoku puzzles - an easy puzzle, or a hard puzzle using a Constraint Satisfaction Problems approach, based on your input.")
    done = False
    while not done:
        done = run()


def run() -> bool:
    '''
    Purpose: 
        Runs the primary program logic, accepts user input and handles commands
    Inputs: none
    Outputs: 
        -done: a boolean indicating if the program is done running
    '''
    try:
        cmd = input("Please enter either 'easy', 'hard', or  'quit' now:")
    except KeyboardInterrupt:
        cmd = "quit"
    print('\n')
    if cmd == "easy":
        findPrintSol(filename = 'easy_puzzle.csv')
    elif cmd == "hard":
        findPrintSol(filename = 'hard_puzzle.csv')
    elif cmd == "quit":
        print("Goodbye!")
        return True
    else:
        print('Your command was not valid, please try again.')
        return False
    print("\nIf you'd like to run again, you may enter a new command.")
    return False


def findPrintSol(filename:str) -> None:
    '''
    Purpose: 
        Finds the solution to the problem of from a given file, and prints the initial puzzle as well as the solution and the time to solve to the console in a user friendly format
    Inputs: 
        -filename: The csv file from which the initial sudoku board is generated
    Outputs: none
    '''
    s = Solver(filename)
    print('Starting puzzle: ')
    s.starting_puzzle.print_board()
    print('\n')
    start = time.time()
    result = s.backtrackingSearch()
    end = time.time()
    if type(result) is Sudoku:
        print('Solution:')
        result.print_board()
        print(f'Time to solution: {round(end - start,3)} seconds')
    else:
        print("Could not find a solution")


if __name__ == "__main__":
    main()