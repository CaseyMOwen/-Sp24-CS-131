# Casey Owen
# CS131
# Assignment 4, A Sudoku Solver
'''
Searches for the solution to a Sudoku Puzzle, using a Constraint Satisfaction Problem Method
'''

import csv
from sudoku import *
import copy

class Solver():
    def __init__(self, filename: str) -> None:
        '''
        Purpose: 
            Solver Constructor for Solver class, used for searching for solutions to a Sudoku puzzle. Creates the given puzzle as a Sudoku object.
        Inputs: 
            -filename: The csv file containing the initial state of the puzzle to be solved. It should be a csv file with headers, where the 3 columns represent the x position, y position, and the value at that position. 0-indexed, starting at the top left of the puzzle.
        Outputs: none
        '''
        self.starting_puzzle = Sudoku()
        with open(filename, newline='') as f:
            puzzlereader = csv.reader(f)
            # Skip header
            next(puzzlereader, None)
            for entry in puzzlereader:
                self.starting_puzzle.assignVal(int(entry[0]), int(entry[1]), int(entry[2]))
        

    def backtrackingSearch(self) -> Sudoku:
        '''
        Purpose: 
            Backtracking Search Algorithm, wrapper for the recursive backtracking function. Calls it on the initial puzzle and returns the result
        Inputs: none
        Outputs: The sudoku puzzle solution, or the string 'Failure', if the solution could not be found
        '''
        return self.recursiveBacktracking(self.starting_puzzle)

    def recursiveBacktracking(self, puzzle: Sudoku) -> Sudoku:
        '''
        Purpose: 
            Recursive Backtracking Search Algorithm, using a Constraint Satisfaction problem method. Utilizes forward checking, variable ordering by MRV (minimum remaining values), and value ordering by LCV (least constraining value)
        Inputs: 
            -puzzle: A puzzle state to attempt to solve
        Outputs: 
            -result: The sudoku puzzle solution, or the string 'Failure', if the solution could not be found
        '''
        if puzzle.isSolved():
            return puzzle
        unassigned_row_idx, unassigned_col_idx = self.selectUnassignedVariable(puzzle)
        domain = self.getOrderedDomain(puzzle, unassigned_row_idx, unassigned_col_idx)
        init_puzzle = copy.deepcopy(puzzle)
        for val in domain:
            if puzzle.assignVal(unassigned_row_idx, unassigned_col_idx, val):
                result = self.recursiveBacktracking(copy.deepcopy(puzzle))
                if result != 'Failure':
                    return result
                puzzle = copy.deepcopy(init_puzzle)
        return 'Failure'

    def selectUnassignedVariable(self, puzzle: Sudoku) -> tuple[int, int]:
        '''
        Purpose: 
            Selects an unassigned variable for use in the recursive backtracking algorithm, selecting by MRV heuristic (minimum remaining values). Checks the domain sizes of all cells/variables, and returns the indices of the cell with the smallest domain.
        Inputs: 
            -puzzle: A puzzle to select the unassigned variable for
        Outputs: 
            -min_domain_row_idx: The row index of the cell with the smallest domain
            -min_domain_col_idx: The column index of the cell with the smallest domain
        '''
        min_domain_size = 10
        min_domain_row_idx = None
        min_domain_col_idx = None
        for row_idx, domain_row in enumerate(puzzle.domains):
            for col_idx, domain in enumerate(domain_row):
                domain_size = len(domain)
                if puzzle.board[row_idx,col_idx] == 0 and domain_size < min_domain_size:
                    min_domain_size = domain_size
                    min_domain_row_idx = row_idx
                    min_domain_col_idx = col_idx
        return min_domain_row_idx, min_domain_col_idx
        
    def getOrderedDomain(self, puzzle:Sudoku, row_idx: int, col_idx: int) -> list[int]:
        '''
        Purpose: 
            Orders the domain of a selected variable for use in the recursive backtracking algorithm, ordering by LCV heuristic (least constraining value). For each possible assignment in the domain, checks the how many domains would be reduced by that assignement, and orders the domain into a list ordered such that the assignments that rule out the fewest choices in other domains are pushed to the front of the list
        Inputs: 
            -puzzle: The puzzle to select the unassigned variable for
            -row_idx: The row index of the variable that has been selected
            -col_idx: The column index of the variable that has been selected
        Outputs: 
            -ordered_domain: The domain put into a list ordered such that the assignments that rule out the fewest choices in other domains are pushed to the front of the list
        '''
        unordered_domain = list(puzzle.domains[row_idx, col_idx])
        num_affected_cells = []
        for assignment in unordered_domain:
            num_affected_cells.append(self.getNumAffectedDomains(puzzle, row_idx, col_idx, assignment))
        ordered_domain = [x for _, x in sorted(zip(num_affected_cells, unordered_domain))]
        return ordered_domain    

    def getNumAffectedDomains(self, puzzle:Sudoku, row_idx: int, col_idx: int, val:int) -> int:
        '''
        Purpose: 
            Checks how many domains would be affected by a given assignment to a variable within a sudoku puzzle. For use within the LCV calculation.
        Inputs: 
            -puzzle: The puzzle to check the number of affected domains for
            -row_idx: The row index of the assignment
            -col_idx: The column index of the assignment
            -val: The value of the assignment
        Outputs: 
            -num_affected_domains: The number of domains that would be affected by the given assignment to the given puzzle
        '''
        min_block_row_idx = (3*math.floor((row_idx+1)/3))
        max_block_row_idx = (3*math.ceil((row_idx+1)/3))
        min_block_col_idx = (3*math.floor((col_idx+1)/3))
        max_block_col_idx = (3*math.ceil((col_idx+1)/3))
        num_affected_domains = 0
        for i, domain_row in enumerate(puzzle.domains):
            for j, domain in enumerate(domain_row):
                ij_in_bock = (i >= min_block_row_idx and i < max_block_row_idx and j >= min_block_col_idx and j < max_block_col_idx)
                # if in same row, column, or block, and the value is in the domain 
                if ((i == row_idx or j == col_idx or (ij_in_bock)) and (val in domain)):
                    num_affected_domains += 1
        return num_affected_domains



