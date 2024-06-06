# Casey Owen
# CS131
# Assignment 4, A Sudoku Solver
'''
Implementation of the Sudoku Class - a representation of the sudoku puzzle in the Sudoku Solver. Contains the puzzle board, domains for each cell, and maintains the sudoku invariant - all Sudoku objects represent legal puzzle boards that do not violate the constraints.
'''

import numpy as np
import math

class Sudoku():

    def __init__(self) -> None:
        '''
        Purpose: 
            Constructor for Sudoku class, used for representing legal sudoku boards and the domains of each cell. Creates an empty puzzle.
        Inputs: none
        Outputs: none
        '''
        # Empty cells are represented by 0's
        self.board = np.zeros((9,9), dtype='int')
        # Need to use list comprehension rather than np.full so that each set is a distinct object
        self.domains = np.array([[set(range(1,10)) for _ in range(9)] for _ in range(9)])

    def containsDups(self, arr) -> bool:
        '''
        Purpose: 
            Checks if a given array contains duplicate values other than 0, which represents an empty cell.
        Inputs: 
            -arr: The array to check for duplicate values
        Outputs: 
            -result: Boolean indicating whether the array contain duplicate values.
        '''
        arr = arr[arr != 0]
        if len(arr) != len(np.unique(arr)):
            return True
        else:
            return False

    def checkRows(self) -> bool:
        '''
        Purpose: 
            Checks if any rows in the board contain duplicate values
        Inputs: none
        Outputs: 
            -result: Boolean indicating whether any rows contain duplicate values.
        '''
        for row in self.board:
            if self.containsDups(row):
                return False
        return True
    
    def checkCols(self) -> bool:
        '''
        Purpose: 
            Checks if any columns in the board contain duplicate values
        Inputs: none
        Outputs: 
            -result: Boolean indicating whether any columns contain duplicate values.
        '''
        for col in self.board.T:
            if self.containsDups(col):
                return False
        return True

    def checkBlocks(self) -> bool:
        '''
        Purpose: 
            Checks if any of the 9 3x3 blocks in the board contain duplicate values
        Inputs: none
        Outputs: 
            -result: Boolean indicating whether any blocks contain duplicate values.
        '''
        for row_idx in [0, 3, 6]:
            for col_idx in [0, 3, 6]:
                block = self.board[row_idx:row_idx+3,col_idx:col_idx+3]
                block.flatten()
                if self.containsDups(block):
                    return False
        return True
    
    def isValid(self) -> bool:
        '''
        Purpose: 
            Checks if the current puzzle board violates any of the sudoku constraints (rows, blocks, or columns)
        Inputs: none
        Outputs: 
            -result: Boolean indicating whether any constraints are violated.
        '''
        return self.checkRows() and self.checkCols() and self.checkBlocks()

    # Assigns and returns true if value does not break constraints, otherwise returns false
    def assignVal(self, row_idx: int, col_idx: int, val: int) -> bool:
        '''
        Purpose: 
            Assigns a value to given row and column indexes, if the assignment does not violate any constraints, else does not make the assignment. Returns true on success, false on failure. If true, also updates the domains. 
        Inputs: 
            row_idx: Row index of the value to be assigned
            col_idx: Column index of the value to be assigned
            val: Value to be assigned
        Outputs: 
            -result: Boolean indicating success or failure of assignment.
        '''
        old_val = self.board[(row_idx, col_idx)] 
        self.board[(row_idx, col_idx)] = val
        if self.isValid() and val <= 9 and val >= 1:
            self.updateDomains(row_idx, col_idx, val)
            return True
        else:
            self.board[(row_idx, col_idx)] = old_val
            return False


    def updateDomains(self, row_idx: int, col_idx: int, val: int) -> None:
        '''
        Purpose: 
            Updates the stored domains of all cells based on a given variable assignment.
        Inputs: 
            row_idx: Row index of the value to be assigned
            col_idx: Column index of the value to be assigned
            val: Value to be assigned
        Outputs: none
        '''
        for domain in self.domains[row_idx, :]:
            domain.discard(val)
        for domain in self.domains[:, col_idx]:
            domain.discard(val)
        min_block_row_idx = (3*math.floor((row_idx+1)/3))
        max_block_row_idx = (3*math.ceil((row_idx+1)/3))
        min_block_col_idx = (3*math.floor((col_idx+1)/3))
        max_block_col_idx = (3*math.ceil((col_idx+1)/3))
        block_domains = self.domains[min_block_row_idx:max_block_row_idx,min_block_col_idx:max_block_col_idx]
        for domain_row in block_domains:
            for domain in domain_row:
                domain.discard(val)
        
    def isSolved(self) -> bool:
        '''
        Purpose: 
            Checks if the puzzle is solved (all cells are filled)
        Inputs: none
        Outputs: 
            -result: Boolean indicating if the puzzle is solved
        '''
        # Because no-duplicates invariant is maintained on insertion
        if self.board.sum() == 45*9:
            return True
        else:
            return False
        
    def print_board(self) ->None:
        '''
        Purpose: 
            Pretty-prints the sudoku board at its current state.
        Inputs: none
        Outputs: none
        '''
        for j, row in enumerate(self.board):
            if j % 3 == 0:
                print(" " + "-"*30)
            s = "| "
            for j, number in enumerate(row):
                num_str = str(number)
                if num_str == '0':
                    num_str = "-"
                if j % 3 == 2:
                    s += num_str + " | "
                else:
                    s += num_str + "  "
            print(s)
        print(" " + "-"*30)

    