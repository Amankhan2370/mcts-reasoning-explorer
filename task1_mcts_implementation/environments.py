"""
Game environments for testing MCTS.
Each environment must implement:
    - get_legal_actions(state)
    - get_next_state(state, action)
    - is_terminal(state)
    - get_reward(state, action, next_state)
    - get_terminal_reward(state)
"""

import numpy as np
from typing import List, Tuple, Any


class TicTacToe:
    """
    Tic-Tac-Toe environment.
    
    State representation: 3x3 numpy array
        0 = empty
        1 = player 1 (X)
        -1 = player 2 (O)
    """
    
    def __init__(self):
        self.size = 3
    
    def get_initial_state(self) -> np.ndarray:
        """Return empty board."""
        return np.zeros((self.size, self.size), dtype=int)
    
    def get_legal_actions(self, state: np.ndarray) -> List[Tuple[int, int]]:
        """Return list of (row, col) positions that are empty."""
        actions = []
        for i in range(self.size):
            for j in range(self.size):
                if state[i, j] == 0:
                    actions.append((i, j))
        return actions
    
    def get_next_state(self, state: np.ndarray, action: Tuple[int, int]) -> np.ndarray:
        """Apply action and return new state."""
        new_state = state.copy()
        row, col = action
        
        # Determine current player
        current_player = 1 if np.sum(np.abs(state)) % 2 == 0 else -1
        new_state[row, col] = current_player
        
        return new_state
    
    def is_terminal(self, state: np.ndarray) -> bool:
        """Check if game is over."""
        return self._check_winner(state) != 0 or len(self.get_legal_actions(state)) == 0
    
    def _check_winner(self, state: np.ndarray) -> int:
        """
        Check for winner.
        Returns: 1 if player 1 wins, -1 if player 2 wins, 0 if no winner
        """
        # Check rows
        for i in range(self.size):
            if abs(sum(state[i, :])) == self.size:
                return state[i, 0]
        
        # Check columns
        for j in range(self.size):
            if abs(sum(state[:, j])) == self.size:
                return state[0, j]
        
        # Check diagonals
        if abs(sum(state.diagonal())) == self.size:
            return state[0, 0]
        
        if abs(sum(np.fliplr(state).diagonal())) == self.size:
            return state[0, self.size - 1]
        
        return 0
    
    def get_reward(self, state: np.ndarray, action: Tuple[int, int], 
                   next_state: np.ndarray) -> float:
        """Intermediate rewards (0 for non-terminal moves)."""
        return 0.0
    
    def get_terminal_reward(self, state: np.ndarray) -> float:
        """
        Terminal reward from perspective of player 1.
        +1 if player 1 wins, -1 if player 2 wins, 0 for draw
        """
        winner = self._check_winner(state)
        return float(winner)
    
    def render(self, state: np.ndarray):
        """Print the board."""
        symbols = {0: '.', 1: 'X', -1: 'O'}
        print("\n  0 1 2")
        for i in range(self.size):
            print(f"{i} ", end="")
            for j in range(self.size):
                print(symbols[state[i, j]] + " ", end="")
            print()
        print()


class SimpleMaze:
    """
    Simple grid maze environment.
    
    State: (row, col, steps_taken)
    Goal: Reach goal position in minimum steps
    """
    
    def __init__(self, size: int = 5):
        self.size = size
        self.start = (0, 0)
        self.goal = (size - 1, size - 1)
        self.max_steps = size * size * 2
        
        # Create simple maze (1 = wall, 0 = free)
        self.maze = np.zeros((size, size), dtype=int)
        # Add some walls
        if size >= 5:
            self.maze[2, 1:4] = 1
            self.maze[1:4, 3] = 1
    
    def get_initial_state(self) -> Tuple[int, int, int]:
        """Return starting state."""
        return (*self.start, 0)
    
    def get_legal_actions(self, state: Tuple[int, int, int]) -> List[str]:
        """Return legal moves: up, down, left, right."""
        row, col, steps = state
        actions = []
        
        moves = {
            'up': (-1, 0),
            'down': (1, 0),
            'left': (0, -1),
            'right': (0, 1)
        }
        
        for action, (dr, dc) in moves.items():
            new_row, new_col = row + dr, col + dc
            
            if (0 <= new_row < self.size and 
                0 <= new_col < self.size and 
                self.maze[new_row, new_col] == 0):
                actions.append(action)
        
        return actions
    
    def get_next_state(self, state: Tuple[int, int, int], 
                       action: str) -> Tuple[int, int, int]:
        """Apply action and return new state."""
        row, col, steps = state
        
        moves = {
            'up': (-1, 0),
            'down': (1, 0),
            'left': (0, -1),
            'right': (0, 1)
        }
        
        dr, dc = moves[action]
        return (row + dr, col + dc, steps + 1)
    
    def is_terminal(self, state: Tuple[int, int, int]) -> bool:
        """Check if reached goal or exceeded step limit."""
        row, col, steps = state
        return (row, col) == self.goal or steps >= self.max_steps
    
    def get_reward(self, state: Tuple[int, int, int], action: str, 
                   next_state: Tuple[int, int, int]) -> float:
        """Small negative reward for each step."""
        return -1.0
    
    def get_terminal_reward(self, state: Tuple[int, int, int]) -> float:
        """Large positive reward for reaching goal."""
        row, col, steps = state
        if (row, col) == self.goal:
            return 100.0
        return -10.0
    
    def render(self, state: Tuple[int, int, int]):
        """Print the maze with current position."""
        row, col, steps = state
        print(f"\nStep: {steps}")
        for i in range(self.size):
            for j in range(self.size):
                if (i, j) == (row, col):
                    print("A ", end="")
                elif (i, j) == self.goal:
                    print("G ", end="")
                elif self.maze[i, j] == 1:
                    print("█ ", end="")
                else:
                    print(". ", end="")
            print()
        print()