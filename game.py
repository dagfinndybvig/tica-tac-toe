"""
Tic-Tac-Toe game logic
"""
import numpy as np


class TicTacToe:
    """Tic-Tac-Toe game implementation"""
    
    def __init__(self):
        self.board = np.zeros(9, dtype=int)
        self.current_player = 1  # 1 for X, -1 for O
        
    def reset(self):
        """Reset the game board"""
        self.board = np.zeros(9, dtype=int)
        self.current_player = 1
        return self.board.copy()
    
    def get_valid_moves(self):
        """Get list of valid move indices"""
        return [i for i in range(9) if self.board[i] == 0]
    
    def make_move(self, position):
        """Make a move at the given position"""
        if self.board[position] != 0:
            return False
        self.board[position] = self.current_player
        self.current_player *= -1
        return True
    
    def check_winner(self):
        """
        Check if there's a winner
        Returns: 1 if player 1 wins, -1 if player 2 wins, 0 if draw, None if game ongoing
        """
        # Winning combinations
        wins = [
            [0, 1, 2], [3, 4, 5], [6, 7, 8],  # rows
            [0, 3, 6], [1, 4, 7], [2, 5, 8],  # columns
            [0, 4, 8], [2, 4, 6]              # diagonals
        ]
        
        for combo in wins:
            if self.board[combo[0]] == self.board[combo[1]] == self.board[combo[2]] != 0:
                return self.board[combo[0]]
        
        # Check for draw
        if 0 not in self.board:
            return 0
        
        # Game ongoing
        return None
    
    def get_board_state(self):
        """Get current board state"""
        return self.board.copy()
    
    def get_board_2d(self):
        """Get board as 2D array for display"""
        return self.board.reshape(3, 3)
