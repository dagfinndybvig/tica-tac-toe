"""
Self-play training for Tic-Tac-Toe
"""
import numpy as np
import random
from game import TicTacToe
from model import TicTacToeModel


class SelfPlayTrainer:
    """Train the model through self-play"""
    
    def __init__(self, model, epsilon=0.2, gamma=0.95):
        self.model = model
        self.epsilon = epsilon  # Exploration rate
        self.gamma = gamma      # Discount factor
        
    def select_action(self, game, explore=True):
        """Select an action using epsilon-greedy policy"""
        valid_moves = game.get_valid_moves()
        
        if not valid_moves:
            return None
        
        # Exploration
        if explore and random.random() < self.epsilon:
            return random.choice(valid_moves)
        
        # Exploitation
        state = game.get_board_state()
        return self.model.get_best_move(state, valid_moves)
    
    def play_game(self, explore=True):
        """Play one game of self-play and return the game history"""
        game = TicTacToe()
        history = []
        
        while True:
            # Get current state
            state = game.get_board_state()
            valid_moves = game.get_valid_moves()
            
            if not valid_moves:
                break
            
            # Select and make move
            action = self.select_action(game, explore)
            if action is None:
                break
                
            player = game.current_player
            game.make_move(action)
            
            # Check for end of game
            winner = game.check_winner()
            
            # Store transition
            next_state = game.get_board_state()
            history.append({
                'state': state,
                'action': action,
                'player': player,
                'next_state': next_state,
                'winner': winner
            })
            
            if winner is not None:
                break
        
        return history, winner
    
    def calculate_rewards(self, history, winner):
        """Calculate rewards for each move in the game"""
        rewards = []
        for move in history:
            if winner == 0:  # Draw
                rewards.append(0)
            elif winner == move['player']:  # Win
                rewards.append(1)
            else:  # Loss
                rewards.append(-1)
        return rewards
    
    def train_on_game(self, history, rewards):
        """Train the model on a single game"""
        if not history:
            return
        
        states = []
        targets = []
        
        for i, move in enumerate(history):
            state = move['state']
            action = move['action']
            next_state = move['next_state']
            reward = rewards[i]
            
            # Get current Q-values
            q_values = self.model.predict(state)
            
            # Update Q-value for the action taken
            if move['winner'] is not None:
                # Terminal state
                q_values[action] = reward
            else:
                # Non-terminal state: Q-learning update
                next_q = self.model.predict(next_state)
                q_values[action] = reward + self.gamma * np.max(next_q)
            
            states.append(state)
            targets.append(q_values)
        
        # Train on batch
        if states:
            self.model.train_on_batch(states, targets)
    
    def train(self, num_games=100, callback=None):
        """Train the model for a specified number of games"""
        results = {'wins': 0, 'losses': 0, 'draws': 0}
        
        for game_num in range(num_games):
            # Play a game
            history, winner = self.play_game(explore=True)
            
            # Calculate rewards
            rewards = self.calculate_rewards(history, winner)
            
            # Train on the game
            self.train_on_game(history, rewards)
            
            # Track results
            if winner == 1:
                results['wins'] += 1
            elif winner == -1:
                results['losses'] += 1
            else:
                results['draws'] += 1
            
            # Callback for progress updates
            if callback:
                callback(game_num + 1, num_games, results)
        
        return results
