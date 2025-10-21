"""
Deep Learning model for Tic-Tac-Toe
"""
import numpy as np
import keras
from keras import layers
import os


class TicTacToeModel:
    """Neural network model for playing tic-tac-toe"""
    
    def __init__(self, model_path='models/tictactoe_model.keras'):
        self.model_path = model_path
        self.model = self._build_model()
        
    def _build_model(self):
        """Build the neural network architecture"""
        model = keras.Sequential([
            layers.Input(shape=(9,)),
            layers.Dense(64, activation='relu'),
            layers.Dense(64, activation='relu'),
            layers.Dense(32, activation='relu'),
            layers.Dense(9, activation='linear')  # Q-values for each position
        ])
        model.compile(optimizer='adam', loss='mse')
        return model
    
    def predict(self, state):
        """Predict Q-values for a given state"""
        state = np.array(state, dtype=np.float32).reshape(1, -1)
        return self.model.predict(state, verbose=0)[0]
    
    def train_on_batch(self, states, targets):
        """Train the model on a batch of states and targets"""
        states = np.array(states, dtype=np.float32)
        targets = np.array(targets, dtype=np.float32)
        return self.model.train_on_batch(states, targets)
    
    def save_model(self):
        """Save the model to disk"""
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        self.model.save(self.model_path)
        
    def load_model(self):
        """Load the model from disk"""
        if os.path.exists(self.model_path):
            self.model = keras.models.load_model(self.model_path)
            return True
        return False
    
    def get_best_move(self, state, valid_moves):
        """Get the best move for a given state"""
        q_values = self.predict(state)
        # Mask invalid moves
        masked_q = np.full(9, -np.inf)
        for move in valid_moves:
            masked_q[move] = q_values[move]
        return np.argmax(masked_q)
