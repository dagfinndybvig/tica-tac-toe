"""
Flask application for Tic-Tac-Toe with Deep Learning
"""
from flask import Flask, render_template, jsonify, request, session
import numpy as np
import os
from game import TicTacToe
from model import TicTacToeModel
from training import SelfPlayTrainer
import secrets

app = Flask(__name__)
app.secret_key = secrets.token_hex(16)

# Global model instance
model = TicTacToeModel()

# Training status
training_status = {
    'is_training': False,
    'progress': 0,
    'total_games': 0,
    'results': {'wins': 0, 'losses': 0, 'draws': 0}
}


@app.route('/')
def index():
    """Home page"""
    return render_template('index.html')


@app.route('/train')
def train_page():
    """Training page"""
    return render_template('train.html')


@app.route('/play')
def play_page():
    """Play page"""
    # Initialize new game in session
    session['game_board'] = [0] * 9
    session['current_player'] = 1
    session['game_over'] = False
    return render_template('play.html')


@app.route('/api/start_training', methods=['POST'])
def start_training():
    """Start training the model"""
    global training_status, model
    
    if training_status['is_training']:
        return jsonify({'error': 'Training already in progress'}), 400
    
    # Reset training status
    training_status = {
        'is_training': True,
        'progress': 0,
        'total_games': 100,
        'results': {'wins': 0, 'losses': 0, 'draws': 0}
    }
    
    def training_callback(game_num, total, results):
        """Update training progress"""
        training_status['progress'] = game_num
        training_status['results'] = results.copy()
    
    try:
        # Create new model for training
        model = TicTacToeModel()
        trainer = SelfPlayTrainer(model)
        
        # Train the model
        results = trainer.train(num_games=100, callback=training_callback)
        
        # Save the trained model
        model.save_model()
        
        training_status['is_training'] = False
        return jsonify({
            'success': True,
            'results': results
        })
    except Exception as e:
        training_status['is_training'] = False
        return jsonify({'error': str(e)}), 500


@app.route('/api/training_status')
def get_training_status():
    """Get current training status"""
    return jsonify(training_status)


@app.route('/api/load_model', methods=['POST'])
def load_model():
    """Load trained model"""
    global model
    try:
        model = TicTacToeModel()
        if model.load_model():
            return jsonify({'success': True, 'message': 'Model loaded successfully'})
        else:
            return jsonify({'success': False, 'message': 'No trained model found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/make_move', methods=['POST'])
def make_move():
    """Make a move in the game"""
    data = request.get_json()
    position = data.get('position')
    
    if position is None or position < 0 or position > 8:
        return jsonify({'error': 'Invalid position'}), 400
    
    # Get game state from session
    board = session.get('game_board', [0] * 9)
    current_player = session.get('current_player', 1)
    game_over = session.get('game_over', False)
    
    if game_over:
        return jsonify({'error': 'Game is already over'}), 400
    
    if board[position] != 0:
        return jsonify({'error': 'Position already occupied'}), 400
    
    # Make player move
    board[position] = current_player
    
    # Create game instance to check winner
    game = TicTacToe()
    game.board = np.array(board)
    winner = game.check_winner()
    
    if winner is not None:
        session['game_board'] = board
        session['game_over'] = True
        return jsonify({
            'board': board,
            'winner': winner,
            'game_over': True
        })
    
    # AI's turn
    current_player *= -1
    game.current_player = current_player
    valid_moves = game.get_valid_moves()
    
    if valid_moves:
        # AI makes a move
        ai_move = model.get_best_move(np.array(board), valid_moves)
        board[ai_move] = current_player
        game.board = np.array(board)
        winner = game.check_winner()
        
        if winner is not None:
            session['game_board'] = board
            session['game_over'] = True
            return jsonify({
                'board': board,
                'ai_move': ai_move,
                'winner': winner,
                'game_over': True
            })
    
    # Update session
    current_player *= -1
    session['game_board'] = board
    session['current_player'] = current_player
    
    return jsonify({
        'board': board,
        'ai_move': ai_move if valid_moves else None,
        'winner': None,
        'game_over': False
    })


@app.route('/api/reset_game', methods=['POST'])
def reset_game():
    """Reset the game"""
    session['game_board'] = [0] * 9
    session['current_player'] = 1
    session['game_over'] = False
    return jsonify({'success': True})


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
