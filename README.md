# tica-tac-toe
Deep learning tic-tac-toe experiment

## Overview
A Flask web application that trains a deep learning model to play tic-tac-toe through self-play and allows users to play against the trained AI.

## Features
- **Training**: Train a neural network through self-play for 100 games with real-time progress visualization
- **Play**: Play tic-tac-toe against the trained AI model
- **Model Persistence**: Trained models are saved to disk and can be loaded for gameplay

## Setup

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Installation

1. Clone the repository:
```bash
git clone https://github.com/dagfinndybvig/tica-tac-toe.git
cd tica-tac-toe
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Running the Application

Start the Flask application:
```bash
python app.py
```

The application will be available at `http://localhost:5000`

## Usage

1. **Home Page** (`/`): Overview of the application and navigation
2. **Train Model** (`/train`): Train the AI model through 100 games of self-play
   - Click "Start Training" to begin
   - Watch real-time progress with statistics
   - Model is automatically saved when training completes
3. **Play Game** (`/play`): Play against the trained AI
   - Click "Load Trained Model" to load the trained AI
   - Click any cell to make your move (you play as X)
   - AI automatically responds (plays as O)
   - Click "New Game" to restart

## Technical Details

### Dependencies
- **Flask**: Web framework for the application
- **TensorFlow/Keras**: Deep learning framework for the neural network
- **NumPy**: Numerical computing for game state management

### Architecture
- **Game Logic** (`game.py`): Core tic-tac-toe game implementation
- **Model** (`model.py`): Neural network architecture and predictions
- **Training** (`training.py`): Self-play training algorithm with Q-learning
- **Flask App** (`app.py`): Web server and API endpoints
- **Templates**: HTML frontend for training and gameplay

### Model Architecture
- Input: 9 features (board state)
- Hidden layers: 64 → 64 → 32 neurons (ReLU activation)
- Output: 9 Q-values (one per board position)
- Training: Q-learning with experience replay from self-play games
