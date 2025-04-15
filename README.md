# Uno-Bot: Reinforcement Learning Project

## Environment Setup
1. Install `Python 3.6` and `pip`
2. Install required packages: `pip install numpy pytest`

## Files
- **Game Files (Starting version retrieved from [here](https://github.com/bennuttall/uno)):**
	- `uno.py` - CLI version of game
	- `uno_pgz.py` - GUI version of game
	- `images/` - Graphics for GUI version
	- `uno_tests.py` - Unit tests
	- `play_game.py` - Example game loop
	- `Player.py` - Base player class
	- `HumanPlayer.py` - Human player implementation
	- `RandomPlayer.py` - Random AI player (additional implementation)

- **Training Files:**
	- `TrainingEnvironment.py` - Training environment
	- `reinforcement.py` - Q-Learning-based reinforcement learning, based on running a set number of episodes (pivoted away from this approach; see `time_reinforcement.py`)
	- `time_reinforcement.py` - Q-Learning-based reinforcement learning, based on running for a period of time (primarily used this file for training)
	- `test_qtable.py` - Evaluate trained Agent after `time_reinforcement.py`
	- `random_testing.py` - Evaluate Random Agent

## Usage
- Training an Agent: `python reinforcement.py`
- Test the trained agent: `python test_qtable.py`
