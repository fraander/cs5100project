# Uno-Bot: Reinforcement Learning Project

## Environment Setup
1. Install `Python 3.6` and `pip`
2. Install required packages: `pip install numpy pytest`

## Usage
- To train an agent, configure then run `time_reinforcement.py`
	- Lines 10-12: Configure `LOGS = ` and `PICKLE = ` to define where to store logging and Q-table files. Configure `RUNTIME = ` to define how long to train, in minutes.
- To test a trained agent, configure then run `test_qtable.py`. After completion, the number of wins divided by the total number of games (win percentage) completed is printed to the console.
	- Lines 6-8: Configure `LOGS = ` and `PICKLE = ` to define where to store logging and Q-table files. Configure `TEST_SIZE = ` to define how many games to test with.

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

## Further exploration
Following submission of the project we've explored Deep Q-Learning as a potential future improvement. See the progress on the `experiment/dqn-model` branch!