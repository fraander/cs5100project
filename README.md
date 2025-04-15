# Uno-Bot: Reinforcement Learning Project

## Environment Setup
1. Install `Python 3.6`, `pip`, and `pypy 3.1`
2. Install required packages: `pip install numpy pytest pgzero`

## Usage
- **[TRAIN]** To train an agent, configure then run `time_reinforcement.py`
	- Configure `LOGS = ` and `PICKLE = ` to define where to store logging and Q-table files. Configure `RUNTIME = ` to define how long to train, in minutes.
- **[TEST]** To test a trained agent, configure then run `test_qtable.py`. After completion, the number of wins divided by the total number of games (win percentage) completed is printed to the console.
	- Configure `LOGS = ` and `PICKLE = ` to define where to store logging and Q-table files. Configure `TEST_SIZE = ` to define how many games to test with.
- **[RUN WITH GUI]** To run with a GUI active, run the command `pgzrun uno_pgz.py` in the Terminal, to run using the `pgzero` library.
	- To choose which Q-table is used in play, configure `PICKLE = ` in `QPlayer.py`

## Files
- **Game Files (Starting version retrieved from [here](https://github.com/bennuttall/uno)):**
	- `uno.py` - CLI version of game
	- `uno_pgz.py` - GUI version of game
	- `play_game_pgz.py` - Example game loop with GUI active
	- `images/` - Graphics for GUI version
	- `uno_tests.py` - Unit tests
	- `play_game.py` - Example game loop
	- `Player.py` - Base player class
	- `HumanPlayer.py` - Human player implementation
	- `RandomPlayer.py` - Random AI player (additional implementation)
	- `QPlayer.py` - AI Agent player

- **Training Files:**
	- `TrainingEnvironment.py` - Training environment
	- `time_reinforcement.py` - Q-Learning-based reinforcement learning, based on running for a period of time (primarily used this file for training)
	- `test_qtable.py` - Evaluate trained Agent after `time_reinforcement.py`
	- `random_testing.py` - Evaluate Random Agent
	- `transplant.py` and `transplant2.py` - convert Pickle file so they can be read by Python. Solved a permissions issue with opening some files.

## Comparison Implementations
We compared against other agent implementations to verify the signficance of our results. They can be found on the following branches:
- **Rule-based** [`compare/rule-based`] Prioritizes playing Wild & +4 cards, then +2/Skip/Reverse cards, then standard cards
- **Limited Action Space** [`compare/limited`] Automatically chooses the color after Wild for the agent
- **Random** [`compare/random`] Randomly takes valid actions

## Further exploration
Following submission of the project we've explored Deep Q-Learning as a potential future improvement. See the progress on the `experiment/dqn-model` branch!
