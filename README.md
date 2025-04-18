# Uno-Bot: Reinforcement Learning Project

## Environment Setup
1. Install `Python 3.6` and `pip`
2. Install required packages: `pip install numpy pytest pgzero`

## Usage
- **[TRAIN]** To train an agent, configure then run `python time_reinforcement.py`
	- Configure `LOGS = ` and `PICKLE = ` to define where to store logging and Q-table files. Configure `RUNTIME = ` to define how long to train, in minutes. 
- **[TEST]** To test a trained agent, configure then run `python test_qtable.py`. After completion, the number of wins divided by the total number of games (win percentage) completed is printed to the console.
	- Configure `LOGS = ` and `PICKLE = ` to define where to store logging and Q-table files. Configure `TEST_SIZE = ` to define how many games to test with.
	- `test_qtable.py` also prints out the distribution of moves. This is an array where the first element is the number of times move 0 was made, the second element is the number of times move 1 was made, etc...
- **[RUN WITH GUI]** To run with a GUI active, run the command `pgzrun uno_pgz.py` in the Terminal, to run using the `pgzero` library.
	- To choose which Q-table is used in play, configure `PICKLE = ` in the `QPlayer.py` file
	- `uno_pgz.py` will run one uno game between the Q-table and 2 players playing randomly. The current state and action chosen by the Q-table will be printed to the console at each turn.
- **[PLAY A GAME]** To play a game yourself agains the Q Agent, run `pgzrun play_uno.py`. This will open up a 3-player game with graphics where you play against one Q Agent and one Random Player.
	- To play a card, simply click the card. If you have no cards that you can play, click on the deck in the upper left to draw a card.
	- Sometimes you may need to click multiple times to get the card to play.
	- When a black card is played, the new color will display next to the pile.
	- When you play a black card, the list of possible colors you can switch to will display at the top. Simply click on the one you want.

## Files
- **Game Files (Starting version retrieved from [here](https://github.com/bennuttall/uno)):**
	- `uno.py` - CLI version of game
	- `uno_pgz.py` - GUI version of game
	- `images/` - Graphics for GUI version
	- `uno_tests.py` - Unit tests
	- `Player.py` - Base player class
	- `HumanPlayer.py` - Human player implementation
	- `RandomPlayer.py` - Random AI player (additional implementation)
	- `QPlayer.py` - AI Agent player

- **Training Files:**
	- `TrainingEnvironment.py` - Training environment
	- `time_reinforcement.py` - Q-Learning-based reinforcement learning, based on running for a period of time (primarily used this file for training)
	- `test_qtable.py` - Evaluate trained Agent after `time_reinforcement.py`
	- `random_testing.py` - Evaluate Random Agent
	- `transplant.py` and `transplant2.py` - These files convert the Q-table from a pickle file to a csv file and vice-versa. This solved an issue involving interoperability between pickle files created by different versions of python.

## Comparison Implementations
We compared against other agent implementations to verify the signficance of our results. They can be found on the following branches:
- **Rule-based** [[compare/rule-based](https://github.com/fraander/cs5100project/tree/compare/rule-based)] Prioritizes playing Wild & +4 cards, then +2/Skip/Reverse cards, then standard cards
- **Limited Action Space** [[compare/limited](https://github.com/fraander/cs5100project/tree/compare/limited)] Automatically chooses the color after Wild for the agent
- **Random** [[compare/random](https://github.com/fraander/cs5100project/tree/compare/random)] Randomly takes valid actions

## Further exploration
Following submission of the project we've explored Deep Q-Learning as a potential future improvement. See the progress on the [experiment/dqn-model](https://github.com/fraander/cs5100project/tree/experiment/dqn-model) branch!
