# Uno Bot: Experiment with Deep Q Network model 
This branch demonstrates a DQN-based approach to training an Uno agent. It reuses the same Uno game and player classes from the main branch, adding new files for the neural network model and training environment.
## Environment Setup
1. Install `Python 3.6` and `pip`
2. Install required packages: `pip install numpy torch pytest`
## Usage
- To train an agent, configure then run `train_dqn.py`
	- Adjust hyperparameters (e.g., number of episodes, batch size, learning rate) as needed.
	- A model checkpoint (e.g., dqn_save.pth) will be saved upon completion.
- To testing the DQN Agent, configure then run `test_pth.py`
	- This file loads the sample trained model (dqn_sample.pth) and evaluates win rate over multiple Uno games.
## Files
- **Training Files:**
	- `dqn.py` – Defines the DQN model (PyTorch-based).
	- `dqn_env.py` – Environment specifically tailored for DQN training and inference.
	- `train_dqn.py` – Runs the DQN training loop.
	- `test_pth.py` – Evaluates the sample trained DQN model.
- **Uno and Player Files:**
	- These are reused from the main branch (e.g., uno.py, RandomPlayer.py). They manage the game flow and provide random players. No changes are required to the Uno logic; the new DQN files simply import and use those classes.

## Conclusion
Under identical conditions—with the same partially observable environment and similar resource and time constraints—the DQN model achieved an average win rate of 34.6%. This performance indicates that, in our current setup, the DQN approach did not outperform our existing Q-learning method.





