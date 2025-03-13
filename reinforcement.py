import pickle
import random

import numpy as np
from TrainingEnvironment import TrainingEnvironment

env = TrainingEnvironment()
NUM_ACTIONS = 9

color_indices = {
    'red': 0,
    'blue': 1,
    'green': 2,
    'yellow': 3,
}


# TODO: Frank, update hash function to match `todo.md`

def hash(obs) -> int:
    # unpack observation
    current, hand, history, player_num = obs

    # calculate competitor ids
    cw_num = (player_num - 1) % 3
    acw_num = (player_num + 1) % 3

    # calculate each match
    match_num = 1 if any([c.card_type == current.card_type for c in hand]) else 0 # any card matches current card type
    match_color = min(2, len([c for c in hand if c.color == current.color])) # any card matches current card color

    wild = 1 if any(c.card_type == 'wildcard' for c in hand) else 0 # any wildcard in hand
    draw_4 = 1 if any(c.card_type == '+4' for c in hand) else 0 # any +4 in hand

    match_draw_2 = 1 if any([c.color == current.color and c.card_type == '+2' for c in hand]) else 0 # any +2 of color in hand
    match_skip = 1 if any([c.color == current.color and c.card_type == 'skip' for c in hand]) else 0 # any skip of color in hand
    match_reverse = 1 if any([c.color == current.color and c.card_type == 'reverse' for c in hand]) else 0 # any rev of color in hand

    # clockwise (cw) is default, anti-clockwise (acw) is if an odd number of reverses have been played
    direction_of_play = "cw" if len([h for h in history if h.played_card == "reverse"]) % 2 else "acw"

    # start at 7 cards, add any drawn cards, remove any played cards
    cw_hand_size = 7 + sum([h.num_cards for h in history if h.player == cw_num and h.action == 'draw']) \
                   - len([h for h in history if h.player == cw_num and h.action == 'play'])
    acw_hand_size = 7 + sum([h.num_cards for h in history if h.player == acw_num and h.action == 'draw']) \
                   - len([h for h in history if h.player == acw_num and h.action == 'play'])

    # based on direction of play, use cw or acw and next and next next
    next_uno = cw_hand_size == 1 if direction_of_play == 'cw' else acw_hand_size == 1
    next_next_uno = acw_hand_size == 1 if direction_of_play == 'acw' else cw_hand_size == 1

    # find the most recent play by the player before the most recent 'draw' that isn't 'black'
    cw_found_draw = False
    cw_last_draw_color = random.choice(list(color_indices.values()))
    for h in reversed(history):
        if h.player == cw_num and h.action == 'draw':
            cw_found_draw = True
        elif cw_found_draw and h.action == 'play' and h.played_card.color != "black":
            cw_last_draw_color = h.played_card.color
            break

    acw_found_draw = False
    acw_last_draw_color = random.choice(list(color_indices.values()))
    for h in reversed(history):
        if h.player == acw_num and h.action == 'draw':
            acw_found_draw = True
        elif acw_found_draw and h.action == 'play' and h.played_card.color != "black":
            acw_last_draw_color = color_indices[h.played_card.color]
            break

    # based on direction of play, use cw or acw and next and next next
    next_last_draw_color = cw_last_draw_color == 1 if direction_of_play == 'cw' else acw_last_draw_color == 1
    next_next_last_draw_color = acw_last_draw_color == 1 if direction_of_play == 'acw' else acw_last_draw_color == 1

    # convert matches to index and return
    return (next_next_last_draw_color
            + 4 * next_last_draw_color
            + 4 * 4 * next_next_uno
            + 4 * 4 * 2 * next_uno
            + 4 * 4 * 2 * 2 * draw_4
            + 4 * 4 * 2 * 2 * 2 * wild
            + 4 * 4 * 2 * 2 * 2 * 2 * match_reverse
            + 4 * 4 * 2 * 2 * 2 * 2 * 2 * match_skip
            + 4 * 4 * 2 * 2 * 2 * 2 * 2 * 2 * match_draw_2
            + 4 * 4 * 2 * 2 * 2 * 2 * 2 * 2 * 2 * match_color
            + 4 * 4 * 2 * 2 * 2 * 2 * 2 * 2 * 2 * 3 * match_num)


def test_table(q, num_episodes=1000):
    """
    Runs 1000 games, checks wins vs loses
    """
    avg_reward = 0
    wins = 0
    loses = 0

    for _ in range(num_episodes):

        obs, reward, done = env.reset()
        steps = 0

        while done == False and steps < 100:
            hs = hash(obs)

            action = np.argmax(q[hs])
            obs, reward, done = env.move(action)
            steps += 1

            avg_reward += reward

        if reward == 10000:
            wins += 1
        elif reward == -10000:
            loses += 1

    return avg_reward / num_episodes, wins, loses


def Q_learning(num_episodes=10000, gamma=0.9, epsilon=1, decay=0.999):
    Q = {}
    for i in range(375):
        Q[i] = np.zeros(NUM_ACTIONS)
    num_updates = np.zeros((375, NUM_ACTIONS))

    for episode in range(num_episodes):

        if episode % 10000 == 0:
            print("{} out of {} episodes. The Q table has {} entries, the exploration rate is {}".format(episode,
                                                                                                         num_episodes,
                                                                                                         len(Q.keys()),
                                                                                                         epsilon))
            avg, wins, loses = test_table(Q, 1000)
            print("The Q_Table got an average reward of {} with {} wins and {} loses for a win percentage of {}".format(
                avg, wins, loses, wins / (wins + loses) if wins + loses > 0 else 0))

        obs, reward, done = env.reset()
        moves = 0

        while not done:
            hs = hash(obs)

            if np.random.random() > epsilon:
                action = np.argmax(Q[hs])
            else:
                action = np.random.randint(0, NUM_ACTIONS)

            obs, reward, done = env.move(action)
            moves += 1

            eta = 1 / (1 + num_updates[hs, action])
            newQ = ((1 - eta) * Q[hs][action]) + (eta * (reward + (gamma * np.max(Q[hash(obs)]))))

            Q[hs][action] = newQ
            num_updates[hs, action] += 1

        epsilon *= decay
        # print("That game had {} steps and ended with result {}".format(moves, reward))

    return Q


decay_rate = 0.999995

Q_table = Q_learning(num_episodes=1000000, gamma=0.9, epsilon=1, decay=decay_rate)  # Run Q-learning
# Q_table = Q_learning(num_episodes=10000, gamma=0.9, epsilon=1, decay_rate=decay_rate) # Run Q-learning

# Save the Q-table dict to a file
with open('Q_table.pickle', 'wb') as handle:
    pickle.dump(Q_table, handle, protocol=pickle.HIGHEST_PROTOCOL)

'''
Uncomment the code below to play an episode using the saved Q-table. Useful for debugging/visualization.

Comment before final submission or autograder may fail.
'''

# Q_table = np.load('Q_table.pickle', allow_pickle=True)

# obs, reward, done, info = env.reset()
# total_reward = 0
# while not done:
# 	state = hash(obs)
# 	action = np.argmax(Q_table[state])
# 	obs, reward, done, info = env.step(action)
# 	total_reward += reward
# 	if gui_flag:
# 		refresh(obs, reward, done, info)  # Update the game screen [GUI only]

# print("Total reward:", total_reward)

# # Close the
# env.close() # Close the environment
