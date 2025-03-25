import pickle
from datetime import datetime, timedelta
import csv
import numpy as np

from TrainingEnvironment import TrainingEnvironment

env = TrainingEnvironment()
NUM_ACTIONS = 7
NUM_STATES = 51200

# TODO: Make it lose if it gets over 9 cards

LOGS = "./logs/logs3.23.csv"
PICKLE = "./pickles/pickle3.23.pickle"

color_indices = {
    'none': 0,
    'red': 1,
    'blue': 2,
    'green': 3,
    'yellow': 4,
}


# Open the Q-table from a file
def read_q(file_path=None):
    if file_path is not None:
        with open(file_path, "rb") as file:
            return pickle.load(file)
    return None


def hash(obs) -> int:
    # unpack observation
    current, hand, history, player_num, direction = obs['current_card'], obs['hand'], obs['history'], obs[
        'player_number'], obs['direction']

    my_cols = [0, 0, 0, 0]
    for i in hand:
        if i.color != 'black':
            my_cols[color_indices[i.color] - 1] += 1
    hand_max_color = np.argmax(my_cols)

    # calculate competitor ids
    cw_num = (player_num - 1) % 3
    acw_num = (player_num + 1) % 3

    # calculate each match
    match_num = 1 if any([c.card_type == current.card_type for c in hand]) else 0  # any card matches current card type
    match_color = min(1, len([c for c in hand if c.color == current.color and c.card_type not in ['skip', 'reverse',
                                                                                                  '+2']]))  # any card matches current card color

    wild_or_draw4 = wild = 1 if any(
        c.card_type == 'wildcard' or c.card_type == '+4' for c in hand) else 0  # any wildcard in hand

    match_draw_2_skip_or_rev = 1 if any(
        [c.color == current.color and (c.card_type == '+2' or c.card_type == 'skip' or c.card_type == 'reverse') for c
         in hand]) else 0  # any +2 of color in hand

    # clockwise (cw) is default, anti-clockwise (acw) is if an odd number of reverses have been played
    direction_of_play = "cw" if direction else "acw"

    # start at 7 cards, add any drawn cards, remove any played cards
    # print(history[0])
    cw_hand_size = 7 + sum([h['num_cards'] for h in history if h['player'] == cw_num and h['action'] == 'draw']) \
                   - len([h for h in history if h['player'] == cw_num and h['action'] == 'play'])
    acw_hand_size = 7 + sum([h['num_cards'] for h in history if h['player'] == acw_num and h['action'] == 'draw']) \
                    - len([h for h in history if h['player'] == acw_num and h['action'] == 'play'])

    # based on direction of play, use cw or acw and next and next next
    next_uno = cw_hand_size == 1 if direction_of_play == 'cw' else acw_hand_size == 1
    next_next_uno = acw_hand_size == 1 if direction_of_play == 'cw' else cw_hand_size == 1

    # find the most recent play by the player before the most recent 'draw' that isn't 'black'
    cw_found_draw = False
    cw_last_draw_color = 'none'  # random.choice(list(color_indices.values()))
    for h in reversed(history):
        if h['player'] == cw_num and h['action'] == 'draw':
            cw_found_draw = True
        elif cw_found_draw and h['action'] == 'play' and h['played_card'].color != "black":
            cw_last_draw_color = h['played_card'].color
            break

    acw_found_draw = False
    acw_last_draw_color = 'none'  # random.choice(list(color_indices.values()))
    for h in reversed(history):
        if h['player'] == acw_num and h['action'] == 'draw':
            acw_found_draw = True
        elif acw_found_draw and h['action'] == 'play' and h['played_card'].color != "black":
            acw_last_draw_color = h['played_card'].color
            break

    # based on direction of play, use cw or acw and next and next next
    next_last_draw_color = color_indices[cw_last_draw_color if direction_of_play == 'cw' else acw_last_draw_color]
    next_next_last_draw_color = color_indices[acw_last_draw_color if direction_of_play == 'cw' else cw_last_draw_color]

    if next_next_last_draw_color > 4:
        print('next_last_draw_color bad')
    if next_last_draw_color > 4:
        print('next_last_draw_color bad')
    if next_next_uno > 1:
        print('next_next_uno bad')
    if next_uno > 1:
        print('next_uno bad')
    # if draw_4 > 1:
    #     print('draw_4 bad')
    if wild > 1:
        print('wild bad')
    # if match_reverse > 1:
    #     print('match_reverse bad')
    # if match_skip > 1:
    #     print('match_skip bad')
    # if match_draw_2 > 1:
    #     print('match_draw_2 bad')
    if match_color > 2:
        print('match_color bad')
    if match_num > 1:
        print('match_num bad')
    if hand_max_color > 3:
        print("hand_max_color bad")

    # convert matches to index and return
    return (next_next_last_draw_color
            + 5 * next_last_draw_color
            + 5 * 5 * (1 if next_next_uno else 0)
            + 5 * 5 * 2 * (1 if next_uno else 0)
            + 5 * 5 * 2 * 2 * wild_or_draw4
            + 5 * 5 * 2 * 2 * 2 * match_draw_2_skip_or_rev
            + 5 * 5 * 2 * 2 * 2 * 2 * match_color
            + 5 * 5 * 2 * 2 * 2 * 2 * 2 * match_num
            + 5 * 5 * 2 * 2 * 2 * 2 * 2 * 2 * hand_max_color)


def test_table(q, num_episodes=1000):
    """
    Runs 1000 games, checks wins vs loses
    """
    avg_reward = 0
    wins = 0
    loses = 0
    illegal_moves = 0
    legal_moves = 0

    for _ in range(num_episodes):

        obs, reward, done = env.reset()
        steps = 0

        while done == False and steps < 100:
            hs = hash(obs)

            action = np.argmax(q[hs])
            obs, reward, done = env.move(action)
            steps += 1

            avg_reward += reward
            if reward == TrainingEnvironment.rewards['wrong_card']:
                illegal_moves += 1
            else:
                legal_moves += 1

        if reward == TrainingEnvironment.rewards['win']:
            wins += 1
        elif reward == TrainingEnvironment.rewards['lose']:
            loses += 1

    print("The table made {} legal and {} illegal moves".format(legal_moves, illegal_moves))
    return avg_reward / num_episodes, wins, loses, legal_moves, illegal_moves


def avg_score(Q):
    actions = [0] * NUM_ACTIONS
    taz = 0
    for i in Q.keys():
        all_zero = True
        for action in range(NUM_ACTIONS):
            actions[action] += Q[i][action]
            if Q[i][action] != 0:
                all_zero = False
        taz += 1 if all_zero else 0
    return [round(a / len(Q.keys()), 2) for a in actions], taz


def Q_learning(gamma=0.9, epsilon=1, decay=0.999, q_path=None):
    start = datetime.now()
    checkpoint = datetime.now()
    logging = []
    unique_states = set()
    episode = 0

    loaded = read_q(q_path)
    Q = loaded if loaded is not None else {}

    if loaded is None:
        for i in range(NUM_STATES):
            Q[i] = np.zeros(NUM_ACTIONS)
    num_updates = np.zeros((NUM_STATES, NUM_ACTIONS))

    while datetime.now() < start + timedelta(minutes=480):

        if datetime.now() > checkpoint + timedelta(minutes=5):
            print(
                "{} out of ?? episodes. The Q table has {} entries and has seen {} unique states, the exploration rate is {}".format(
                    episode,
                    len(Q.keys()),
                    len(unique_states),
                    epsilon))
            avg, wins, loses, legal, illegal = test_table(Q, 10000)
            print("The Q_Table got an average reward of {} with {} wins and {} loses for a win percentage of {}".format(
                avg, wins, loses, wins / (wins + loses) if wins + loses > 0 else 0))
            avg_scores, taz = avg_score(Q)
            print("Average Q score for each action: {}. There are {} states with no scores".format(avg_scores, taz))
            print()
            logging.append([datetime.now(), episode, epsilon, len(unique_states), legal, illegal,
                            legal / (legal + illegal) if legal + illegal > 0 else 0, avg, wins, loses,
                            wins / (wins + loses) if wins + loses > 0 else 0, taz])
            for i in avg_scores:
                logging[-1].append(i)
            checkpoint = datetime.now()
            epsilon *= decay
            with open(LOGS, 'w', newline='') as file:
                csvwriter = csv.writer(file)
                csvwriter.writerows(logging)

            # Save the Q-table dict to a file
            with open(PICKLE, 'wb') as handle:
                pickle.dump(Q, handle, protocol=pickle.HIGHEST_PROTOCOL)

        obs, reward, done = env.reset()
        moves = 0

        while not done:

            hs = hash(obs)

            unique_states.add(hs)

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

        episode += 1
        # print("That game had {} steps and ended with result {}".format(moves, reward))

    return Q, logging


decay_rate = 0.99

# Give the file path of the Q_table.pickle to load an existing Q_table
Q_table, logs = Q_learning(gamma=0.9, epsilon=1, decay=decay_rate, q_path=None)  # Run Q-learning
# Q_table = Q_learning(num_episodes=10000, gamma=0.9, epsilon=1, decay_rate=decay_rate) # Run Q-learning

with open(LOGS, 'w', newline='') as file:
    csvwriter = csv.writer(file)
    csvwriter.writerows(logs)

# Save the Q-table dict to a file
with open(PICKLE, 'wb') as handle:
    pickle.dump(Q_table, handle, protocol=pickle.HIGHEST_PROTOCOL)

'''
Uncomment the code below to play an episode using the saved Q-table. Useful for debugging/visualization.
'''

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
