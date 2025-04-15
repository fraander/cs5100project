import pickle
import numpy as np
from TrainingEnvironment import TrainingEnvironment
import csv

PICKLE = "./pickles/pickle3.22.pickle"
LOGS = "./logs/pickle_test_3.23.csv"
TEST_SIZE = 1000000 # number of games to play

color_indices = {
    'none': 0,
    'red': 1,
    'blue': 2,
    'green': 3,
    'yellow': 4,
}

def hash(obs) -> int:
    # unpack observation
    current, hand, history, player_num, direction = obs['current_card'], obs['hand'], obs['history'], obs['player_number'], obs['direction']

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
    direction_of_play = "cw" if direction else "acw"

    # start at 7 cards, add any drawn cards, remove any played cards
    #print(history[0])
    cw_hand_size = 7 + sum([h['num_cards'] for h in history if h['player'] == cw_num and h['action'] == 'draw']) \
                   - len([h for h in history if h['player'] == cw_num and h['action'] == 'play'])
    acw_hand_size = 7 + sum([h['num_cards'] for h in history if h['player'] == acw_num and h['action'] == 'draw']) \
                   - len([h for h in history if h['player'] == acw_num and h['action'] == 'play'])

    # based on direction of play, use cw or acw and next and next next
    next_uno = cw_hand_size == 1 if direction_of_play == 'cw' else acw_hand_size == 1
    next_next_uno = acw_hand_size == 1 if direction_of_play == 'acw' else cw_hand_size == 1

    # find the most recent play by the player before the most recent 'draw' that isn't 'black'
    cw_found_draw = False
    cw_last_draw_color = 'none' # random.choice(list(color_indices.values()))
    for h in reversed(history):
        if h['player'] == cw_num and h['action'] == 'draw':
            cw_found_draw = True
        elif cw_found_draw and h['action'] == 'play' and h['played_card'].color != "black":
            cw_last_draw_color = h['played_card'].color
            break

    acw_found_draw = False
    acw_last_draw_color = 'none' # random.choice(list(color_indices.values()))
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
    if draw_4 > 1:
        print('draw_4 bad')
    if wild > 1:
        print('wild bad')
    if match_reverse > 1:
        print('match_reverse bad')
    if match_skip > 1:
        print('match_skip bad')
    if match_draw_2 > 1:
        print('match_draw_2 bad')
    if match_color > 2:
        print('match_color bad')
    if match_num> 1:
        print('match_num bad')

    # convert matches to index and return
    return (next_next_last_draw_color
            + 5 * next_last_draw_color
            + 5 * 5 * (1 if next_next_uno else 0)
            + 5 * 5 * 2 * (1 if next_uno else 0)
            + 5 * 5 * 2 * 2 * draw_4
            + 5 * 5 * 2 * 2 * 2 * wild
            + 5 * 5 * 2 * 2 * 2 * 2 * match_reverse
            + 5 * 5 * 2 * 2 * 2 * 2 * 2 * match_skip
            + 5 * 5 * 2 * 2 * 2 * 2 * 2 * 2 * match_draw_2
            + 5 * 5 * 2 * 2 * 2 * 2 * 2 * 2 * 2 * match_color
            + 5 * 5 * 2 * 2 * 2 * 2 * 2 * 2 * 2 * 3 * match_num)

with open(PICKLE, "rb") as file:
    Q = pickle.load(file)

wins = 0
loses = 0
env = TrainingEnvironment()
logs = []

for episode in range(TEST_SIZE):
    obs, reward, done = env.reset()
    moves = 0

    while not done:

        hs = hash(obs)

        action = np.argmax(Q[hs])

        obs, reward, done = env.move(action)
        
        if reward == TrainingEnvironment.rewards['wrong_card']:
            done = True
            loses += 1
        elif reward == TrainingEnvironment.rewards['lose']:
            done = True
            loses += 1
        elif reward == TrainingEnvironment.rewards['win']:
            done = True
            wins += 1
    
    logs.append([episode, wins, loses, wins / (wins + loses) if wins + loses > 0 else 0])

with open(LOGS, 'w', newline='') as file:
    csvwriter = csv.writer(file)
    csvwriter.writerows(logs)

print(wins/(wins+loses))
