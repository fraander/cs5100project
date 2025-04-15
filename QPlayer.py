from Player import Player
import numpy as np
import pickle

PICKLE = "./pickle_final.pickle"

class QPlayer(Player):
    
    def __init__(self):
        with open(PICKLE, "rb") as file:
            self.Q = pickle.load(file)
        
        for i in range(len(self.Q)):
            self.Q[i] = [float(l.split("(")[1][:-1]) for l in self.Q[i][0]]

        super().__init__()

    @staticmethod
    def hash(obs) -> int:
        # unpack observation
        current, hand, history, player_num, direction = obs['current_card'], obs['hand'], obs['history'], obs['player_number'], obs['direction']

        color_inds = ["red", "blue", "green", "yellow"]
        if current.color == "blue":
            color_inds = color_inds[1:] + color_inds[:1]
        if current.color == "green":
            color_inds = color_inds[2:] + color_inds[:2]
        if current.color == "yellow":
            color_inds = color_inds[3:] + color_inds[:3]
        color_indices = {
            'none': 0,
            color_inds[0]: 1,
            color_inds[1]: 2,
            color_inds[2]: 3,
            color_inds[3]: 4,
        }

        my_cols = [0,0,0,0]
        for i in hand:
            if i.color != 'black':
                my_cols[color_indices[i.color]-1] += 1
        hand_max_color = np.argmax(my_cols)

        # calculate competitor ids
        cw_num = (player_num - 1) % 3
        acw_num = (player_num + 1) % 3

        # calculate each match
        match_num = 1 if any([c.card_type == current.card_type for c in hand]) else 0 # any card matches current card type
        if current.color == 'black':
            match_color = min(2, len([c for c in hand if c.color == current.temp_color])) # any card matches current card color
        else:
            match_color = min(2, len([c for c in hand if c.color == current.color]))

        wild = 1 if any(c.card_type == 'wildcard' for c in hand) else 0 # any wildcard in hand
        draw_4 = 1 if any(c.card_type == '+4' for c in hand) else 0 # any +4 in hand

        match_draw_2 = 1 if any([current.playable(c) and c.card_type == '+2' for c in hand]) else 0 # any +2 of color in hand
        match_skip = 1 if any([current.playable(c) and c.card_type == 'skip' for c in hand]) else 0 # any skip of color in hand
        match_reverse = 1 if any([current.playable(c) and c.card_type == 'reverse' for c in hand]) else 0 # any rev of color in hand

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
        next_next_uno = acw_hand_size == 1 if direction_of_play == 'cw' else cw_hand_size == 1

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
        if hand_max_color > 3:
            print("hand_max_color bad")

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
                + 5 * 5 * 2 * 2 * 2 * 2 * 2 * 2 * 2 * 3 * match_num
                + 5 * 5 * 2 * 2 * 2 * 2 * 2 * 2 * 2 * 3 * 2 * hand_max_color)

    @staticmethod
    def matching_number(card, current):
        return card.card_type == current.card_type and current.playable(card)
    
    @staticmethod
    def matching_color(card, current):
        if current.color == "black":
            return card.color == current.temp_color and current.playable(card)
        return card.color == current.color and current.playable(card)

    @staticmethod
    def play_skip(card, current):
        return card.card_type == 'skip' and current.playable(card)

    @staticmethod
    def play_reverse(card, current):
        return card.card_type == 'reverse' and current.playable(card)

    @staticmethod
    def play_draw_two(card, current):
        return card.card_type == '+2' and current.playable(card)

    @staticmethod
    def play_draw_four(card, current):
        return card.card_type == '+4' and current.playable(card)

    @staticmethod
    def play_wild(card, current):
        return card.card_type == 'wildcard' and current.playable(card)

    def get_move_filter(self, move):
        move_mapping = {
            0: self.matching_color,
            1: self.matching_number,
            2: self.play_skip,
            3: self.play_reverse,
            4: self.play_draw_two,
            5: self.play_draw_four,
            6: self.play_draw_four,
            7: self.play_draw_four,
            8: self.play_draw_four,
            9: self.play_wild,
            10: self.play_wild,
            11: self.play_wild,
            12: self.play_wild,
        }
        return move_mapping.get(move, None)

    @staticmethod
    def choose_card_index(hand, current, filter_fn, move):
        best_card = None
        for idx, card in enumerate(hand):
            if filter_fn(card, current):
                if move < 5 and card.color == 'black':
                    continue
                if card.card_type in ["+2"]:  
                    return idx  # Prioritize only the +2 action card 
                best_card = idx  
        return best_card  # Play normal card only if no action cards available

    def take_turn(self, hand, current_card, history, player_num, direction):

        hs = QPlayer.hash({'current_card': current_card, 'hand': hand, 'history': history, 'player_number': player_num, 'direction': direction})
        
        action = np.argmax(self.Q[hs])

        # Get the filtering function and intended new color for the move
        filter_fn = self.get_move_filter(action)

        # Choose a legal card index based on the filter
        card_index = self.choose_card_index(hand, current_card, filter_fn, action)
        if card_index is None:
            print("---------------------Returning early----------------------")
            playable = [card for card in hand if current_card.playable(card)]
            return hand.index(np.random.choice(playable)), np.random.choice(["red", "blue", "green", "yellow"])
        
        color_inds = ["red", "blue", "green", "yellow"]
        if current_card.color == "blue":
            color_inds = color_inds[1:] + color_inds[:1]
        if current_card.color == "green":
            color_inds = color_inds[2:] + color_inds[:2]
        if current_card.color == "yellow":
            color_inds = color_inds[3:] + color_inds[:3]
        
        if action == 5 or action == 9:
            new_color = color_inds[0]
        elif action == 6 or action == 10:
            new_color = color_inds[1]
        elif action == 7 or action == 11:
            new_color = color_inds[2]
        elif action == 8 or action == 12:
            new_color = color_inds[3]
        else:
            new_color = None
        
        actions = {
            0: "match the current color",
            1: "match the current number",
            2: "play a skip card",
            3: "play a reverse card",
            4: "play a draw 2",
            5: "play a draw 4 - switch to {}".format(new_color),
            6: "play a draw 4 - switch to {}".format(new_color),
            7: "play a draw 4 - switch to {}".format(new_color),
            8: "play a draw 4 - switch to {}".format(new_color),
            9: "play a wild - switch to {}".format(new_color),
            10: "play a wild - switch to {}".format(new_color),
            11: "play a wild - switch to {}".format(new_color),
            12: "play a wild - switch to {}".format(new_color)
        }

        print("Q Players turn:")
        print("State is {}, action selected is {}: {}".format(hs, action, actions[action]))

        return card_index, new_color