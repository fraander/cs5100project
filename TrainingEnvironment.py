import numpy as np

from RandomPlayer import RandomPlayer
from uno import UnoGame, COLORS


class TrainingEnvironment:
    num_players = 3

    rewards = {
        'play_card': 10,
        'wrong_card': -50000,
        'two_left': 100,
        'uno': 500,
        'win': 10000,
        'lose': -10000,
    }

    # TODO: Update (FRANK)
    actions = {
        1: "match_color",
        2: "match_number",
        3: "skip",
        4: "reverse",
        5: "draw_2",
        6: "draw_4_red",
        7: "draw_4_yellow",
        8: "draw_4_green",
        9: "draw_4_blue",
        10: "wild_red",
        11: "wild_yellow",
        12: "wild_green",
        13: "wild_blue"
    }


    def reset(self):
        self.game = UnoGame(self.num_players)
        self.player_number = np.random.randint(0, self.num_players)

        self.ai_player = RandomPlayer()

        self.handle_other_players()

        obs = {
    "current_card": self.game.current_card,
    "hand": self.game.current_player.hand,
    "history": self.game.history,
    "player_number": self.player_number,
    "direction": self.game._player_cycle._reverse,
}

        reward = 0
        done = not self.game.is_active
        return obs, reward, done

    '''
    helper methods for actions
    '''

    # TODO: Update these functions
    @staticmethod
    def matching_number(card, current):
        return card.card_type == current.card_type and current.playable(card)
    
    @staticmethod
    def matching_color(card, current):
        return (card.color == current.color or (current.color=="black" and card.color == current.temp_color)) and card.card_type not in ['skip', 'reverse', '+2'] and current.playable(card)

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

    # Moves are:
    # 1 - play the matching color
    # 2 - play the matching number
    # 3 - play a skip
    # 4 - play a reverse
    # 5 - play a draw 2
    # 6 - play a draw 4
    # 7 - play a wildcard
    def get_move_filter(self, move):
        # TODO: Update (FRANK)
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

    def handle_other_players(self):
        while self.game.is_active and (
                self.game.current_player.player_id != self.player_number or
                not self.game.current_player.can_play(self.game.current_card)
        ):
            player = self.game.current_player
            player_id = player.player_id
            if player.can_play(self.game.current_card):
                i, new_color = self.ai_player.take_turn(player.hand, self.game.current_card, self.game.history)
                self.game.play(player=player_id, card=i, new_color=new_color)
            else:
                self.game.play(player=player_id, card=None)

    def move(self, move):
        #print("Move", move)
        hand = self.game.current_player.hand
        current = self.game.current_card

        # Get the filtering function and intended new color for the move
        filter_fn = self.get_move_filter(move)
        if filter_fn is None:
            #print("Thats not an action")
            obs = {"hand": hand, "current_card": current, "history": self.game.history, "player_number": self.player_number, "direction": self.game._player_cycle._reverse}
            reward = self.rewards['wrong_card']
            done = not self.game.is_active
            return obs, reward, done

        # Choose a legal card index based on the filter
        card_index = self.choose_card_index(hand, current, filter_fn, move)
        if card_index is None:
            #print("No matching card", move, hand, current, current.color, current.temp_color)
            obs = {"hand": hand, "current_card": current, "history": self.game.history, "player_number": self.player_number, "direction": self.game._player_cycle._reverse}
            reward = self.rewards['wrong_card']
            done = not self.game.is_active
            return obs, reward, done

        # TODO: Update FRANK
        if move == 5 or move == 9:
            new_color = "red"
        elif move == 6 or move == 10:
            new_color = "yellow"
        elif move == 7 or move == 11:
            new_color = "green"
        elif move == 8 or move == 12:
            new_color = "blue"
        else:
            new_color = None
        #print(move, new_color)
        # Attempt to play the selected card
        try:
            self.game.play(player=self.player_number, card=card_index, new_color=new_color)
        except ValueError:
            #print("Error when playing card")
            obs = {"hand": hand, "current_card": current, "history": self.game.history, "player_number": self.player_number, "direction": self.game._player_cycle._reverse}
            reward = self.rewards['wrong_card']
            done = not self.game.is_active
            return obs, reward, done

        # Process turns for other players
        self.handle_other_players()

        # Build the new observation 
        obs = {
            "current_card": self.game.current_card,
            "hand": self.game.current_player.hand,
            "history": self.game.history,
            "player_number": self.player_number,
            "direction": self.game._player_cycle._reverse,
        }

        if not self.game.is_active:
            if self.game.winner.player_id == self.player_number:
                reward = self.rewards['win']
            else:
                reward = self.rewards['lose']
        else:
            if len(self.game.current_player.hand) == 1:
                reward = self.rewards['uno']
            elif len(self.game.current_player.hand) == 2:
                reward = self.rewards['two_left']
            else:
                reward = self.rewards['play_card']
        done = not self.game.is_active
        return obs, reward, done
