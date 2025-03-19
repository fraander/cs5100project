import numpy as np

from RandomPlayer import RandomPlayer
from uno import UnoGame


# TODO: Rahul, update action space to match `todo.md`

def matching_color(card, current):
    return (card.color == current.color or current.color == 'black') and current.playable(card)


class TrainingEnvironment:
    num_players = 3

    rewards = {
        'play_card': 10,
        'wrong_card': 0,
        'two_left': 100,
        'uno': 500,
        'win': 10000,
        'lose': -10000,
    }

    actions = {
        1: "match_color",
        2: "match_number",
        3: "skip",
        4: "reverse",
        5: "draw_2",
        6: "draw_4",
        7: "wild"
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

    @staticmethod
    def matching_number(card, current):
        return (card.card_type == current.card_type or current.color == 'black') and current.playable(card)

    def matching_color_and_number(self, card, current):
        return matching_color(card, current) and self.matching_number(card, current)

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
        move_mapping = {
            1: (matching_color, None),
            2: (self.matching_number, None),
            3: (self.play_skip, None),
            4: (self.play_reverse, None),
            5: (self.play_draw_two, None),
            6: (self.play_draw_four, None),
            7: (self.play_wild, None),
        }
        return move_mapping.get(move, (None, None))

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
        hand = self.game.current_player.hand
        current = self.game.current_card

        # Get the filtering function and intended new color for the move
        filter_fn, intended_color = self.get_move_filter(move)
        if filter_fn is None:
            obs = {"hand": hand, "current_card": current, "history": self.game.history, "player_number": self.player_number, "direction": self.game._player_cycle._reverse}
            reward = self.rewards['wrong_card']
            done = not self.game.is_active
            return obs, reward, done

        # Choose a legal card index based on the filter
        card_index = self.choose_card_index(hand, current, filter_fn, move)
        if card_index is None:
            obs = {"hand": hand, "current_card": current, "history": self.game.history, "player_number": self.player_number, "direction": self.game._player_cycle._reverse}
            reward = self.rewards['wrong_card']
            done = not self.game.is_active
            return obs, reward, done

        # Attempt to play the selected card
        try:
            self.game.play(player=self.player_number, card=card_index, new_color=intended_color)
        except ValueError:
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

        reward = 0
        if not self.game.is_active:
            if self.game.winner.player_id == self.player_number:
                reward = self.rewards['win']
            else:
                reward = self.rewards['lose']
        done = not self.game.is_active
        return obs, reward, done
