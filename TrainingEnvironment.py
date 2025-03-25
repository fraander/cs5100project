import random
import numpy as np

from RandomPlayer import RandomPlayer
from uno import UnoGame


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

    actions = {
        1: "match_color",
        2: "match_number",
        3: "skip/reverse/draw_2",
        4: "wild/draw_4_red",
        5: "wild/draw_4_yellow",
        6: "wild/draw_4_green",
        7: "wild/draw_4_blue",
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
        return card.card_type == current.card_type and current.playable(card)

    @staticmethod
    def matching_color(card, current):
        return (card.color == current.color or (current.color == "black" and card.color == current.temp_color)) \
            and card.card_type not in ['skip', 'reverse', '+2'] and current.playable(card)

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

    @staticmethod
    def can_draw_2(hand, current):
        return any([c.color == current.color and c.card_type == '+2' for c in hand])

    @staticmethod
    def can_skip(hand, current):
        return any([c.color == current.color and c.card_type == 'skip' for c in hand])

    @staticmethod
    def can_reverse(hand, current):
        return any([c.color == current.color and c.card_type == 'reverse' for c in hand])

    @staticmethod
    def can_wild(hand):
        return any(c.card_type == 'wildcard' for c in hand)

    @staticmethod
    def can_draw_4(hand):
        return any(c.card_type == '+4' for c in hand)

    def play_wild_draw_4(self, card, current):
        hand = self.game.current_player.hand  # get the current hand

        # get move if valid, otherwise get None
        wild = self.play_wild(card, current) if self.can_wild(hand) else None
        draw_4 = self.play_draw_four(card, current) if self.can_draw_4(hand) else None

        # build array of non-None moves for choosing at random
        choices = [choice for choice in [wild, draw_4] if choice is not None]

        # return a random valid choice; and try skip otherwise (since it gets rejected anyways)
        return self.play_wild(card, current) if len(choices) == 0 else random.choice(choices)

    def play_skip_reverse_draw_two(self, card, current):
        hand = self.game.current_player.hand  # get the current hand

        # get move if valid, otherwise get None
        skip = self.play_skip(card, current) if self.can_skip(hand, current) else None
        reverse = self.play_reverse(card, current) if self.can_reverse(hand, current) else None
        draw_two = self.play_draw_two(card, current) if self.can_draw_2(hand, current) else None

        # build array of non-None moves for choosing at random
        choices = [choice for choice in [skip, reverse, draw_two] if choice is not None]

        # return a random valid choice; and try skip otherwise (since it gets rejected anyways)
        return self.play_skip(card, current) if len(choices) == 0 else random.choice(choices)

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
            0: self.matching_color,
            1: self.matching_number,
            2: self.play_skip_reverse_draw_two,
            3: self.play_wild_draw_4,
            4: self.play_wild_draw_4,
            5: self.play_wild_draw_4,
            6: self.play_wild_draw_4,
        }
        return move_mapping.get(move, None)

    @staticmethod
    def choose_card_index(hand, current, filter_fn, move):
        best_card = None
        for idx, card in enumerate(hand):
            if filter_fn(card, current):
                if move < 3 and card.color == 'black':
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
        # print("Move", move)
        hand = self.game.current_player.hand
        current = self.game.current_card

        # Get the filtering function and intended new color for the move
        filter_fn = self.get_move_filter(move)
        if filter_fn is None:
            # print("That's not an action")
            obs = {"hand": hand, "current_card": current, "history": self.game.history,
                   "player_number": self.player_number, "direction": self.game._player_cycle._reverse}
            reward = self.rewards['wrong_card']
            done = not self.game.is_active
            return obs, reward, done

        # Choose a legal card index based on the filter
        card_index = self.choose_card_index(hand, current, filter_fn, move)
        if card_index is None:
            # print("No matching card", move, hand, current, current.color, current.temp_color)
            obs = {"hand": hand, "current_card": current, "history": self.game.history,
                   "player_number": self.player_number, "direction": self.game._player_cycle._reverse}
            reward = self.rewards['wrong_card']
            done = not self.game.is_active
            return obs, reward, done

        if move == 3:
            new_color = "red"
        elif move == 4:
            new_color = "yellow"
        elif move == 5:
            new_color = "green"
        elif move == 6:
            new_color = "blue"
        else:
            new_color = None
        # print(move, new_color)
        # Attempt to play the selected card
        try:
            self.game.play(player=self.player_number, card=card_index, new_color=new_color)
        except ValueError:
            # print("Error when playing card")
            obs = {"hand": hand, "current_card": current, "history": self.game.history,
                   "player_number": self.player_number, "direction": self.game._player_cycle._reverse}
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

        # force a loss if the player goes over 9 cards in their hand
        if len(self.game.current_player.hand) > 9:
            self.game._winner = True

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
