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
    "hand": self.game.current_player.hand,
    "current_card": self.game.current_card,
    "history": self.game.history,
    "player_number": self.player_number,
    "cards_p1": len(self.game.players[0].hand),
    "cards_p2": len(self.game.players[1].hand),
    "cards_p3": len(self.game.players[2].hand),
    "last_color_p1": self.game.players[0].last_color,
    "last_color_p2": self.game.players[1].last_color,
    "last_color_p3": self.game.players[2].last_color,
    "last_number_p1": self.game.players[0].last_number,
    "last_number_p2": self.game.players[1].last_number,
    "last_number_p3": self.game.players[2].last_number,
    "direction": self.game.direction,
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
    def play_black(card, current):
        return card.color == 'black' and current.playable(card)

    # Moves are:
    # 1 - play the matching color
    # 2 - play the matching number
    # 3 - play the matching color and number
    # 4 - play a skip
    # 5 - play a reverse
    # 6 - play a black card, change color to RED
    # 7 - play a black card, change color to GREEN
    # 8 - play a black card, change color to YELLOW
    # 9 - play a black card, change color to BLUE
    def get_move_filter(self, move):
        move_mapping = {
            1: (matching_color, None),
            2: (self.matching_number, None),
            3: (self.play_skip, None),
            4: (self.play_reverse, None),
            5: (self.play_black, None),
            6: (self.play_black, 'red'),
            7: (self.play_black, 'green'),
            8: (self.play_black, 'blue'),
            9: (self.play_black, 'yellow'),
        }
        return move_mapping.get(move, (None, None))

    @staticmethod
    def choose_card_index(hand, current, filter_fn, move):
        for idx, card in enumerate(hand):
            if filter_fn(card, current):
                # For moves 0-4, avoid playing a black card.
                if move < 5 and card.color == 'black':
                    continue
                return idx
        return None

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
            obs = {"hand": hand, "current_card": current, "history": self.game.history}
            reward = self.rewards['wrong_card']
            done = not self.game.is_active
            return obs, reward, done

        # Choose a legal card index based on the filter
        card_index = self.choose_card_index(hand, current, filter_fn, move)
        if card_index is None:
            obs = {"hand": hand, "current_card": current, "history": self.game.history}
            reward = self.rewards['wrong_card']
            done = not self.game.is_active
            return obs, reward, done

        # Attempt to play the selected card
        try:
            self.game.play(player=self.player_number, card=card_index, new_color=intended_color)
        except ValueError:
            obs = {"hand": hand, "current_card": current, "history": self.game.history}
            reward = self.rewards['wrong_card']
            done = not self.game.is_active
            return obs, reward, done

        # Process turns for other players
        self.handle_other_players()

        # Build the new observation 
        obs = {
    "hand": self.game.current_player.hand,
    "current_card": self.game.current_card,
    "history": self.game.history,
    "cards_p1": len(self.game.players[0].hand),
    "cards_p2": len(self.game.players[1].hand),
    "cards_p3": len(self.game.players[2].hand),
    "last_color_p1": self.game.players[0].last_color,
    "last_color_p2": self.game.players[1].last_color,
    "last_color_p3": self.game.players[2].last_color,
    "last_number_p1": self.game.players[0].last_number,
    "last_number_p2": self.game.players[1].last_number,
    "last_number_p3": self.game.players[2].last_number,
    "direction": self.game.direction,
}

        reward = 0
        if not self.game.is_active:
            if self.game.winner.player_id == self.player_number:
                reward = self.rewards['win']
            else:
                reward = self.rewards['lose']
        done = not self.game.is_active
        return obs, reward, done
