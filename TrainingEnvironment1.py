from uno import UnoGame, COLORS
import numpy as np
from AIPlayer import AIPlayer

class TrainingEnvironment1:

    num_players = 4

    rewards = {
        'wrong_card': -10,
        'win': 10000,
        'lose': -10000,
    }

    def reset(self):
        self.game = UnoGame(self.num_players)
        self.player_number = np.random.randint(0, self.num_players)
        
        self.ai_player = AIPlayer()

        while self.game.is_active and (self.game.current_player.player_id != self.player_number or not self.game.current_player.can_play(self.game.current_card)):
            #print(self.game.is_active, self.game.current_player.player_id, self.player_number)
            player = self.game.current_player
            
            # print("History", game.history)
            player_id = player.player_id
            if player.can_play(self.game.current_card):
                i, new_color = self.ai_player.take_turn(player.hand, self.game.current_card, self.game.history)
                self.game.play(player=player_id, card=i, new_color=new_color)
            else:
                self.game.play(player=player_id, card=None)

        obs = {"hand": self.game.current_player.hand,
               "current_card": self.game.current_card,
               "history": self.game.history}
        reward = 0
        done = not self.game.is_active
        return obs, reward, done
    

    '''
    helper methods for actions
    '''
    def matching_color(self, card, current):
        return (card.color == current.color or current.color == 'black') and current.playable(card)

    def matching_number(self, card, current):
        return (card.card_type == current.card_type or current.color == 'black') and current.playable(card)
    
    def matching_color_and_number(self, card, current):
        return self.matching_color(card, current) and self.matching_number(card, current)

    def play_skip(self, card, current):
        return card.card_type == 'skip' and current.playable(card)

    def play_reverse(self, card, current):
        return card.card_type == 'reverse' and current.playable(card)

    def play_black(self, card, current):
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
            0: (self.matching_color, None),
            1: (self.matching_number, None),
            2: (self.matching_color_and_number, None),
            3: (self.play_skip, None),
            4: (self.play_reverse, None),
            5: (self.play_black, 'red'),
            6: (self.play_black, 'yellow'),
            7: (self.play_black, 'green'),
            8: (self.play_black, 'blue'),
        }
        return move_mapping.get(move, (None, None))
    
    def choose_card_index(self, hand, current, filter_fn, move):
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
        obs = {"hand": self.game.current_player.hand,
               "current_card": self.game.current_card,
               "history": self.game.history}
        reward = 0
        if not self.game.is_active:
            if self.game.winner.player_id == self.player_number:
                reward = self.rewards['win']
            else:
                reward = self.rewards['lose']
        done = not self.game.is_active
        return obs, reward, done
    

        