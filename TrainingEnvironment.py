import numpy as np

from RandomPlayer import RandomPlayer
from uno import UnoGame, COLORS

class TrainingEnvironment:
    """
    Handles player moves in the appropriate order. Gives the appropriate reward to the AI agent after the AI Player goes.
    """


    num_players = 3

    rewards = {
        'play_card': 10,
        'wrong_card': -50000,
        'two_left': 100,
        'uno': 500,
        'win': 10000,
        'lose': -10000,
    }


    def reset(self):
        """
        Reset the game
        """
        
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


    def handle_other_players(self):
        """
        Runt through actions of non-AI players.
        """
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
        """
        Called during reinforcement with the QPlayer's move to make.
        """
    
        hand = self.game.current_player.hand
        current = self.game.current_card

        # Get the filtering function and intended new color for the move
        filter_fn = QPlayer.get_move_filter(move)
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
        
        color_inds = ["red", "blue", "green", "yellow"]
        if current.color == "blue":
            color_inds = color_inds[1:] + color_inds[:1]
        if current.color == "green":
            color_inds = color_inds[2:] + color_inds[:2]
        if current.color == "yellow":
            color_inds = color_inds[3:] + color_inds[:3]
        
        if move == 5 or move == 9:
            new_color = color_inds[0]
        elif move == 6 or move == 10:
            new_color = color_inds[1]
        elif move == 7 or move == 11:
            new_color = color_inds[2]
        elif move == 8 or move == 12:
            new_color = color_inds[3]
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
