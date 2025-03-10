from uno import UnoGame, COLORS
import numpy as np
from AIPlayer import AIPlayer

class TrainingEnvironment:

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

    def move(self, move):
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
        
        card = None
        if not self.game.current_player.can_play(self.game.current_card):
            print(self.game.current_card)
            raise ValueError("Can't play bro")
        match move:
            case 0:
                for n, c in enumerate(self.game.current_player.hand):
                    if (c.color == self.game.current_card.color or self.game.current_card.color == 'black') and self.game.current_card.playable(c):
                        card = n
                new_color = None
            case 1:
                for n, c in enumerate(self.game.current_player.hand):
                    if (c.card_type == self.game.current_card.card_type or self.game.current_card.color == 'black') and self.game.current_card.playable(c):
                        card = n
                new_color = None
            case 2:
                for n, c in enumerate(self.game.current_player.hand):
                    if (c.card_type == self.game.current_card.card_type or self.game.current_card.color == 'black') and c.color == self.game.current_card.color and self.game.current_card.playable(c):
                        card = n
                new_color = None
            case 3:
                for n, c in enumerate(self.game.current_player.hand):
                    if c.card_type == 'skip' and self.game.current_card.playable(c):
                        card = n
                new_color = None
            case 4:
                for n, c in enumerate(self.game.current_player.hand):
                    if c.card_type == 'reverse' and self.game.current_card.playable(c):
                        card = n
                new_color = None
            case 5:
                for n, c in enumerate(self.game.current_player.hand):
                    if c.color == 'black' and self.game.current_card.playable(c):
                        card = n
                new_color = 'red'
            case 6:
                for n, c in enumerate(self.game.current_player.hand):
                    if c.color == 'black' and self.game.current_card.playable(c):
                        card = n
                new_color = 'yellow'
            case 7:
                for n, c in enumerate(self.game.current_player.hand):
                    if c.color == 'black' and self.game.current_card.playable(c):
                        card = n
                new_color = 'green'
            case 8:
                for n, c in enumerate(self.game.current_player.hand):
                    if c.color == 'black' and self.game.current_card.playable(c):
                        card = n
                new_color = 'blue'

        if card == None or (move < 5 and self.game.current_player.hand[card].color == 'black'):
            #print("Invalid play", card, move)
            obs = {"hand": self.game.current_player.hand,
               "current_card": self.game.current_card,
               "history": self.game.history}
        
            reward = self.rewards['wrong_card']
            done = not self.game.is_active
            return obs, reward, done

        self.game.play(player=self.player_number, card = card, new_color=new_color)

        while self.game.is_active and (self.game.current_player.player_id != self.player_number or not self.game.current_player.can_play(self.game.current_card)):
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
        if not self.game.is_active:
            if self.game.winner.player_id == self.player_number:
                reward = self.rewards['win']
            else:
                reward = self.rewards['lose']
        done = not self.game.is_active
        return obs, reward, done
    

        