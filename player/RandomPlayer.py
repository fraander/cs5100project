from uno.uno import COLORS
from player.Player import Player
import random

class RandomPlayer(Player):

    def __init__(self):
        super().__init__()

    def take_turn(self, hand, current_card, history):
        
        possible_cards = [card for card in hand if current_card.playable(card)]

        my_choice = random.choice(possible_cards)

        if my_choice.color == 'black':
            new_color = random.choice(COLORS)
        else:
            new_color = None

        return (hand.index(my_choice), new_color)