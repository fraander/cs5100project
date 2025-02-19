from uno import UnoGame, COLORS
import random
from AIPlayer import AIPlayer
from HumanPlayer import HumanPlayer

#players = random.randint(2, 15)
num_players = 4
players = [AIPlayer(), AIPlayer(), AIPlayer(), HumanPlayer()]
game = UnoGame(num_players)

print("Starting a {} player game".format(num_players))

count = 0
while game.is_active:
    count += 1
    player = game.current_player
    print("History", game.history)
    player_id = player.player_id
    if player.can_play(game.current_card):
        i, new_color = players[player_id].take_turn(player.hand, game.current_card, game.history)
 
        print("Player {} played {}".format(player, player.hand[i]))
        game.play(player=player_id, card=i, new_color=new_color)
    else:
        print("Player {} picked up".format(player))
        game.play(player=player_id, card=None)

print("{} player game - {} cards played".format(num_players, count))
