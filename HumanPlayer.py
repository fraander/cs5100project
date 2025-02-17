from Player import Player

class HumanPlayer(Player):

    def __init__(self):
        super().__init__()

    def take_turn(self, hand, current_card):
        
        possible_cards = [card for card in hand if current_card.playable(card)]

        print("The possible cards you can play are:", [str(c) for c in possible_cards])
        print("Enter the index of the card you would like to play:")
        choice = possible_cards[int(input())]

        if choice.color == "black":
            print("Which color would you like to switch too? (R, Y, G, B)")
            c = input().lower()
            match c:
                case "r":
                    new_color = "red"
                case "y":
                    new_color = "yellow"
                case "g":
                    new_color = "green"
                case "b":
                    new_color = "blue"
            print("Switching to", new_color)
        else:
            new_color = None

        return (hand.index(choice), new_color)