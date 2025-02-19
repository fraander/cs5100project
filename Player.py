class Player:

    def __init__(self):
        pass
    
    def take_turn(self, hand, current_card, history):
        """
        Takes one turn.

        Args:
            hand [UnoCard]: The cards in my hand right now.
            current_card (UnoCard): The most recently played card.

        Returns:
            int: The index of the card in my hand that I would like to play
            str: The color that I would like to change to, if I'm playing a 
                  black card. Can be None if not playing a black card.
        """
        return (0, None)