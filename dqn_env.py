import random
import numpy as np
from player.RandomPlayer import RandomPlayer
from uno.uno import UnoGame, COLORS
from collections import defaultdict

COLOR_MAP = {'red': 0, 'yellow': 1, 'green': 2, 'blue': 3}
VALUE_LIST = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'skip', 'reverse', '+2', 'wildcard', '+4']
VALUE_MAP = {trait: i for i, trait in enumerate(VALUE_LIST)}
ACTION_LIST = []
i = 0
for c in COLOR_MAP:
    for v in VALUE_LIST:
        action = f"{c}.{v}"
        ACTION_LIST.append(action)
ACTION_LIST.append('draw')
WILD = ['red.wildcard', 'yellow.wildcard', 'green.wildcard', 'blue.wildcard']
WILD_4 = ['red.+4', 'yellow.+4', 'green.+4', 'blue.+4']

class TrainingDQNEnv:
    def __init__(self, num_players=3):
        self.num_players = num_players
        self.ai_player = RandomPlayer()
        self.game = UnoGame(num_players)
        self.obs = None
        self.agent_id = 2

    def handle_other_players(self):
        while self.game.is_active and self.game.current_player.player_id != self.agent_id:
            player = self.game.current_player
            player_id = player.player_id
            if player.can_play(self.game.current_card):
                i, new_color = self.ai_player.take_turn(
                    hand=player.hand,
                    current_card=self.game.current_card,
                    history=self.game.history
                )
                self.game.play(player=player_id, card=i, new_color=new_color)
            else:
                self.game.play(player=player_id, card=None)

    '''
    To encode the current hand of our agent to 3 planes -> 0, 1, 2+ cards in hand with 4*15 diff cards
    For wild/+4 cards, add one count to each color
    '''
    def _encode_hand(self):
        plane = np.zeros((3, 4, 15), dtype=np.int8)
        plane[0, :, :] = 1  # Baseline
        counts = defaultdict(int)

        for card in self.obs['hand']:
            value_str = str(card.card_type)
            value_idx = VALUE_MAP[value_str]

            if card.color.lower() == 'black':
                for c in range(4):
                    counts[(c, value_idx)] += 1
            else:
                color_idx = COLOR_MAP[card.color.lower()]
                counts[(color_idx, value_idx)] += 1

        for (color_idx, value_idx), count in counts.items():
            for i in range(1, min(count + 1, 3)):
                plane[0, color_idx, value_idx] = 0
                plane[i, color_idx, value_idx] = 1
        return plane

    def _encode_current(self):
        plane = np.zeros((1, 4, 15), dtype=np.int8)
        card = self.game.current_card
        agent_hand = self.game.players[self.agent_id].hand

        if card.color.lower() == 'black':
            if not card.temp_color:
                best_color = max(COLORS, key=lambda c: sum(card.color.lower() == c for card in agent_hand))
                color_idx = COLOR_MAP[best_color.lower()]
            else:
                color_idx = COLOR_MAP[card.temp_color.lower()]
        else:
            color_idx = COLOR_MAP[card.color.lower()]
        value_idx = VALUE_MAP[str(card.card_type)]
        plane[0, color_idx, value_idx] = 1

        return plane

    """
    Convert self.obs into a 1D array of size 242:
      - (3,4,15) => 180 from _encode_hand()
      - (1,4,15) => 60 from _encode_current()
      - 2 scalars from self.obs["counts"] => 2 for a 3-player game
    """
    def get_inputs(self):

        hand_enc = self._encode_hand()
        current_enc = self._encode_current()
        hand_enc_flat = hand_enc.flatten()
        current_enc_flat = current_enc.flatten()
        counts = np.array(self.obs["counts"], dtype=np.float32)  # shape (2,) if 2 opponents

        # 180 + 60 + 2 = 242
        inputs = np.concatenate((hand_enc_flat, current_enc_flat, counts))

        return inputs

    def reset(self):
        self.game = UnoGame(self.num_players)
        agent_id = 2
        self.agent_id = agent_id

        self.handle_other_players()
        self.obs = {
            "hand": list(self.game.players[agent_id].hand),
            "current_card": self.game.current_card,
            "counts": [(len(self.game.players[i].hand) / 20.0)
                       for i in range(self.num_players) if i != agent_id]
        }
        reward = 0
        return self.obs, reward, False

    '''
    find the list of string of legal actions for current hand
    '''
    def get_legal_actions(self):
        hand = self.game.players[self.agent_id].hand
        possible_cards = [card for card in hand if self.game.current_card.playable(card)]
        legals = []
        for card in possible_cards:
            c, v = card.color, str(card.card_type)
            if c == 'black':
                if v == 'wildcard':
                    legals.extend(WILD)
                else:
                    legals.extend(WILD_4)
            else:
                legals.append(c + "." + v)

        # if no valid move
        if not legals:
            legals.append('draw')
        return legals

    def _execute_action(self, action):
        reward = 0
        legal_actions = self.get_legal_actions()
        if action == 'draw':
            self.game.play(player=self.agent_id, card=None)
            return reward
        if action not in legal_actions:
            action = random.choice(legal_actions)
            reward = -5.0
        if action == 'draw':
            self.game.play(player=self.agent_id, card=None)
            return reward

        color, value = action.split(".")
        card_index = None
        for i, card in enumerate(self.game.players[self.agent_id].hand):
            if card.color.lower() == 'black':
                if value == 'wildcard' and card.card_type == 'wildcard':
                    card_index = i
                    break
                if value == '+4' and card.card_type == '+4':
                    card_index = i
                    break
            else:
                if card.color.lower() == color and str(card.card_type) == value:
                    card_index = i
                    break

        if card_index is not None:
            self.game.play(
                player=self.agent_id,
                card=card_index,
                new_color=color
            )
        else:
            # If we somehow couldn't find the actual card, fallback draw
            print("Error: when playing card, cannot find the card, have to draw")
            self.game.play(player=self.agent_id, card=None)
        return reward

    def step(self, action):
        done = False
        reward = 0

        reward += self._execute_action(action)

        if not self.game.is_active:
            done = True
            if self.game.winner == self.game.players[self.agent_id]:
                reward += 1.0
            else:
                reward += -1.0
        else:
            self.handle_other_players()
            if not self.game.is_active:
                done = True
                if self.game.winner == self.game.players[self.agent_id]:
                    reward += 1.0
                else:
                    reward += -1.0
        counts = []
        p = 0
        for i in range(self.num_players):
            if i != self.agent_id:
                counts.append(len(self.game.players[i].hand) / 10.0)
                p += 1
        self.obs = {
            "hand": list(self.game.players[self.agent_id].hand),
            "current_card": self.game.current_card,
            "counts": counts
        }

        return self.obs, reward, done


if __name__ == "__main__":
    env = TrainingDQNEnv(num_players=3)
    num_episodes = 5

    for e in range(num_episodes):
        print(f"\n===== Starting Episode {e + 1} =====")
        obs, reward, done = env.reset()

        print("\n[reset()] returned:")
        print("Initial observation (dictionary):", obs)
        print("Initial reward:", reward)
        print("Initial done:", done)

        step_count = 0
        max_steps = 50
        total_reward = 0

        while not done and step_count < max_steps:
            # The env’s own method to get legal actions (strings like "red.3", "draw", etc.)
            legal_acts = env.get_legal_actions()
            print(f"\nStep {step_count} — Legal actions: {legal_acts}")
            print(f"\nCurrent card: {env.game.current_card}")

            # For testing, pick a random valid action:
            action = random.choice(legal_acts)
            print(f"Chosen action: {action}")

            # Step the environment
            obs_array, reward, done = env.step(action)

            step_count += 1

        # Episode finished
        print(f"Episode {e + 1} finished after {step_count} steps with final reward {reward}")