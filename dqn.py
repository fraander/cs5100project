import torch.nn as nn
import torch.optim as optim

class DQN(nn.Module):
    def __init__(self, input_dim=4*4*15+2, output_dim=61, lr=0.001):
        super(DQN, self).__init__()
        # 4 planes * 4 colors * 15 number/types + 2 counts = 240 + 2 counts = 242 inputs
        # 4 planes -> 0,1,2 in hand + 1 top card
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim)  # 61 actions for Uno
        )

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.criterion = nn.SmoothL1Loss()
        self.loss = None

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits

    def compute_loss(self, predicted, target):
        self.loss = self.criterion(predicted, target)
        return self.loss

    def update_params(self, clip_value):
        self.optimizer.zero_grad()
        self.loss.backward()
        nn.utils.clip_grad_norm_(self.parameters(), max_norm=clip_value)
        self.optimizer.step()
