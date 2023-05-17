import torch
import torch.nn as nn
import torch.nn.functional as F

class ICMModel(nn.Module):
    def __init__(self, state_dim, context_dim, action_dim):
        super(ICMModel, self).__init__()

        self.state_dim = state_dim + context_dim
        self.action_dim = action_dim

        self.inverse_net = nn.Sequential(
            nn.Linear(self.state_dim * 2, 64),
            nn.LeakyReLU(),
            nn.Linear(64, self.action_dim),
        ).cuda()

        self.forward_net1 = nn.Sequential(
            nn.Linear(self.action_dim + self.state_dim, 64),
            nn.LeakyReLU(),
            nn.Linear(64, self.state_dim),
            nn.LeakyReLU(),
        ).cuda()

        self.forward_net2 = nn.Sequential(
            nn.Linear(self.action_dim + self.state_dim, 64),
            nn.LeakyReLU(),
            nn.Linear(64, self.state_dim),
        ).cuda()

        for p in self.modules():
            if isinstance(p, nn.Linear):
                nn.init.kaiming_uniform_(p.weight, a=1.0)
                p.bias.data.zero_()

    def forward(self, state, next_state, action, return_reward=False):
        pred_action = self.inverse_net(torch.cat((state, next_state), dim=-1))

        pred_next_state = self.forward_net1(torch.cat((state, action), dim=-1))
        pred_next_state = self.forward_net2(torch.cat((pred_next_state, action), dim=-1))

        if return_reward:
            intrisic_reward = F.mse_loss(next_state, pred_next_state, reduction='none')
            return pred_action, pred_next_state, intrisic_reward
        else:
            return pred_action, pred_next_state

