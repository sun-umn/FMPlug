import torch
import torch.nn as nn


# -----------------
# Bayesian regressor with separate heads
# -----------------
class BayesianRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1, 512)
        self.fc2 = nn.Linear(512, 512)

        # For mean target
        self.fc_mu_mean = nn.Linear(512, 1)
        self.fc_logvar_mean = nn.Linear(512, 1)

        # For log(var) target
        self.fc_mu_var = nn.Linear(512, 1)
        self.fc_logvar_var = nn.Linear(512, 1)

    def forward(self, t):
        h = torch.relu(self.fc1(t))
        h = torch.relu(self.fc2(h))

        mu_mean = self.fc_mu_mean(h)
        logvar_mean = self.fc_logvar_mean(h)  # unbounded

        mu_var = self.fc_mu_var(h)
        logvar_var = self.fc_logvar_var(h)  # unbounded

        return mu_mean, logvar_mean, mu_var, logvar_var





