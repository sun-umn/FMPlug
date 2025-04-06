# stdlib
import logging
from typing import List

# third party
import torch
import torch.nn as nn

# Create a logger for this module
logger = logging.getLogger(__name__)


class EMA(nn.Module):
    def __init__(self, model: nn.Module, decay: float = 0.999):
        super().__init__()
        self.model = model
        self.decay = decay

        # Put this in a buffer so that it gets included in the state dict
        self.register_buffer("num_updates", torch.tensor(0))

        self.shadow_params: nn.ParameterList = nn.ParameterList(
            [
                nn.Parameter(p.clone().detach(), requires_grad=False)
                for p in model.parameters()
                if p.requires_grad
            ]
        )
        self.backup_params: List[torch.Tensor] = []

    def train(self, mode: bool) -> None:  # type: ignore
        if self.training == mode:
            super().train(mode)  # type: ignore
            return

        if not mode:
            logger.info(
                "EMA: Switching from train to eval, backing up "
                "parameters and copying EMA params"
            )
            self.backup()
            self.copy_to_model()
        else:
            logger.info("EMA: Switching from eval to train,restoring saved parameters")
            self.restore_to_model()

        super().train(mode)

    def update_ema(self) -> None:
        self.num_updates += 1
        num_updates = self.num_updates.item()
        decay = min(self.decay, (1 + num_updates) / (10 + num_updates))

        with torch.no_grad():
            params = [p for p in self.model.parameters() if p.requires_grad]
            for shadow, param in zip(self.shadow_params, params):
                shadow.sub_((1 - decay) * (shadow - param))

    def forward(self, *args, **kwargs) -> torch.Tensor:
        return self.model(*args, **kwargs)

    def copy_to_model(self) -> None:
        params = [p for p in self.model.parameters() if p.requires_grad]
        for shadow, param in zip(self.shadow_params, params):
            param.data.copy_(shadow.data)

    def backup(self) -> None:
        assert (
            self.training
        ), "Backup can only be created in train mode to avoid backing-up ema weights."
        if len(self.backup_params) > 0:
            for p, b in zip(self.model.parameters(), self.backup_params):
                b.data.copy_(p.data)

        else:
            self.backup_params = [param.clone() for param in self.model.parameters()]

    def restore_to_model(self) -> None:
        for param, backup in zip(self.model.parameters(), self.backup_params):
            param.data.copy_(backup.data)
