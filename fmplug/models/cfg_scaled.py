# third party
import torch
import torch.nn as nn
from flow_matching.utils import ModelWrapper


class CFGScaledModel(ModelWrapper):
    def __init__(self, model: nn.Module):
        super().__init__(model)
        self.nfe_counter = 0

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        cfg_scale: float,
        label: torch.Tensor,
        is_discrete: bool = False,
    ):
        assert cfg_scale == 0.0 or not is_discrete, (
            "Cfg scaling does not work for the logit outputs "
            f"of discrete models. Got cfg weight={cfg_scale} and "
            f"model {type(self.model)}."
        )

        t = torch.zeros(x.shape[0], device=x.device) + t

        if cfg_scale != 0.0:
            with (
                torch.cuda.amp.autocast(enabled=True, dtype=torch.float32),
                #                 torch.no_grad(),
            ):
                conditional = self.model(x, t, extra={"label": label})
                condition_free = self.model(x, t, extra={})

            result = (1.0 + cfg_scale) * conditional - cfg_scale * condition_free
            # cond + cfg * cond - cfg * uncond
            #
        else:
            # Model is fully conditional, no cfg weighting needed
            with (
                torch.cuda.amp.autocast(enabled=True, dtype=torch.float32),
                #                 torch.no_grad(),
            ):
                result = self.model(
                    x,
                    t,
                    extra={},
                )

        # NOTE: autocast here is float32 because there was bad
        # under and overflow with float16

        self.nfe_counter += 1

        return result.to(dtype=torch.float32)

    def reset_nfe_counter(self) -> None:
        self.nfe_counter = 0

    def get_nfe(self) -> int:
        return self.nfe_counter
