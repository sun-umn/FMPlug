# stdlib
import gc
import logging
from typing import Iterable

# third party
import torch
from flow_matching.path import CondOTProbPath
from torch.nn.parallel import DistributedDataParallel
from torchmetrics.aggregation import MeanMetric

# first party
from fmplug.engine.scaling import NativeScaler
from fmplug.engine.utils import skewed_timestep_sample
from fmplug.layers.ema import EMA

# global variable
PRINT_FREQUENCY = 50


# Create logger
logger = logging.getLogger(__name__)


def train_one_epoch(
    model: torch.nn.Module,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    lr_schedule: torch.torch.optim.lr_scheduler.LRScheduler,
    device: torch.device,
    epoch: int,
    loss_scaler: NativeScaler,
):
    gc.collect()
    model.train(True)
    batch_loss = MeanMetric().to(device, non_blocking=True)
    epoch_loss = MeanMetric().to(device, non_blocking=True)

    accum_iter = 1
    loss_violation = 0
    path = CondOTProbPath()

    for data_iter_step, (samples, labels) in enumerate(data_loader):
        if data_iter_step % accum_iter == 0:
            optimizer.zero_grad()
            batch_loss.reset()

        samples = samples.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        if torch.rand(1) < 1.0:
            conditioning = {}
        else:
            conditioning = {"label": labels}

        # Scaling to [-1, 1] from [0, 1]
        samples = samples * 2.0 - 1.0
        noise = torch.randn_like(samples).to(device)

        if True:
            t = skewed_timestep_sample(samples.shape[0], device=device)
        else:
            t = torch.torch.rand(samples.shape[0]).to(device)

        path_sample = path.sample(t=t, x_0=noise, x_1=samples)
        x_t = path_sample.x_t
        u_t = path_sample.dx_t

        with torch.cuda.amp.autocast(enabled=True, dtype=torch.float32):
            model_output = model(x_t, t, extra=conditioning)
            diff = model_output - u_t
            loss = torch.pow(diff, 2).mean()

        batch_loss.update(loss)
        epoch_loss.update(loss)

        loss /= accum_iter

        # Loss scaler applies the optimizer when update_grad is set to true.
        # Otherwise just updates the internal gradient scales
        apply_update = (data_iter_step + 1) % accum_iter == 0
        loss_scaler(
            loss,
            optimizer,
            parameters=model.parameters(),
            update_grad=apply_update,
            clip_grad=2.5,
        )

        if apply_update and isinstance(model, EMA):
            model.update_ema()

        elif (
            apply_update
            and isinstance(model, DistributedDataParallel)
            and isinstance(model.module, EMA)
        ):
            model.module.update_ema()

        lr = optimizer.param_groups[0]["lr"]
        if data_iter_step % PRINT_FREQUENCY == 0:
            logger.info(
                f"Epoch {epoch} [{data_iter_step}/{len(data_loader)}]:"  # type: ignore
                f"loss = {batch_loss.compute()}, lr = {lr}"
            )

        # TODO: Add back in the evaluation

    lr_schedule.step()
    print(f"Number of loss violations: {loss_violation}")
    return {"loss": float(epoch_loss.compute().detach().cpu())}
