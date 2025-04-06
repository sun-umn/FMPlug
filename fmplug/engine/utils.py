# third party
import torch


def get_grad_norm_(parameters, norm_type: float = 2.0) -> torch.Tensor:
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)

    if len(parameters) == 0:
        return torch.Tensor(0.0)

    device = parameters[0].grad.device
    if norm_type == torch.inf:
        total_norm = max(p.grad.detach().abs().max().to(device) for p in parameters)

    else:
        total_norm = torch.norm(
            torch.stack(
                [torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]
            ),
            norm_type,
        )

    return total_norm


def skewed_timestep_sample(num_samples: int, device: torch.device) -> torch.Tensor:
    P_mean = -1.2
    P_std = 1.2
    rnd_normal = torch.randn((num_samples,), device=device)
    sigma = (rnd_normal * P_std + P_mean).exp()
    time = 1 / (1 + sigma)
    time = torch.clip(time, min=0.0001, max=1.0)

    return time


def get_time_discretization(nfes: int, rho=7):
    step_indices = torch.arange(nfes, dtype=torch.float64)
    sigma_min = 0.002
    sigma_max = 80.0
    sigma_vec = (
        sigma_max ** (1 / rho)
        + step_indices / (nfes - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))
    ) ** rho
    sigma_vec = torch.cat([sigma_vec, torch.zeros_like(sigma_vec[:1])])
    time_vec = (sigma_vec / (1 + sigma_vec)).squeeze()
    t_samples = 1.0 - torch.clip(time_vec, min=0.0, max=1.0)
    return t_samples
