import torch


class SAM(torch.optim.Optimizer):
    """Sharpness-Aware Minimization wrapper around a base optimizer."""

    def __init__(
        self,
        params,
        base_optimizer_cls,
        rho: float = 0.05,
        adaptive: bool = False,
        **kwargs,
    ):
        if rho < 0.0:
            raise ValueError(f"Invalid SAM rho: {rho}")

        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super().__init__(params, defaults)

        self.base_optimizer = base_optimizer_cls(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def first_step(self, zero_grad: bool = False):
        grad_norm = self._grad_norm()
        scale = 0.0 if grad_norm == 0.0 else 1.0 / (grad_norm + 1e-12)

        for group in self.param_groups:
            rho = group["rho"]
            adaptive = group["adaptive"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                self.state[p]["old_p"] = p.data.clone()
                e_w = p.grad * (rho * scale)
                if adaptive:
                    e_w = e_w * p.abs()
                p.add_(e_w)

        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad: bool = False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                old_p = self.state[p].pop("old_p", None)
                if old_p is None:
                    continue
                p.data.copy_(old_p)

        self.base_optimizer.step()

        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        if closure is None:
            raise RuntimeError("SAM requires a closure for step().")

        closure = torch.enable_grad()(closure)
        loss = closure()
        self.first_step(zero_grad=True)
        closure()
        self.second_step(zero_grad=True)
        return loss

    def zero_grad(self, set_to_none: bool = False):
        self.base_optimizer.zero_grad(set_to_none=set_to_none)

    def state_dict(self):
        return {
            "base_optimizer": self.base_optimizer.state_dict(),
            "state": self.state,
            "param_groups": self.param_groups,
        }

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.load_state_dict(state_dict["base_optimizer"])
        self.param_groups = self.base_optimizer.param_groups

    def add_param_group(self, param_group):
        super().add_param_group(param_group)
        if hasattr(self, "base_optimizer"):
            self.base_optimizer.add_param_group(param_group)
            self.param_groups = self.base_optimizer.param_groups

    def _grad_norm(self) -> float:
        shared_device = self.param_groups[0]["params"][0].device
        norms = []
        for group in self.param_groups:
            adaptive = group["adaptive"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if adaptive:
                    grad = grad * p.abs()
                norms.append(grad.norm(p=2).to(shared_device))
        if not norms:
            return 0.0
        return torch.norm(torch.stack(norms), p=2).item()
