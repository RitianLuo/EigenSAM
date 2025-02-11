"""
https://github.com/davda54/sam
"""

import torch

class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(SAM, self).__init__(params, defaults)

        # Initialize base optimizer with the same parameters
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def first_step(self, zero_grad=False, normalization=True):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12) if normalization else torch.tensor(group["rho"])

            for p in group["params"]:
                if p.grad is None:
                    continue
                self.state[p]["old_p"] = p.data.clone()
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p.device)
                p.add_(e_w)  # Climb to the local maximum "w + e(w)"

        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False, update=True):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                p.data = self.state[p]["old_p"]  # Restore "w" from "w + e(w)"

        if update:
            self.base_optimizer.step()  # Perform the actual optimizer step

        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def normal_step(self, zero_grad=False):
        """Regular optimizer step without SAM modifications."""
        self.base_optimizer.step()
        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a Sharpness-Aware Minimization (SAM) optimization step."""
        assert closure is not None, "SAM requires a closure, but it was not provided."
        closure = torch.enable_grad()(closure)  # Ensure closure runs with gradients enabled

        self.first_step(zero_grad=True)
        closure()
        self.second_step(update=True)  # Ensure update happens

    def _grad_norm(self):
        """Computes the norm of the gradient for the SAM perturbation."""
        shared_device = self.param_groups[0]["params"][0].device
        norms = []
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    norm_val = ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
                    norms.append(norm_val)
        
        if len(norms) == 0:  # Edge case: no gradients
            return torch.tensor(0.0, device=shared_device)

        return torch.norm(torch.stack(norms), p=2)

    def load_state_dict(self, state_dict):
        """Loads the optimizer state."""
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups
