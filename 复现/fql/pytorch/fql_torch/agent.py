"""Faithful state-only PyTorch FQL; CPU and Apple MPS share the same algorithm."""

from __future__ import annotations

import copy
from collections.abc import Mapping
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

from .networks import ActorVectorField, Value


def get_config() -> dict[str, Any]:
    """Pinned official state architecture and selected cube-single alpha=300."""
    return {
        "agent_name": "fql", "ob_dims": (28,), "action_dim": 5,
        "lr": 3e-4, "batch_size": 256,
        "actor_hidden_dims": (512, 512, 512, 512),
        "value_hidden_dims": (512, 512, 512, 512),
        "layer_norm": True, "actor_layer_norm": False,
        "discount": 0.99, "tau": 0.005, "q_agg": "mean", "alpha": 300.0,
        "flow_steps": 10, "normalize_q_loss": False, "encoder": None,
        "optimizer_impl": "foreach",
    }


class FQLAgent(nn.Module):
    """Mutable PyTorch agent; update returns detached device tensors, without sync."""

    def __init__(self, config: Mapping[str, Any] | None = None,
                 device: str | torch.device = "cpu", seed: int = 0):
        super().__init__()
        self.config = get_config()
        self.config.update(dict(config or {}))
        if self.config["encoder"] is not None:
            raise ValueError("This port supports state observations only (encoder=None).")
        if tuple(self.config["ob_dims"]) != (int(self.config["ob_dims"][0]),):
            raise ValueError("This port requires one-dimensional state observations.")
        if self.config["q_agg"] not in {"mean", "min"}:
            raise ValueError("q_agg must be 'mean' or 'min'.")
        if int(self.config["flow_steps"]) < 1:
            raise ValueError("flow_steps must be positive.")
        if self.config["optimizer_impl"] not in {"foreach", "fused"}:
            raise ValueError("optimizer_impl must be 'foreach' or 'fused'.")
        torch.manual_seed(seed)
        self.seed = int(seed)
        observation_dim = int(self.config["ob_dims"][0])
        action_dim = int(self.config["action_dim"])
        self.critic = Value(observation_dim, action_dim, self.config["value_hidden_dims"],
                            self.config["layer_norm"])
        self.target_critic = copy.deepcopy(self.critic)
        self.target_critic.requires_grad_(False)
        self.actor_bc_flow = ActorVectorField(
            observation_dim, action_dim, self.config["actor_hidden_dims"],
            self.config["actor_layer_norm"], with_time=True)
        self.actor_onestep_flow = ActorVectorField(
            observation_dim, action_dim, self.config["actor_hidden_dims"],
            self.config["actor_layer_norm"], with_time=False)
        self.to(device=device, dtype=torch.float32)
        # One optimizer matches the original summed loss; target weights are not
        # differentiable. Adam's eps is outside sqrt, as in optax.adam.
        self.optimizer = torch.optim.Adam(
            self.parameters(), lr=float(self.config["lr"]), betas=(0.9, 0.999),
            eps=1e-8, foreach=self.config["optimizer_impl"] == "foreach",
            fused=self.config["optimizer_impl"] == "fused")
        self.updates = 0

    def configure_loss_compile(self, enabled: bool = True) -> None:
        """Experimental static loss/backward compilation; preserves module keys.

        Native random operators are retained so saved CPU/MPS RNG states remain
        meaningful. A compiled run still needs its own numerical/resume checks.
        This changes neither the optimizer nor the network state_dict layout.
        """
        if not hasattr(self, "_eager_compute_losses"):
            self._eager_compute_losses = self.compute_losses
        self.compute_losses = torch.compile(
            self._eager_compute_losses, backend="inductor", fullgraph=True,
            dynamic=False, options={"fallback_random": True},
        ) if enabled else self._eager_compute_losses

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def _tensor(self, value: Any) -> Tensor:
        return torch.as_tensor(value, dtype=torch.float32, device=self.device)

    def _noise(self, randomness: Mapping[str, Any], key: str,
               shape: tuple[int, ...], uniform: bool = False) -> Tensor:
        if key in randomness:
            value = self._tensor(randomness[key])
            if tuple(value.shape) != tuple(shape):
                raise ValueError(f"randomness[{key!r}] shape {tuple(value.shape)} != {shape}")
            return value
        fn = torch.rand if uniform else torch.randn
        return fn(shape, dtype=torch.float32, device=self.device)

    @torch.no_grad()
    def sample_actions(self, observations: Any, noises: Any | None = None) -> Tensor:
        observations = self._tensor(observations)
        shape = (*observations.shape[:-1], int(self.config["action_dim"]))
        noises = torch.randn(shape, dtype=torch.float32, device=self.device) if noises is None else self._tensor(noises)
        return self.actor_onestep_flow(observations, noises).clamp(-1, 1)

    @torch.no_grad()
    def compute_flow_actions(self, observations: Any, noises: Any) -> Tensor:
        observations = self._tensor(observations)
        actions = self._tensor(noises)
        steps = int(self.config["flow_steps"])
        for i in range(steps):
            times = torch.full((*observations.shape[:-1], 1), i / steps,
                               dtype=observations.dtype, device=observations.device)
            actions = actions + self.actor_bc_flow(observations, actions, times) / steps
        return actions.clamp(-1, 1)

    def compute_losses(self, batch: Mapping[str, Any],
                       randomness: Mapping[str, Any] | None = None) -> tuple[Tensor, dict[str, Tensor]]:
        randomness = randomness or {}
        observations = self._tensor(batch["observations"])
        actions = self._tensor(batch["actions"])
        next_observations = self._tensor(batch["next_observations"])
        rewards = self._tensor(batch["rewards"]).reshape(-1)
        masks = self._tensor(batch["masks"]).reshape(-1)
        batch_size, action_dim = actions.shape
        shape = (batch_size, action_dim)

        # TD target uses the pre-update one-step actor and target critic.
        next_noises = self._noise(randomness, "next_noises", shape)
        with torch.no_grad():
            next_actions = self.sample_actions(next_observations, next_noises)
            next_qs = self.target_critic(next_observations, next_actions)
            next_q = next_qs.min(dim=0).values if self.config["q_agg"] == "min" else next_qs.mean(dim=0)
            target_q = rewards + float(self.config["discount"]) * masks * next_q
        qs = self.critic(observations, actions)
        critic_loss = (qs - target_q).square().mean()

        x0 = self._noise(randomness, "x0", shape)
        times = self._noise(randomness, "t", (batch_size, 1), uniform=True)
        xt = (1 - times) * x0 + times * actions
        velocity = actions - x0
        predicted = self.actor_bc_flow(observations, xt, times)
        bc_flow_loss = (predicted - velocity).square().mean()

        distill_noises = self._noise(randomness, "distill_noises", shape)
        # Detached old BC flow is the teacher; it is still trained by BC above.
        target_actions = self.compute_flow_actions(observations, distill_noises)
        actor_actions = self.actor_onestep_flow(observations, distill_noises)
        distill_loss = (actor_actions - target_actions).square().mean()
        # Detach weights, not the action input: Q gradients must reach the actor.
        actor_q = self.critic(observations, actor_actions.clamp(-1, 1),
                              detach_parameters=True).mean(dim=0)
        q_loss = -actor_q.mean()
        if self.config["normalize_q_loss"]:
            q_loss = q_loss * (1 / actor_q.abs().mean()).detach()
        actor_loss = bc_flow_loss + float(self.config["alpha"]) * distill_loss + q_loss
        total_loss = critic_loss + actor_loss

        metric_noises = self._noise(randomness, "metric_noises", shape)
        metric_actions = self.sample_actions(observations, metric_noises)
        metrics = {
            "loss": total_loss, "critic_loss": critic_loss,
            "q_mean": qs.mean(), "q_max": qs.max(), "q_min": qs.min(),
            "actor_loss": actor_loss, "bc_flow_loss": bc_flow_loss,
            "distill_loss": distill_loss, "q_loss": q_loss, "q": actor_q.mean(),
            "mse": (metric_actions - actions).square().mean(),
        }
        # Author-compatible labels simplify comparing archived JAX logs.
        for name in ("critic_loss", "q_mean", "q_max", "q_min"):
            metrics[f"critic/{name}"] = metrics[name]
        for name in ("actor_loss", "bc_flow_loss", "distill_loss", "q_loss", "q", "mse"):
            metrics[f"actor/{name}"] = metrics[name]
        return total_loss, metrics

    @torch.no_grad()
    def target_update(self) -> None:
        tau = float(self.config["tau"])
        # The original uses OLD critic weights, even though it calls this after
        # producing a new TrainState. Update before Adam to preserve that order.
        targets = list(self.target_critic.parameters())
        sources = list(self.critic.parameters())
        torch._foreach_mul_(targets, 1 - tau)
        torch._foreach_add_(targets, sources, alpha=tau)

    def update(self, batch: Mapping[str, Any],
               randomness: Mapping[str, Any] | None = None,
               compute_grad_stats: bool = False) -> dict[str, Tensor]:
        self.optimizer.zero_grad(set_to_none=True)
        loss, metrics = self.compute_losses(batch, randomness=randomness)
        loss.backward()
        if compute_grad_stats:
            # The author sums each parameter leaf's L2 norm (not global L2).
            # Include zero target gradients in extrema, as in the Flax tree.
            zero = loss.new_zeros(())
            grads = [p.grad for p in self.parameters() if p.grad is not None]
            metrics["grad/max"] = torch.stack([zero, *(g.max() for g in grads)]).max()
            metrics["grad/min"] = torch.stack([zero, *(g.min() for g in grads)]).min()
            metrics["grad/norm"] = torch.stack([g.norm() for g in grads]).sum()
        self.target_update()
        self.optimizer.step()
        self.updates += 1
        return {key: value.detach() for key, value in metrics.items()}

    @torch.no_grad()
    def load_flax_params(self, params: Mapping[str, Any]) -> None:
        """Import author's state-model params; Flax [in,out] → Torch [out,in]."""
        for name in ("critic", "target_critic", "actor_bc_flow", "actor_onestep_flow"):
            module = getattr(self, name)
            critic = name in {"critic", "target_critic"}
            mlp = module.value_net if critic else module.mlp
            source = params[f"modules_{name}"]["value_net" if critic else "mlp"]
            for i, layer in enumerate(mlp.layers):
                dense = source[f"Dense_{i}"]
                kernel = np.asarray(dense["kernel"])
                layer.weight.copy_(self._tensor(np.swapaxes(kernel, -1, -2).copy()))
                layer.bias.copy_(self._tensor(np.asarray(dense["bias"]).copy()))
            for i, norm in enumerate(mlp.norms):
                layer_norm = source[f"LayerNorm_{i}"]
                norm.weight.copy_(self._tensor(np.asarray(layer_norm["scale"]).copy()))
                norm.bias.copy_(self._tensor(np.asarray(layer_norm["bias"]).copy()))

    def export_flax_params(self, gradients: bool = False) -> dict[str, Any]:
        """Export NumPy trees for numerical comparison; no JAX dependency."""
        def array(parameter: Tensor, transpose: bool = False) -> np.ndarray:
            value = parameter.grad if gradients else parameter
            if value is None:
                value = torch.zeros_like(parameter)
            result = value.detach().cpu().numpy().copy()
            return np.swapaxes(result, -1, -2).copy() if transpose else result

        params = {}
        for name in ("critic", "target_critic", "actor_bc_flow", "actor_onestep_flow"):
            module = getattr(self, name)
            critic = name in {"critic", "target_critic"}
            mlp = module.value_net if critic else module.mlp
            tree = {}
            for i, layer in enumerate(mlp.layers):
                tree[f"Dense_{i}"] = {"kernel": array(layer.weight, True), "bias": array(layer.bias)}
            for i, norm in enumerate(mlp.norms):
                tree[f"LayerNorm_{i}"] = {"scale": array(norm.weight), "bias": array(norm.bias)}
            params[f"modules_{name}"] = {"value_net" if critic else "mlp": tree}
        return params
