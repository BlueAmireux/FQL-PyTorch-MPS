"""Image/proprioception action distillation, separate from state-only FQL."""

from collections.abc import Mapping

import torch
from torch import Tensor, nn

from fql_torch.networks import ActorVectorField


class VisualStudent(nn.Module):
    """Accept raw RGB [0,255], proprioception, and the teacher's Gaussian noise.

    The nine visual features are unconstrained latent values, not pose estimates.
    No true cube state is accepted. All encoder and actor parameters are trainable.
    ``teacher_student_state_dict`` is actor_onestep_flow.state_dict(), not the
    full FQL model checkpoint. Proprioception retains the official state scaling.
    """

    def __init__(self, teacher_student_state_dict: Mapping[str, Tensor] | None = None):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1), nn.GELU(approximate="tanh"),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.GELU(approximate="tanh"),
            nn.Conv2d(64, 64, 3, stride=2, padding=1), nn.GELU(approximate="tanh"),
            nn.AdaptiveAvgPool2d((4, 4)), nn.Flatten(),
            nn.Linear(64 * 4 * 4, 128), nn.GELU(approximate="tanh"),
            nn.Linear(128, 9),
        )
        # Preserve the teacher's proprio/noise columns and all downstream layers.
        self.actor = ActorVectorField(28, 5, (512, 512, 512, 512),
                                      layer_norm=False, with_time=False)
        if teacher_student_state_dict is not None:
            self.actor.load_state_dict(teacher_student_state_dict, strict=True)
        self.to(dtype=torch.float32)

    def raw_actions(self, images: Tensor, proprio: Tensor, noises: Tensor) -> Tensor:
        """Unclipped output for MSE training, avoiding clamp's zero gradients."""
        if images.ndim != 4 or tuple(images.shape[1:]) != (3, 64, 64):
            raise ValueError("images must have shape [B,3,64,64] (RGB in [0,255]).")
        batch_size = images.shape[0]
        if tuple(proprio.shape) != (batch_size, 19):
            raise ValueError("proprio must have shape [B,19]; cube state is forbidden.")
        if tuple(noises.shape) != (batch_size, 5):
            raise ValueError("noises must have shape [B,5].")
        visual = self.encoder(images.to(dtype=torch.float32) / 255.0)
        observations = torch.cat((proprio.to(dtype=torch.float32), visual), dim=-1)
        return self.actor(observations, noises.to(dtype=torch.float32)).float()

    def forward(self, images: Tensor, proprio: Tensor, noises: Tensor) -> Tensor:
        return self.raw_actions(images, proprio, noises).clamp(-1, 1)
