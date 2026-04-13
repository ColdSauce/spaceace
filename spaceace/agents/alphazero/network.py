"""AlphaZero dual-head network: policy + value from observation."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AlphaZeroNet(nn.Module):
    def __init__(self, obs_dim: int = 27, num_actions: int = 6, hidden_dim: int = 256):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_actions),
        )
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        shared = self.shared(x)
        policy_logits = self.policy_head(shared)
        value = self.value_head(shared).squeeze(-1)
        return policy_logits, value


def export_to_onnx(model: AlphaZeroNet, path: str, obs_dim: int = 27):
    """Export trained model to ONNX for Rust inference."""
    model.eval()
    dummy = torch.randn(1, obs_dim)
    torch.onnx.export(
        model, dummy, path,
        input_names=["observation"],
        output_names=["policy", "value"],
        dynamic_axes={
            "observation": {0: "batch"},
            "policy": {0: "batch"},
            "value": {0: "batch"},
        },
        opset_version=17,
    )
