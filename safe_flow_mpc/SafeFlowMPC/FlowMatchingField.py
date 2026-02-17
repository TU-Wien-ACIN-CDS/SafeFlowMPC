import os

import torch
from huggingface_hub import hf_hub_download

from safe_flow_mpc.Models import EMA, TemporalUnet
from safe_flow_mpc.SafeFlowMPC import PlannerConfig


class FlowMatchingField:
    """Handles neural network-based velocity field computation."""

    def __init__(self, config: PlannerConfig, device: str):
        self.config = config
        self.device = device
        self.velocity_field = None
        self.ema = None

        self._initialize_model()
        self._load_weights()

    def _initialize_model(self) -> None:
        """Initialize the neural network model."""
        self.velocity_field = TemporalUnet(
            horizon=self.config.n_horizon,
            transition_dim=self.config.n_out,
            cond_dim=self.config.cond_dim,
            dim=self.config.fm_dim,
            dim_mults=self.config.fm_dim_mults,
        ).to(self.device)

        self.ema = EMA(self.velocity_field, decay=0.999)

    def _load_weights(self) -> None:
        """Load pre-trained model weights."""
        try:
            # Try loading locally
            print(
                f"Attempting to load {self.config.model_path}{self.config.model_name} locally..."
            )
            checkpoint = torch.load(
                f"{self.config.model_path}{self.config.model_name}.pth",
                weights_only=True,
            )
        except FileNotFoundError:
            # If not found, download from Hugging Face Hub
            print("Local weights not found. Downloading from Hugging Face Hub...")
            model_path = hf_hub_download(
                repo_id="ThiesOelerich/SafeFlowMPC",
                filename=f"{self.config.model_name}.pth",
                repo_type="model",
            )
            checkpoint = torch.load(model_path, weights_only=True)

        self.velocity_field.load_state_dict(checkpoint["model"])
        self.ema.load_state_dict(checkpoint["ema_model"])
        self.ema.ema_model.to(self.device)
        self.ema.ema_model.eval()
        if self.config.compile_fm:
            if self.config.experiment:
                self.ema.ema_model = torch.compile(self.ema.ema_model)
            else:
                self.ema.ema_model = torch.compile(self.ema.ema_model, backend="eager")
        # Create dummy input tensors
        dummy_x = torch.randn(
            1, self.config.n_horizon, self.config.n_out, device=self.device
        )
        dummy_t = torch.randn(1, device=self.device)
        dummy_condition = torch.randn(1, self.config.cond_dim, device=self.device)
        # Run the model once
        with torch.inference_mode():
            _ = self.ema.ema_model(dummy_x, dummy_t, dummy_condition)
        print("Done")

    def compute_velocity(
        self,
        x_current: torch.Tensor,
        t: torch.Tensor,
        condition: torch.Tensor,
        dt: float,
    ) -> torch.Tensor:
        """Compute velocity field at given state and time."""
        x_current_unet = x_current.reshape(-1, self.config.n_horizon, self.config.n_out)

        with torch.inference_mode():
            dx_flow = self.ema.ema_model(x_current_unet, t, condition[None, :]) * dt

        dx_flow = dx_flow.reshape((-1, self.config.n_horizon * self.config.n_out))

        return dx_flow
