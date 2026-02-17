from dataclasses import dataclass
from typing import Tuple


@dataclass
class PlannerConfig:
    """Configuration class for the trajectory planner."""

    # Model configuration
    use_safety_filter: bool = True
    use_safe_model: bool = True
    use_safe_dist: bool = True
    use_term: bool = True
    use_sets: bool = True
    use_cart_goal: bool = True
    use_rti: bool = True
    use_guidance: bool = True
    real_time: bool = False
    experiment: bool = False

    # Network parameters
    n_horizon: int = 16
    n_actions: int = 1
    n_out: int = 7
    flow_steps: int = 10
    fm_dim: int = 16
    fm_dim_mults: Tuple[int] = (1, 2, 4)
    cond_dim = 225 - n_horizon * 7 - 1
    compile_fm: bool = False

    # Guidance parameters
    w_guidance: float = 0.1

    # File paths
    data_path: str = "data/"
    data_name: str = "traj_example_"
    model_name: str = "model"
    model_path: str = "checkpoints/"

    # Simulation parameters
    smooth: bool = True
    build: bool = False
    dt_sim: float = 0.1
    dt_real: float = 0.1
    limit_time: bool = True
    sleep: bool = True

    def __post_init__(self):
        """Post-initialization adjustments."""
        if not self.use_safety_filter:
            self.use_safe_dist = False

        self.cond_dim = 228 - self.n_horizon * 7 - 1
        self.n_out = 7
        tail = "vpsto"
        if self.use_safe_model:
            self.model_name = f"model{'_q' if not self.use_cart_goal else ''}{'_term' if self.use_term else ''}_{tail}"
        else:
            self.model_name = f"model_unsafe_{tail}"
        self.model_path = self.model_path
