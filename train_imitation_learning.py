import time
import warnings

import matplotlib.pyplot as plt
import numpy as np
import torch
from flow_matching.path import AffineProbPath
from flow_matching.path.scheduler import CondOTScheduler

from safe_flow_mpc.Models import EMA, MLP, TemporalUnet
from safe_flow_mpc.RobotModel import RobotModel

warnings.filterwarnings("ignore", category=UserWarning, module="torch")

if torch.cuda.is_available():
    device = "cuda:0"
    print("Using gpu")
else:
    device = "cpu"
    print("Using cpu")

torch.manual_seed(42)

robot_model = RobotModel()


# training arguments
lr = 0.0001
# batch_size = 1024
iterations = 50001
batch_size = 1024
# iterations = 25001
print_every = 500
use_vpsto = True
use_handover = True if not use_vpsto else False
model_name = (
    f"model_unsafe{'_vpsto' if use_vpsto else ''}{'_handover' if use_handover else ''}"
)
if use_handover:
    path_files = "data/dynamic_handover_dataset/data/"
else:
    path_files = "data/"

data = np.load(
    path_files
    + f"imitation_trajs{'_vpsto_unsafe' if use_vpsto else ''}{'_handover_unsafe' if use_handover else ''}.npz",
    allow_pickle=True,
)
trajectories = torch.Tensor(data["trajectories"]).to(device)
conditional_data = torch.Tensor(data["c_data1"]).to(device)


def get_train_data(batch_size: int = 200, device: str = "cpu"):
    idx = torch.randint(0, len(trajectories), (batch_size,)).to(device)
    p = trajectories[idx].float()
    c_data = conditional_data[idx].float()

    return p, c_data


n_horizon = 16
if use_handover:
    n_out = 8
    cond_dim = 225 - n_horizon * 7 - 1
else:
    n_out = 7
    cond_dim = 228 - n_horizon * 7 - 1
velocity_field = TemporalUnet(
    horizon=n_horizon,
    transition_dim=n_out,
    cond_dim=cond_dim,
    dim=32,
    dim_mults=(1, 2, 4, 8),
).to(device)
ema = EMA(velocity_field)

# instantiate an affine path object
path = AffineProbPath(scheduler=CondOTScheduler())

# init optimizer
optim = torch.optim.AdamW(
    velocity_field.parameters(),
    lr=lr,
    weight_decay=1e-7,
)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optim, T_max=(iterations) * 2, eta_min=1e-7
)

# train
train = True
dt = 0.1
max_delta_q = 0.1
if train:
    loss_log = []
    iter_log = []
    start_time = time.time()

    alpha = torch.tensor(1.5).to(device)
    beta = torch.tensor(1.0).to(device)
    beta_dist = torch.distributions.Beta(alpha, beta)

    for i in range(iterations):
        # Sample trajectories
        trajs, c_data = get_train_data(batch_size=batch_size, device=device)

        x_1 = trajs
        x_0 = torch.randn_like(x_1).to(device)
        optim.zero_grad()

        # Sample a random value from the Beta distribution
        t = beta_dist.sample((batch_size,))
        t = 1 - t
        # t = torch.rand((batch_size,)).to(device)

        # sample probability path
        path_sample = path.sample(t=t, x_0=x_0, x_1=x_1)

        x_t = path_sample.x_t.reshape((-1, n_horizon, n_out))
        dxc = velocity_field(x_t, path_sample.t, c_data)
        dxc = dxc.reshape((x_t.shape[0], -1))

        loss = torch.pow(dxc - path_sample.dx_t, 2).mean()

        loss_log.append(loss.item())
        iter_log.append(i)

        # optimizer step
        loss.backward()  # backward
        optim.step()  # update
        scheduler.step()
        ema.update(velocity_field)

        # Log loss
        if (i + 1) % print_every == 0:
            elapsed = time.time() - start_time
            print(
                f"| iter {i + 1:6d} | {int((elapsed * 1000) / print_every)} ms/step | loss {loss.item():8.4f} | lr {scheduler.get_lr()[0]:8.6f} "
            )
            start_time = time.time()
    # Save
    torch.save(
        {"model": velocity_field.state_dict(), "ema_model": ema.state_dict()},
        f"checkpoints/{model_name}.pth",
    )

    plt.figure()
    plt.title("Loss curve")
    plt.plot(iter_log, loss_log)
    plt.show()
