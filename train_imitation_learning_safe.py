import time
import warnings
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
import torch
from datasets import load_dataset
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
    print("Using cpu.")

torch.manual_seed(42)

robot_model = RobotModel()

# training arguments
finetuning = True
lr = 0.001
if finetuning:
    lr = 1e-6
batch_size = 2
samples_per_batch = 10
iterations = 3000
save_every = 200
print_every = 50
hidden_dim = 2048
use_q_goal = False
use_weights = False
use_term = True
use_vpsto = True
model_name = f"model{'_q' if use_q_goal else ''}{'_term' if use_term else ''}vpsto"
print(model_name)
train = True
n_horizon = 16

dataset = load_dataset("ThiesOelerich/SafeFlowMPC", split="train")
dataset.set_format("numpy")

weights = 1 / np.array(
    [
        0.0280,
        0.0281,
        0.3327,
        0.6407,
        0.9240,
        1.3103,
        1.7209,
        2.1585,
        2.6091,
        3.2197,
        3.8210,
        4.5847,
        5.4070,
        6.5930,
        8.0410,
        10.0410,
    ]
)


def get_train_data(batch_size: int = 2, device: str = "cpu", idx=None):
    # Randomly sample indices
    indices = np.random.randint(0, len(dataset), size=batch_size)

    batch = dataset.select(indices)

    # Convert to torch tensors
    p = torch.tensor(batch["trajectories"], device=device).float()
    c_data = torch.tensor(batch["c_data"], device=device).float()
    samples = torch.tensor(batch["samples"], device=device).float()
    dsamples = torch.tensor(batch["dsamples"], device=device).float()
    t_samples = torch.tensor(batch["t_samples"], device=device).float()

    return p, c_data, samples, dsamples, t_samples


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

if finetuning:
    print("Finetuning")
    model_name_ft = f"model_unsafe_vpsto"
    print(model_name_ft)
    checkpoint = torch.load(f"checkpoints/{model_name_ft}.pth")
    velocity_field.load_state_dict(checkpoint["model"])
    ema.load_state_dict(checkpoint["ema_model"])

# instantiate an affine path object
path = AffineProbPath(scheduler=CondOTScheduler())

# init optimizer
optim = torch.optim.AdamW(
    velocity_field.parameters(),
    lr=lr,
    weight_decay=1e-7,
)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optim,
    T_max=(iterations * samples_per_batch) * 2,
    eta_min=5e-5 if not finetuning else 1e-7,
)

weights_full = np.repeat(weights, 7)
weights_tensor = torch.Tensor(weights_full).to(device)
if train:
    loss_log = []
    lr_log = []
    iter_log = []
    start_time = time.time()

    alpha = torch.tensor(1.5).to(device)
    beta = torch.tensor(1.0).to(device)
    beta_dist = torch.distributions.Beta(alpha, beta)

    for i in range(iterations):
        # Sample trajectories
        trajs, c_data, samples, dsamples, t_samples = get_train_data(
            batch_size=batch_size, device=device
        )
        decider = torch.rand(1)
        if decider <= 0.5:
            for k in range(samples_per_batch):
                optim.zero_grad()
                # Sample a random value from the Beta distribution
                t = beta_dist.sample((trajs.shape[0],))
                t = 1 - t

                idx_t = (t * 50).to(torch.int64)
                idx0 = torch.arange(samples.size(0))
                idx_dist = torch.randint(0, t_samples.shape[1], (trajs.shape[0],))
                x_t = samples[idx0, idx_dist, idx_t]
                dx_t = dsamples[idx0, idx_dist, idx_t].clone()
                t_sampled = t_samples[idx0, idx_dist, idx_t]

                x_t = x_t.reshape(-1, n_horizon, 7)
                dxc = velocity_field(x_t, t_sampled, c_data)
                dxc = dxc.reshape((x_t.shape[0], -1))

                if use_weights:
                    loss = torch.pow((dxc - weights_tensor * dx_t), 2).mean()
                else:
                    loss = torch.pow((dxc - dx_t), 2).mean()

                loss_log.append(loss.item())
                iter_log.append(i * samples_per_batch + k)
                lr_log.append(scheduler.get_lr())
                loss.backward()
                optim.step()
                scheduler.step()
                ema.update(velocity_field)
        else:
            for k in range(samples_per_batch):
                optim.zero_grad()
                t = beta_dist.sample((trajs.shape[0],))
                t = 1 - t
                t_sampled = t
                x_1 = trajs
                x_0 = torch.randn_like(x_1)
                x_t = x_0 + t[:, None] * (x_1 - x_0)
                dx_t = x_1 - x_0

                x_t = x_t.reshape(-1, n_horizon, n_out)
                dxc = velocity_field(x_t, t_sampled, c_data)
                dxc = dxc.reshape((x_t.shape[0], -1))

                if use_weights:
                    loss = torch.pow((dxc - weights_tensor * dx_t), 2).mean()
                else:
                    loss = torch.pow((dxc - dx_t), 2).mean()

                loss_log.append(loss.item())
                iter_log.append(i * samples_per_batch + k)
                lr_log.append(scheduler.get_lr())
                loss.backward()
                optim.step()
                scheduler.step()
                ema.update(velocity_field)

        # Log loss
        if i % print_every == 0:
            elapsed = time.time() - start_time
            print(
                f"| iter {i + 1:6d} / {iterations} | {int((elapsed * 1000) / print_every)} ms/step | loss {loss.item():8.4f} | lr {lr_log[-1][0]:8.6f} "
            )
            start_time = time.time()
            loss = 0

    torch.save(
        {"model": velocity_field.state_dict(), "ema_model": ema.state_dict()},
        f"checkpoints/{model_name}.pth",
    )
    plt.figure()
    plt.subplot(211)
    plt.title("Loss")
    plt.plot(iter_log, loss_log)

    plt.subplot(212)
    plt.title("Learning Rate")
    plt.plot(iter_log, lr_log)
    plt.show()
