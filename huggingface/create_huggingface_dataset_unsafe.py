import numpy as np
from datasets import Dataset
from huggingface_hub import login

login()

path_files = "/mnt/hdd/data_safe_flow_match/data/"

data = np.load(
    path_files + "imitation_trajs_vpsto_unsafe.npz",
    allow_pickle=True,
)
trajectories = data["trajectories"]
conditional_data = data["c_data1"]

dataset_dict = {
    "trajectories": trajectories,
    "conditional_data": conditional_data,
}

dataset = Dataset.from_dict(dataset_dict)


dataset.push_to_hub("ThiesOelerich/SafeFlowMPC_pretrain")
