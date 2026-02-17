import os
from glob import glob

import numpy as np
from datasets import (
    Array2D,
    Array4D,
    Dataset,
    Features,
    concatenate_datasets,
    load_from_disk,
)
from huggingface_hub import login
from tqdm import tqdm

features = Features(
    {
        "trajectories": Array2D(dtype="float32", shape=(112, 1)),
        "c_data": Array2D(dtype="float32", shape=(None, 1)),
        "samples": Array4D(dtype="float32", shape=(None, None, None, None)),
        "dsamples": Array4D(dtype="float32", shape=(None, None, None, None)),
        "t_samples": Array2D(dtype="float32", shape=(None, None, None, None)),
    }
)

login()

path_files = "/mnt/hdd/data_safe_flow_match/data/"
shard_dir = "/mnt/hdd/dataset_shards"
os.makedirs(shard_dir, exist_ok=True)


def data_generator():
    npz_files = glob(f"{path_files}imitation_trajs_vpsto_term_*.npz")
    for file in tqdm(npz_files, desc="Processing files"):
        data = np.load(file, mmap_mode="r")
        yield {
            "trajectories": data["trajectories"],
            "c_data": data["c_data1"],
            "samples": data["samples"],
            "dsamples": data["dsamples"],
            "t_samples": data["t_samples"],
        }


def convert_shard_to_dataset(shard_data):
    return Dataset.from_dict(
        {
            "trajectories": np.stack(shard_data["trajectories"]),
            "c_data": np.stack(shard_data["c_data"]),
            "samples": np.stack(shard_data["samples"]),
            "dsamples": np.stack(shard_data["dsamples"]),
            "t_samples": np.stack(shard_data["t_samples"]),
        }
    )


shard_idx = 0
for shard_data in data_generator():
    # ds = Dataset.from_dict(shard_data, features=features)
    ds = convert_shard_to_dataset(shard_data)
    ds.save_to_disk(os.path.join(shard_dir, f"shard_{shard_idx}"))
    print(
        f"Saved shard {shard_idx} with {len(shard_data['trajectories'])} trajectories"
    )
    shard_idx += 1

shards = []
for i in range(shard_idx):
    shards.append(load_from_disk(os.path.join(shard_dir, f"shard_{i}")))
dataset = concatenate_datasets(shards)

dataset.push_to_hub("ThiesOelerich/SafeFlowMPC")
