# %%
import torch
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import json
import numpy as np
import itertools
from tqdm.notebook import tqdm
from dictionary_learning.utils import MCC
from datetime import datetime
import shutil
from demo_config import TARGET_L0s, SPARSITY_PENALTIES
# %%
def fast_decoder_weights_loader(ckpt_path: str):
    """
    Load the weights of the fast decoder from the checkpoint.
    """
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if "decoder.weight" in ckpt.keys():
        A = ckpt["decoder.weight"].T.numpy()
        assert A.shape[0] > A.shape[1], "The first dimension should be larger than the second dimension"
    elif "W_dec" in ckpt.keys():
        A = ckpt["W_dec"].numpy()
        assert A.shape[0] > A.shape[1], "The first dimension should be larger than the second dimension"
    else:
        raise ValueError("The checkpoint does not contain the decoder weights")
    return A

def calculate_mutual_mcc(ae_path_list):
    """
    Calculate the mutual MCC of the AE weights.
    
    Returns:
        dict: {
            'mean': mean_mcc, 
            'std': std_mcc, 
            'values': mcc_list
        }
    """
    A_list = []
    for ae_path in ae_path_list:
        A = fast_decoder_weights_loader(ae_path)
        if A is not None:
            A_list.append(A)
        else:
            print(f"Failed to load weights from {ae_path}")
    if len(A_list) < 2:
        raise ValueError("Not enough valid AE weights to calculate MCC.")
    
    mcc_list = []
    for A_i, A_j in itertools.combinations(A_list, 2):
        mcc = float(MCC(A_i, A_j))
        mcc_list.append(mcc)
    
    return {
        'mean': float(np.mean(mcc_list)),
        'std': float(np.std(mcc_list)),
        'values': mcc_list
    }

def process_config(config_item):
    """
    Process a single configuration (for parallel execution).
    """
    config_name, ae_paths = config_item
    try:
        mcc_result = calculate_mutual_mcc(ae_paths)
        return config_name, mcc_result
    except Exception as e:
        print(f"Error processing {config_name}: {e}")
        return config_name, None
# %%
# Output file path
output_base_path = "output/pythia-160m-ds16k"
output_file_path = Path(f"{output_base_path}/all_mcc_results.json")

# Load existing results if available
existing_results = {}
if output_file_path.exists():
    with open(output_file_path, "r") as f:
        existing_results = json.load(f)
    print(f"Loaded {len(existing_results)} existing results from {output_file_path}")

# first generate ae weight paths and group with random seeds
linear_steps = [int(244140 * step) for step in torch.linspace(0, 1, 11)]
log_checkpoints = [int(244140 * step) for step in torch.logspace(-3, 0, 7)[:-1]]
save_steps = sorted(linear_steps + log_checkpoints)

k_model_names = ["top_k", "batch_top_k", "jump_relu", "matryoshka_batch_top_k"]
l1_model_names = ["standard", "standard_new", "p_anneal", "gated"]

# lrs = ["3e-4"] #["1e-3", "3e-4"]
random_seeds = [42,43,44]
target_l0s = TARGET_L0s
config_name_to_ae_paths = {}

config_name_to_ae_paths = {}
for model_name in k_model_names:
    # for lr in lrs:
    for target_l0 in target_l0s:
        for step in save_steps:
            config_name = f"{model_name}-k{target_l0}-step{step}"
            config_name_to_ae_paths[config_name] = []
for model_name in l1_model_names:
    # for lr in lrs:
    for sparsity_penalty in getattr(SPARSITY_PENALTIES, model_name,[]):
        for step in save_steps:
            config_name = f"{model_name}-l{sparsity_penalty}-step{step}"
            config_name_to_ae_paths[config_name] = []

for model_name in k_model_names:
    # for lr in lrs:
    output_path = Path(output_base_path) / f"{model_name}" / "resid_post_layer_8"
    if not output_path.exists():
        print(f"Output path {output_path} does not exist")
        continue
    for trainer_path in output_path.iterdir():
        if not trainer_path.is_dir():
            continue
        with open(trainer_path / "config.json", "r") as f:
            config = json.load(f)
        if "target_l0" in config["trainer"]:
            target_l0 = config["trainer"]["target_l0"]
        elif "k" in config["trainer"]:
            target_l0 = config["trainer"]["k"]
        else:
            raise ValueError("target_l0 or k not found in config")
        for step in save_steps[:-1]:  # final step is saved in as ae.pt without step
            config_name = f"{model_name}-k{target_l0}-step{step}"
            ae_path = trainer_path / "checkpoints" / f"ae_{step}.pt"
            if ae_path.exists():
                config_name_to_ae_paths[config_name].append(str(ae_path))
        ae_path = trainer_path / "ae.pt"
        if ae_path.exists():
            config_name = f"{model_name}-k{target_l0}-step{save_steps[-1]}"
            config_name_to_ae_paths[config_name].append(str(ae_path))
for model_name in l1_model_names:
    # for lr in lrs:
    output_path = Path(output_base_path) / f"{model_name}" / "resid_post_layer_8"
    if not output_path.exists():
        print(f"Output path {output_path} does not exist")
        continue
    for trainer_path in output_path.iterdir():
        if not trainer_path.is_dir():
            continue
        with open(trainer_path / "config.json", "r") as f:
            config = json.load(f)
        if "l1_penalty" in config["trainer"]:
            sparsity_penalty = config["trainer"]["l1_penalty"]
        elif "sparsity_penalty" in config["trainer"]:
            sparsity_penalty = config["trainer"]["sparsity_penalty"]
        else:
            raise ValueError("l1_penalty or sparsity_penalty not found in config")
        for step in save_steps[:-1]:  # final step is saved in as ae.pt without step
            config_name = f"{model_name}-l{sparsity_penalty}-step{step}"
            ae_path = trainer_path / "checkpoints" / f"ae_{step}.pt"
            if ae_path.exists():
                config_name_to_ae_paths[config_name].append(str(ae_path))
        ae_path = trainer_path / "ae.pt"
        if ae_path.exists():
            config_name = f"{model_name}-l{sparsity_penalty}-step{save_steps[-1]}"
            config_name_to_ae_paths[config_name].append(str(ae_path))

# only keep the configs that have 3 ae weights
for k, v in config_name_to_ae_paths.items():
    if len(v) < 3:
        print(f"Config {k} has {len(v)} ae weights")
config_name_to_ae_paths = {k: v for k, v in config_name_to_ae_paths.items() if len(v) == 3}

# Filter out configs that are already processed
configs_to_process = {
    k: v for k, v in config_name_to_ae_paths.items() 
    if k not in existing_results
}

print(f"Found {len(config_name_to_ae_paths)} total configurations")
print(f"Already processed {len(existing_results)} configurations")
print(f"Need to process {len(configs_to_process)} new configurations")

if not configs_to_process:
    print("No new configurations to process. Exiting.")
    exit(0)
# %%
# Process all new configs in parallel
results = dict(existing_results)  # Start with existing results
num_workers = min(64, len(configs_to_process))

print(f"Processing {len(configs_to_process)} new configurations using {num_workers} workers")

with ProcessPoolExecutor(max_workers=num_workers) as executor:
    futures = {
        executor.submit(process_config, item): item[0] 
        for item in configs_to_process.items()
    }
    
    for future in tqdm(as_completed(futures), total=len(futures), desc="Processing configs"):
        config_name = futures[future]
        try:
            result = future.result()
            if result[1] is not None:  # If not None, store the result
                config_name, mcc_result = result
                results[config_name] = mcc_result
                print(f"{config_name}: {mcc_result['mean']:.4f} ± {mcc_result['std']:.4f}")
        except Exception as e:
            print(f"Exception processing {config_name}: {e}")

# Create a backup of the previous results file if it exists
if output_file_path.exists():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_file = output_file_path.with_name(f"all_mcc_results_{timestamp}.json")
    shutil.copy2(output_file_path, backup_file)
    print(f"Created backup of previous results at {backup_file}")

# Ensure the output directory exists
output_file_path.parent.mkdir(parents=True, exist_ok=True)

# Save results to file
with open(output_file_path, "w") as f:
    json.dump(results, f, indent=4)

print(f"Processed {len(results) - len(existing_results)} new configurations")
print(f"Total configurations in results: {len(results)}")
# %%
