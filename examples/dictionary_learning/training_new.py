"""
Training dictionaries
"""

import json
import os
import time
import glob
from typing import Optional, List, Tuple, Dict
from contextlib import nullcontext

import torch as t
from tqdm import tqdm

import wandb
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_norm_factor(data, steps: int) -> float:
    """Per Section 3.1, find a fixed scalar factor so activation vectors have unit mean squared norm.
    This is very helpful for hyperparameter transfer between different layers and models.
    Use more steps for more accurate results.
    https://arxiv.org/pdf/2408.05147
    
    If experiencing troubles with hyperparameter transfer between models, it may be worth instead normalizing to the square root of d_model.
    https://transformer-circuits.pub/2024/april-update/index.html#training-saes"""
    total_mean_squared_norm = 0
    count = 0

    for step, act_BD in enumerate(tqdm(data, total=steps, desc="Calculating norm factor")):
        if step >= steps:
            break

        count += 1
        mean_squared_norm = t.mean(t.sum(act_BD ** 2, dim=1))
        total_mean_squared_norm += mean_squared_norm

    average_mean_squared_norm = total_mean_squared_norm / count
    norm_factor = t.sqrt(average_mean_squared_norm).item()

    print(f"Average mean squared norm: {average_mean_squared_norm}")
    print(f"Norm factor: {norm_factor}")
    
    return norm_factor


# def find_latest_checkpoint(save_dir: str, num_trainers: int) -> Tuple[int, bool]:
#     """
#     Find the latest checkpoint in the save directory.
    
#     Args:
#         save_dir: Directory where checkpoints are saved
#         num_trainers: Number of trainers
        
#     Returns:
#         Tuple of (latest_step, is_complete)
#         latest_step: Step number of the latest checkpoint (-1 if none found)
#         is_complete: Whether all trainers have a checkpoint at latest_step
#     """
#     latest_step = -1
#     all_steps = []
    
#     # Check if there's a completed_steps.json file
#     completed_steps_path = os.path.join(save_dir, "completed_steps.json")
#     if os.path.exists(completed_steps_path):
#         try:
#             with open(completed_steps_path, "r") as f:
#                 completed_steps = json.load(f)
#                 if completed_steps and "latest_step" in completed_steps:
#                     return completed_steps["latest_step"], True
#         except:
#             pass
    
#     # If no completed steps file or it's invalid, scan directories
#     for i in range(num_trainers):
#         trainer_dir = os.path.join(save_dir, f"trainer_{i}", "checkpoints")
#         if not os.path.exists(trainer_dir):
#             continue
            
#         checkpoints = glob.glob(os.path.join(trainer_dir, "ae_*.pt"))
#         steps = [int(os.path.basename(cp).replace("ae_", "").replace(".pt", "")) 
#                 for cp in checkpoints]
#         all_steps.extend(steps)
    
#     if all_steps:
#         # Find steps that have checkpoints for all trainers
#         step_counts = {}
#         for step in all_steps:
#             step_counts[step] = step_counts.get(step, 0) + 1
        
#         complete_steps = [step for step, count in step_counts.items() 
#                          if count == num_trainers]
        
#         if complete_steps:
#             latest_step = max(complete_steps)
#             return latest_step, True
#         else:
#             # If no complete steps, just return the maximum step found
#             latest_step = max(all_steps)
#             return latest_step, False
    
#     return latest_step, False


# def cleanup_auto_checkpoints(save_dirs: List[str], current_step: int, 
#                            auto_checkpoint_freq: int, regular_checkpoint_steps: set):
#     """
#     Remove previous auto-checkpoints but preserve regular checkpoints.
    
#     Args:
#         save_dirs: List of directories to clean up
#         current_step: Current training step
#         auto_checkpoint_freq: Frequency of auto-checkpoints
#         regular_checkpoint_steps: Set of steps where regular checkpoints were saved
#     """
#     for save_dir in save_dirs:
#         if save_dir is None:
#             continue
        
#         checkpoint_dir = os.path.join(save_dir, "checkpoints")
#         if not os.path.exists(checkpoint_dir):
#             continue
            
#         checkpoints = glob.glob(os.path.join(checkpoint_dir, "ae_*.pt"))
#         for cp in checkpoints:
#             cp_step = int(os.path.basename(cp).replace("ae_", "").replace(".pt", ""))
#             # Only remove previous auto-checkpoint steps that aren't regular checkpoints
#             if (cp_step < current_step and 
#                 cp_step % auto_checkpoint_freq == 0 and 
#                 cp_step not in regular_checkpoint_steps):
#                 try:
#                     os.remove(cp)
#                     # Also remove corresponding optimizer checkpoint if it exists
#                     opt_path = os.path.join(checkpoint_dir, f"optimizer_{cp_step}.pt")
#                     if os.path.exists(opt_path):
#                         os.remove(opt_path)
#                 except Exception as e:
#                     logger.warning(f"Warning: Failed to remove previous auto-checkpoint {cp}: {e}")


def save_checkpoint(
    save_dirs: List[str], 
    trainers: List, 
    step: int, 
    norm_factor: Optional[float] = None,
    save_optimizer: bool = False,
    mark_as_complete: bool = True
):
    """
    Save checkpoints for all trainers.
    
    Args:
        save_dirs: List of directories to save checkpoints to
        trainers: List of trainers
        step: Current step
        norm_factor: Normalization factor (if using normalization)
        save_optimizer: Whether to save optimizer state
        mark_as_complete: Whether to mark this step as complete in completed_steps.json
    """
    for i, (save_dir, trainer) in enumerate(zip(save_dirs, trainers)):
        if save_dir is None:
            continue
            
        checkpoint_dir = os.path.join(save_dir, "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        if norm_factor is not None:
            # Temporarily scale up biases for checkpoint saving
            trainer.ae.scale_biases(norm_factor)
        
        # Save model checkpoint
        checkpoint = {k: v.cpu() for k, v in trainer.ae.state_dict().items()}
        checkpoint_path = os.path.join(checkpoint_dir, f"ae_{step}.pt")
        t.save(checkpoint, checkpoint_path)
        
        # Save optimizer state if requested
        if save_optimizer and hasattr(trainer, 'optimizer'):
            optimizer_state = {k: v if not isinstance(v, t.Tensor) else v.cpu() 
                             for k, v in trainer.optimizer.state_dict().items()}
            optimizer_path = os.path.join(checkpoint_dir, f"optimizer_{step}.pt")
            t.save(optimizer_state, optimizer_path)
        
        if norm_factor is not None:
            trainer.ae.scale_biases(1 / norm_factor)
    
    # Mark this step as complete
    if mark_as_complete and save_dirs[0] is not None:
        root_dir = os.path.dirname(save_dirs[0])
        with open(os.path.join(root_dir, "completed_steps.json"), "w") as f:
            json.dump({"latest_step": step, "timestamp": time.time()}, f)


def trainSAE(
    data,
    trainer_configs: list[dict],
    steps: int,
    use_wandb: bool = False,
    wandb_entity: str = "",
    wandb_project: str = "",
    save_steps: Optional[list[int]] = None,
    save_dir: Optional[str] = None,
    log_steps: Optional[int] = None,
    activations_split_by_head: bool = False,
    transcoder: bool = False,
    run_cfg: dict = {},
    normalize_activations: bool = False,
    verbose: bool = False,
    device: str = "cuda",
    autocast_dtype: t.dtype = t.float32,
):
    device_type = "cuda" if "cuda" in device else "cpu"
    autocast_context = nullcontext() if device_type == "cpu" else t.autocast(device_type=device_type, dtype=autocast_dtype)

    # Initialize trainers
    trainers = []
    for i, config in enumerate(trainer_configs):
        if "wandb_name" in config:
            config["wandb_name"] = f"{config['wandb_name']}_trainer_{i}"
        trainer_class = config["trainer"]
        del config["trainer"]
        trainers.append(trainer_class(**config))
    
    # Make save dirs, export config
    if save_dir is not None:
        save_dirs = [
            os.path.join(save_dir, f"trainer_{i}") for i in range(len(trainer_configs))
        ]
        for trainer, dir in zip(trainers, save_dirs):
            os.makedirs(dir, exist_ok=True)
            # Save config
            config = {"trainer": trainer.config}
            try:
                config["buffer"] = data.config
            except:
                pass
            with open(os.path.join(dir, "config.json"), "w") as f:
                json.dump(config, f, indent=4)
    else:
        save_dirs = [None for _ in trainer_configs]
    
    # Initialize a single WandB session for all trainers
    wandb_run = None
    if use_wandb:
        # Combine configs from all trainers into a single config
        combined_config = run_cfg.copy()
        for i, trainer in enumerate(trainers):
            for k, v in trainer.config.items():
                # Make sure wandb config doesn't contain any CUDA tensors
                value = v.cpu().item() if isinstance(v, t.Tensor) else v
                combined_config[f"trainer_{i}_{k}"] = value
        
        run_name = combined_config.get("wandb_name", None)
        wandb_run = wandb.init(
            entity=wandb_entity, 
            project=wandb_project,
            config=combined_config, 
            name=run_name
        )
            
        # Save wandb run ID for future resumption
        if save_dir and wandb_run:
            with open(os.path.join(save_dir, "wandb_run_id.txt"), "w") as f:
                f.write(wandb_run.id)
    
    # Compute normalization factor if needed
    norm_factor = None
    if normalize_activations:
        norm_factor_path = None
        if save_dir:
            norm_factor_path = os.path.join(save_dir, "norm_factor.json")
            
        if norm_factor_path and os.path.exists(norm_factor_path):
            with open(norm_factor_path, "r") as f:
                norm_factor = json.load(f)["norm_factor"]
                logger.info(f"Loaded norm factor: {norm_factor}")
        else:
            norm_factor = get_norm_factor(data, steps=100)
            if save_dir:
                with open(os.path.join(save_dir, "norm_factor.json"), "w") as f:
                    json.dump({"norm_factor": norm_factor}, f)
        
        for trainer in trainers:
            trainer.config["norm_factor"] = norm_factor
            # Verify that all autoencoders have a scale_biases method
            trainer.ae.scale_biases(1.0)
    
    # Keep track of regular checkpoint steps to avoid removing them during cleanup
    current_step = 0
    for step, act in enumerate(tqdm(data, total=steps)):
        current_step = step  # Update for signal handler
        
        if step >= steps:
            break
        
        act = act.to(device=device, dtype=autocast_dtype)
        
        if normalize_activations and norm_factor is not None:
            act /= norm_factor
        
        # Logging
        if (use_wandb or verbose) and log_steps and step % log_steps == 0:
            with t.no_grad():
                # Make a single log dict for all trainers
                log = {"step": step}
                z = act.clone()
                
                for i, trainer in enumerate(trainers):
                    trainer_log_name = f"trainer_{i}-{trainer.wandb_name.replace('/', '_')}"
                    act_local = z.clone()
                    if activations_split_by_head:  # x.shape: [batch, pos, n_heads, d_head]
                        act_local = act_local[..., i, :]
                    
                    if not transcoder:
                        act_local, act_hat, f, losslog = trainer.loss(act_local, step=step, logging=True)
                        
                        # L0
                        l0 = (f != 0).float().sum(dim=-1).mean().item()
                        # Fraction of variance explained
                        total_variance = t.var(act_local, dim=0).sum()
                        residual_variance = t.var(act_local - act_hat, dim=0).sum()
                        frac_variance_explained = 1 - residual_variance / total_variance
                        log[f"{trainer_log_name}/frac_variance_explained"] = frac_variance_explained.item()
                    else:  # transcoder
                        x, x_hat, f, losslog = trainer.loss(act_local, step=step, logging=True)
                        # L0
                        l0 = (f != 0).float().sum(dim=-1).mean().item()
                    
                    # Log parameters from training
                    for k, v in losslog.items():
                        log[f"{trainer_log_name}/{k}"] = v.cpu().item() if isinstance(v, t.Tensor) else v
                    
                    log[f"{trainer_log_name}/l0"] = l0
                    
                    trainer_log = trainer.get_logging_parameters()
                    for name, value in trainer_log.items():
                        if isinstance(value, t.Tensor):
                            value = value.cpu().item()
                        log[f"{trainer_log_name}/{name}"] = value
                
                if verbose:
                    print(f"Step {step}:")
                    for k, v in log.items():
                        if k != "step":
                            print(f"  {k} = {v}")
                
                if use_wandb and wandb_run:
                    wandb_run.log(log)
        
        # Regular checkpointing - modified to NOT save optimizer
        if save_steps is not None and step in save_steps:
            if save_dir is not None:
                save_checkpoint(save_dirs, trainers, step, norm_factor, save_optimizer=False)
                logger.info(f"Regular checkpoint saved at step {step}")
        
        # Training step
        for trainer in trainers:
            with autocast_context:
                trainer.update(step, act)
    
    # Save final SAEs
    for i, (save_dir, trainer) in enumerate(zip(save_dirs, trainers)):
        if save_dir is not None:
            if normalize_activations and norm_factor is not None:
                trainer.ae.scale_biases(norm_factor)
            
            # Save final model
            final = {k: v.cpu() for k, v in trainer.ae.state_dict().items()}
            t.save(final, os.path.join(save_dir, "ae.pt"))
            
            # Save final optimizer state
            if hasattr(trainer, 'optimizer'):
                final_opt = {k: v if not isinstance(v, t.Tensor) else v.cpu() 
                           for k, v in trainer.optimizer.state_dict().items()}
                t.save(final_opt, os.path.join(save_dir, "optimizer_final.pt"))
            
            if normalize_activations and norm_factor is not None:
                trainer.ae.scale_biases(1 / norm_factor)
    
    # Save completion marker
    if save_dir is not None:
        with open(os.path.join(save_dir, "training_completed.json"), "w") as f:
            json.dump({
                "completed_at": time.time(),
                "total_steps": steps,
                "actual_steps": current_step
            }, f)
    
    # Finalize wandb
    if use_wandb and wandb_run:
        wandb_run.finish()

    return trainers