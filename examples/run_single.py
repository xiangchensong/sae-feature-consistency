import torch as t
from nnsight import LanguageModel
import argparse
import itertools
import os
import random
import json
import torch.multiprocessing as mp
import time
import huggingface_hub
from datasets import config
import numpy as np

import demo_config
from dictionary_learning.utils import hf_dataset_to_generator
from dictionary_learning.buffer import ActivationBuffer
from dictionary_learning.evaluation import evaluate
from dictionary_learning.training_new import trainSAE
import dictionary_learning.utils as utils
import logging

t.use_deterministic_algorithms(True)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)





def get_submodule(model: LanguageModel, layer: int):
    """Gets the residual stream submodule"""
    model_name = model._model_key

    if "pythia" in model_name:
        return model.gpt_neox.layers[layer]
    elif "gemma" in model_name:
        return model.model.layers[layer]
    elif "SmolLM2" in model_name:
        return model.model.layers[layer]
    elif "DeepSeek" in model_name:
        return model.model.layers[layer]
    else:
        raise ValueError(f"Please add submodule for model {model_name}")



def run_sae_training(
    model_name: str,
    layer: int,
    save_dir: str,
    device: str,
    architectures: list,
    num_tokens: int,
    random_seeds: list[int],
    dictionary_widths: list[int],
    learning_rates: list[float],
    dry_run: bool = False,
    use_wandb: bool = False,
    save_checkpoints: bool = False,
    buffer_tokens: int = 250_000,
    **kwargs,
):
    random.seed(demo_config.random_seeds[0])
    t.manual_seed(demo_config.random_seeds[0])
    np.random.seed(demo_config.random_seeds[0])
    # model and data parameters
    context_length = demo_config.LLM_CONFIG[model_name].context_length

    llm_batch_size = demo_config.LLM_CONFIG[model_name].llm_batch_size
    sae_batch_size = demo_config.LLM_CONFIG[model_name].sae_batch_size
    dtype = demo_config.LLM_CONFIG[model_name].dtype

    num_buffer_inputs = buffer_tokens // context_length
    logger.info(f"buffer_size: {num_buffer_inputs}, buffer_size_in_tokens: {buffer_tokens}")

    log_steps = 100  # Log the training on wandb or logger.info to console every log_steps

    steps = int(num_tokens / sae_batch_size)  # Total number of batches to train

    if save_checkpoints:
        # Creates checkpoints at 0.0%, 0.1%, 0.316%, 1%, 3.16%, 10%, 31.6%, 100% of training
        log_checkpoints = t.logspace(-3, 0, 7).tolist()
        linear_checkpoints = t.linspace(0,1,11).tolist()
        desired_checkpoints = list(set([0.0] + log_checkpoints[:-1] + linear_checkpoints[:-1]))
        desired_checkpoints.sort()
        logger.info(f"desired_checkpoints: {desired_checkpoints}")

        save_steps = [int(steps * step) for step in desired_checkpoints]
        save_steps.sort()
        logger.info(f"save_steps: {save_steps}")
    else:
        save_steps = None

    model = LanguageModel(model_name, dispatch=True, device_map=device)
    model = model.to(dtype=dtype)
    submodule = get_submodule(model, layer)
    submodule_name = f"resid_post_layer_{layer}"
    io = "out"
    activation_dim = model.config.hidden_size

    generator = hf_dataset_to_generator(args.dataset_name)

    activation_buffer = ActivationBuffer(
        generator,
        model,
        submodule,
        n_ctxs=num_buffer_inputs,
        ctx_len=context_length,
        refresh_batch_size=llm_batch_size,
        out_batch_size=sae_batch_size,
        io=io,
        d_submodule=activation_dim,
        device=device,
    )
    trainer_configs = []
    for architecture, learning_rate, random_seed, dictionary_width in itertools.product(
        architectures,
        learning_rates,
        random_seeds,
        dictionary_widths,
    ):  
        if architecture in [
            demo_config.TrainerType.BATCH_TOP_K.value,
            demo_config.TrainerType.TOP_K.value,
            demo_config.TrainerType.Matryoshka_BATCH_TOP_K.value,
            demo_config.TrainerType.JUMP_RELU.value,
        ]:
            target_l0s = kwargs.get("target_l0s")
            if target_l0s is None:
                target_l0s = demo_config.TARGET_L0s
                logger.info(f"Using default target l0s for {architecture}: {target_l0s}")
            if target_l0s is None:
                raise ValueError(f"Please provide target l0s for {architecture}")
            for target_l0 in target_l0s:
                config = demo_config.get_single_trainer_config(
                    architecture,
                    learning_rate,
                    random_seed,
                    activation_dim,
                    dictionary_width,
                    model_name,
                    device,
                    layer,
                    submodule_name,
                    steps,
                    target_l0=target_l0,
                )
                trainer_configs.append(config)
        elif architecture in [
            demo_config.TrainerType.STANDARD.value,
            demo_config.TrainerType.STANDARD_NEW.value,
            demo_config.TrainerType.GATED.value,
            demo_config.TrainerType.P_ANNEAL.value,
        ]:
            sparsity_penalties = kwargs.get("sparsity_penalties")
            if kwargs.get("sparsity_penalties") is None:
                sparsity_penalties = getattr(demo_config.SPARSITY_PENALTIES, architecture)
                logger.info(f"Using default sparsity penalties for {architecture}: {sparsity_penalties}")
            if sparsity_penalties is None:
                raise ValueError(f"Please provide sparsity penalties for {architecture}")
            for sparsity_penalty in sparsity_penalties:
                config = demo_config.get_single_trainer_config(
                    architecture,
                    learning_rate,
                    random_seed,
                    activation_dim,
                    dictionary_width,
                    model_name,
                    device,
                    layer,
                    submodule_name,
                    steps,
                    sparsity_penalty=sparsity_penalty,
                )
                trainer_configs.append(config)
        else:
            raise ValueError("Please provide either sparsity penalties or target l0s")


    logger.info(f"len trainer configs: {len(trainer_configs)}")
    assert len(trainer_configs) > 0

    if kwargs.get("wandb_name") is not None:
        run_cfg = {
            "wandb_name": kwargs.get("wandb_name"),
        }
    else:
        run_cfg = {}
    if not dry_run:
        # actually run the sweep
        trainSAE(
            data=activation_buffer,
            trainer_configs=trainer_configs,
            steps=steps,
            use_wandb=use_wandb,
            wandb_project=demo_config.wandb_project,
            save_steps=save_steps,
            save_dir=save_dir,
            log_steps=log_steps,
            normalize_activations=True,
            verbose=False,
            device=device,
            autocast_dtype=t.bfloat16,
            run_cfg=run_cfg
        )


@t.no_grad()
def eval_saes(
    model_name: str,
    ae_paths: list[str],
    n_inputs: int,
    device: str,
    overwrite_prev_results: bool = False,
    transcoder: bool = False,
) -> dict:
    random.seed(demo_config.random_seeds[0])
    t.manual_seed(demo_config.random_seeds[0])

    if transcoder:
        io = "in_and_out"
    else:
        io = "out"

    context_length = demo_config.LLM_CONFIG[model_name].context_length
    llm_batch_size = demo_config.LLM_CONFIG[model_name].llm_batch_size
    loss_recovered_batch_size = max(llm_batch_size // 5, 1)
    sae_batch_size = loss_recovered_batch_size * context_length
    dtype = demo_config.LLM_CONFIG[model_name].dtype

    model = LanguageModel(model_name, dispatch=True, device_map=device)
    model = model.to(dtype=dtype)

    buffer_size = n_inputs
    io = "out"
    n_batches = n_inputs // loss_recovered_batch_size

    generator = hf_dataset_to_generator(args.dataset_name)

    input_strings = []
    for i, example in enumerate(generator):
        input_strings.append(example)
        if i > n_inputs * 5:
            break

    eval_results = {}

    for ae_path in ae_paths:
        output_filename = f"{ae_path}/eval_results.json"
        if not overwrite_prev_results:
            if os.path.exists(output_filename):
                logger.info(f"Skipping {ae_path} as eval results already exist")
                continue

        dictionary, config = utils.load_dictionary(ae_path, device)
        dictionary = dictionary.to(dtype=model.dtype)

        layer = config["trainer"]["layer"]
        submodule = get_submodule(model, layer)

        activation_dim = config["trainer"]["activation_dim"]

        activation_buffer = ActivationBuffer(
            iter(input_strings),
            model,
            submodule,
            n_ctxs=buffer_size,
            ctx_len=context_length,
            refresh_batch_size=llm_batch_size,
            out_batch_size=sae_batch_size,
            io=io,
            d_submodule=activation_dim,
            device=device,
        )

        eval_results = evaluate(
            dictionary,
            activation_buffer,
            context_length,
            loss_recovered_batch_size,
            io=io,
            device=device,
            n_batches=n_batches,
        )

        hyperparameters = {
            "n_inputs": n_inputs,
            "context_length": context_length,
        }
        eval_results["hyperparameters"] = hyperparameters

        logger.info(eval_results)

        with open(output_filename, "w") as f:
            json.dump(eval_results, f)

    # return the final eval_results for testing purposes
    return eval_results


def push_to_huggingface(save_dir: str, repo_id: str):
    api = huggingface_hub.HfApi()

    api.upload_folder(
        folder_path=save_dir,
        repo_id=repo_id,
        repo_type="model",
        path_in_repo=save_dir,
    )

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", type=str, required=True, help="where to store sweep")
    parser.add_argument("--use_wandb", action="store_true", help="use wandb logging")
    parser.add_argument("--wandb_name", type=str, default=None, help="wandb name")
    parser.add_argument("--dry_run", action="store_true", help="dry run sweep")
    parser.add_argument("--save_checkpoints", action="store_true", help="save checkpoints")
    parser.add_argument("--dataset_name", type=str, default="monology/pile-uncopyrighted", help="dataset name")
    parser.add_argument(
        "--layers", type=int, nargs="+", required=True, help="layers to train SAE on"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="which language model to use",
    )
    parser.add_argument(
        "--architectures",
        type=str,
        nargs="+",
        choices=[e.value for e in demo_config.TrainerType],
        required=True,
        help="which SAE architectures to train",
    )
    parser.add_argument(
        "--random_seeds",
        type=int,
        nargs="+",
        default=demo_config.random_seeds,
        help="random seeds to use",
    )
    parser.add_argument(
        "--dictionary_widths",
        type=int,
        nargs="+",
        default=demo_config.dictionary_widths,
        help="dictionary widths to use",
    )
    parser.add_argument(
        "--learning_rates",
        type=float,
        nargs="+",
        default=demo_config.learning_rates,
        help="learning rates to use",
    )
    parser.add_argument(
        "--target_l0s",
        type=int,
        nargs="+",
        default=None,
        help="target l0s to use",
    )
    parser.add_argument(
        "--sparsity_penalties",
        type=float,
        nargs="+",
        default=None,
        help="sparsity penalties to use",
    )
    parser.add_argument("--device", type=str, default="cuda:0", help="device to train on")
    parser.add_argument("--hf_repo_id", type=str, help="Hugging Face repo ID to push results to")
    args = parser.parse_args()
    
    logger.info("args: %s", json.dumps(vars(args), indent=4))
    return args
    
    
if __name__ == "__main__":
    """
    python run_single.py --save_checkpoints \
        --use_wandb \
        --wandb_name debug \
        --save_dir ./output/pythia-160m/ \
        --model_name EleutherAI/pythia-160m-deduped \
        --layers 8 \
        --architectures batch_top_k \
        --random_seeds 42 43 44 \
        --dictionary_widths 16384 \
        --learning_rates 3e-4 \
        --target_l0s 20 40 80 \
        --device cuda:0 \
        --dataset_name monology/pile-uncopyrighted
    """
    
    args = get_args()

    hf_repo_id = args.hf_repo_id

    if hf_repo_id:
        assert huggingface_hub.repo_exists(repo_id=hf_repo_id, repo_type="model")

    # This prevents random CUDA out of memory errors
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    # For wandb to work with multiprocessing
    # mp.set_start_method("spawn", force=True)

    # Rarely I have internet issues on cloud GPUs and then the streaming read fails
    # Hopefully the outage is shorter than 100 * 20 seconds
    config.STREAMING_READ_MAX_RETRIES = 100
    config.STREAMING_READ_RETRY_INTERVAL = 20

    start_time = time.time()
    save_dir = args.save_dir
    for layer in args.layers:
        run_sae_training(
            model_name=args.model_name,
            layer=layer,
            save_dir=save_dir,
            device=args.device,
            architectures=args.architectures,
            num_tokens=demo_config.num_tokens,
            random_seeds=args.random_seeds,
            dictionary_widths=args.dictionary_widths,
            learning_rates=args.learning_rates,
            dry_run=args.dry_run,
            use_wandb=args.use_wandb,
            save_checkpoints=args.save_checkpoints,
            target_l0s=args.target_l0s,
            sparsity_penalties=args.sparsity_penalties,
            wandb_name=args.wandb_name,
        )

    ae_paths = utils.get_nested_folders(save_dir)

    eval_saes(
        args.model_name,
        ae_paths,
        demo_config.eval_num_inputs,
        args.device,
        overwrite_prev_results=True,
    )

    logger.info(f"Total time: {time.time() - start_time}")

    if hf_repo_id:
        push_to_huggingface(save_dir, hf_repo_id)