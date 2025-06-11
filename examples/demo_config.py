from dataclasses import dataclass, asdict, field
from typing import Optional, Type, Any
from enum import Enum
import torch as t
import itertools

from dictionary_learning.trainers.standard import StandardTrainer, StandardTrainerAprilUpdate
from dictionary_learning.trainers.top_k import TopKTrainer, AutoEncoderTopK
from dictionary_learning.trainers.batch_top_k import BatchTopKTrainer, BatchTopKSAE
from dictionary_learning.trainers.gdm import GatedSAETrainer
from dictionary_learning.trainers.p_anneal import PAnnealTrainer
from dictionary_learning.trainers.jumprelu import JumpReluTrainer
from dictionary_learning.trainers.matryoshka_batch_top_k import (
    MatryoshkaBatchTopKTrainer,
    MatryoshkaBatchTopKSAE,
)
from dictionary_learning.dictionary import (
    AutoEncoder,
    GatedAutoEncoder,
    AutoEncoderNew,
    JumpReluAutoEncoder,
)


class TrainerType(Enum):
    STANDARD = "standard"
    STANDARD_NEW = "standard_new"
    TOP_K = "top_k"
    BATCH_TOP_K = "batch_top_k"
    GATED = "gated"
    P_ANNEAL = "p_anneal"
    JUMP_RELU = "jump_relu"
    Matryoshka_BATCH_TOP_K = "matryoshka_batch_top_k"


@dataclass
class LLMConfig:
    llm_batch_size: int
    context_length: int
    sae_batch_size: int
    dtype: t.dtype


@dataclass
class SparsityPenalties:
    standard: list[float]
    standard_new: list[float]
    p_anneal: list[float]
    gated: list[float]


num_tokens = 500_000_000 # 500M tokens

print(f"NOTE: Training on {num_tokens} tokens")

eval_num_inputs = 200
random_seeds = [42, 43, 44]
dictionary_widths = [2**14] # [2**12, 2**14, 2**16], [4k, 16k, 65k]

WARMUP_STEPS = 1000
SPARSITY_WARMUP_STEPS = 5000
DECAY_START_FRACTION = 0.8

learning_rates = [3e-4]

wandb_project = "sae-feature-consistency-sweep"

LLM_CONFIG = {
    "EleutherAI/pythia-70m-deduped": LLMConfig(
        llm_batch_size=64, context_length=1024, sae_batch_size=2048, dtype=t.float32
    ),
    "EleutherAI/pythia-160m-deduped": LLMConfig(
        llm_batch_size=32, context_length=1024, sae_batch_size=2048, dtype=t.float32
    ),
    "google/gemma-2-2b": LLMConfig(
        llm_batch_size=4, context_length=1024, sae_batch_size=2048, dtype=t.bfloat16
    ),
    # newly added
    "google/gemma-3-1b-pt": LLMConfig(
        llm_batch_size=8, context_length=1024, sae_batch_size=2048, dtype=t.bfloat16
    ),
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B": LLMConfig(
        llm_batch_size=6, context_length=1024, sae_batch_size=2048, dtype=t.bfloat16
    ),
    "HuggingFaceTB/SmolLM2-135M": LLMConfig(
        llm_batch_size=32, context_length=1024, sae_batch_size=2048, dtype=t.float16
    ),
}

SPARSITY_PENALTIES = SparsityPenalties(
    standard=[0.012, 0.015, 0.02, 0.03, 0.04, 0.06],
    standard_new=[0.012, 0.015, 0.02, 0.03, 0.04, 0.06],
    p_anneal=[0.006, 0.008, 0.01, 0.015, 0.02, 0.025],
    gated=[0.012, 0.018, 0.024, 0.04, 0.06, 0.08],
)


TARGET_L0s = [20, 40, 80, 160, 320, 640]


@dataclass
class BaseTrainerConfig:
    activation_dim: int
    device: str
    layer: str
    lm_name: str
    submodule_name: str
    trainer: Type[Any]
    dict_class: Type[Any]
    wandb_name: str
    warmup_steps: int
    steps: int
    decay_start: Optional[int]


@dataclass
class StandardTrainerConfig(BaseTrainerConfig):
    dict_size: int
    seed: int
    lr: float
    l1_penalty: float
    sparsity_warmup_steps: Optional[int]
    resample_steps: Optional[int] = None


@dataclass
class StandardNewTrainerConfig(BaseTrainerConfig):
    dict_size: int
    seed: int
    lr: float
    l1_penalty: float
    sparsity_warmup_steps: Optional[int]


@dataclass
class PAnnealTrainerConfig(BaseTrainerConfig):
    dict_size: int
    seed: int
    lr: float
    initial_sparsity_penalty: float
    sparsity_warmup_steps: Optional[int]
    sparsity_function: str = "Lp^p"
    p_start: float = 1.0
    p_end: float = 0.2
    anneal_start: int = 10000
    anneal_end: Optional[int] = None
    sparsity_queue_length: int = 10
    n_sparsity_updates: int = 10


@dataclass
class TopKTrainerConfig(BaseTrainerConfig):
    dict_size: int
    seed: int
    lr: float
    k: int
    auxk_alpha: float = 1 / 32
    threshold_beta: float = 0.999
    threshold_start_step: int = 1000  # when to begin tracking the average threshold


@dataclass
class MatryoshkaBatchTopKTrainerConfig(BaseTrainerConfig):
    dict_size: int
    seed: int
    lr: float
    k: int
    group_fractions: list[float] = field(
        default_factory=lambda: [
            (1 / 32),
            (1 / 16),
            (1 / 8),
            (1 / 4),
            ((1 / 2) + (1 / 32)),
        ]
    )
    group_weights: Optional[list[float]] = None
    auxk_alpha: float = 1 / 32
    threshold_beta: float = 0.999
    threshold_start_step: int = 1000  # when to begin tracking the average threshold


@dataclass
class GatedTrainerConfig(BaseTrainerConfig):
    dict_size: int
    seed: int
    lr: float
    l1_penalty: float
    sparsity_warmup_steps: Optional[int]


@dataclass
class JumpReluTrainerConfig(BaseTrainerConfig):
    dict_size: int
    seed: int
    lr: float
    target_l0: int
    sparsity_warmup_steps: Optional[int]
    sparsity_penalty: float = 1.0
    bandwidth: float = 0.001


def get_trainer_configs(
    architectures: list[str],
    learning_rates: list[float],
    seeds: list[int],
    activation_dim: int,
    dict_sizes: list[int],
    model_name: str,
    device: str,
    layer: str,
    submodule_name: str,
    steps: int,
    warmup_steps: int = WARMUP_STEPS,
    sparsity_warmup_steps: int = SPARSITY_WARMUP_STEPS,
    decay_start_fraction=DECAY_START_FRACTION,
) -> list[dict]:
    decay_start = int(steps * decay_start_fraction)

    trainer_configs = []

    base_config = {
        "activation_dim": activation_dim,
        "steps": steps,
        "warmup_steps": warmup_steps,
        "decay_start": decay_start,
        "device": device,
        "layer": layer,
        "lm_name": model_name,
        "submodule_name": submodule_name,
    }
    if TrainerType.P_ANNEAL.value in architectures:
        for seed, dict_size, learning_rate, sparsity_penalty in itertools.product(
            seeds, dict_sizes, learning_rates, SPARSITY_PENALTIES.p_anneal
        ):
            config = PAnnealTrainerConfig(
                **base_config,
                trainer=PAnnealTrainer,
                dict_class=AutoEncoder,
                sparsity_warmup_steps=sparsity_warmup_steps,
                lr=learning_rate,
                dict_size=dict_size,
                seed=seed,
                initial_sparsity_penalty=sparsity_penalty,
                wandb_name=f"PAnnealTrainer-{model_name}-{submodule_name}",
            )
            trainer_configs.append(asdict(config))

    if TrainerType.STANDARD.value in architectures:
        for seed, dict_size, learning_rate, l1_penalty in itertools.product(
            seeds, dict_sizes, learning_rates, SPARSITY_PENALTIES.standard
        ):
            config = StandardTrainerConfig(
                **base_config,
                trainer=StandardTrainer,
                dict_class=AutoEncoder,
                sparsity_warmup_steps=sparsity_warmup_steps,
                lr=learning_rate,
                dict_size=dict_size,
                seed=seed,
                l1_penalty=l1_penalty,
                wandb_name=f"StandardTrainer-{model_name}-{submodule_name}",
            )
            trainer_configs.append(asdict(config))

    if TrainerType.STANDARD_NEW.value in architectures:
        for seed, dict_size, learning_rate, l1_penalty in itertools.product(
            seeds, dict_sizes, learning_rates, SPARSITY_PENALTIES.standard_new
        ):
            config = StandardNewTrainerConfig(
                **base_config,
                trainer=StandardTrainerAprilUpdate,
                dict_class=AutoEncoder,
                sparsity_warmup_steps=sparsity_warmup_steps,
                lr=learning_rate,
                dict_size=dict_size,
                seed=seed,
                l1_penalty=l1_penalty,
                wandb_name=f"StandardTrainerNew-{model_name}-{submodule_name}",
            )
            trainer_configs.append(asdict(config))

    if TrainerType.GATED.value in architectures:
        for seed, dict_size, learning_rate, l1_penalty in itertools.product(
            seeds, dict_sizes, learning_rates, SPARSITY_PENALTIES.gated
        ):
            config = GatedTrainerConfig(
                **base_config,
                trainer=GatedSAETrainer,
                dict_class=GatedAutoEncoder,
                sparsity_warmup_steps=sparsity_warmup_steps,
                lr=learning_rate,
                dict_size=dict_size,
                seed=seed,
                l1_penalty=l1_penalty,
                wandb_name=f"GatedTrainer-{model_name}-{submodule_name}",
            )
            trainer_configs.append(asdict(config))

    if TrainerType.TOP_K.value in architectures:
        for seed, dict_size, learning_rate, k in itertools.product(
            seeds, dict_sizes, learning_rates, TARGET_L0s
        ):
            config = TopKTrainerConfig(
                **base_config,
                trainer=TopKTrainer,
                dict_class=AutoEncoderTopK,
                lr=learning_rate,
                dict_size=dict_size,
                seed=seed,
                k=k,
                wandb_name=f"TopKTrainer-{model_name}-{submodule_name}",
            )
            trainer_configs.append(asdict(config))

    if TrainerType.BATCH_TOP_K.value in architectures:
        for seed, dict_size, learning_rate, k in itertools.product(
            seeds, dict_sizes, learning_rates, TARGET_L0s
        ):
            config = TopKTrainerConfig(
                **base_config,
                trainer=BatchTopKTrainer,
                dict_class=BatchTopKSAE,
                lr=learning_rate,
                dict_size=dict_size,
                seed=seed,
                k=k,
                wandb_name=f"BatchTopKTrainer-{model_name}-{submodule_name}",
            )
            trainer_configs.append(asdict(config))

    if TrainerType.Matryoshka_BATCH_TOP_K.value in architectures:
        for seed, dict_size, learning_rate, k in itertools.product(
            seeds, dict_sizes, learning_rates, TARGET_L0s
        ):
            config = MatryoshkaBatchTopKTrainerConfig(
                **base_config,
                trainer=MatryoshkaBatchTopKTrainer,
                dict_class=MatryoshkaBatchTopKSAE,
                lr=learning_rate,
                dict_size=dict_size,
                seed=seed,
                k=k,
                wandb_name=f"MatryoshkaBatchTopKTrainer-{model_name}-{submodule_name}",
            )
            trainer_configs.append(asdict(config))

    if TrainerType.JUMP_RELU.value in architectures:
        for seed, dict_size, learning_rate, target_l0 in itertools.product(
            seeds, dict_sizes, learning_rates, TARGET_L0s
        ):
            config = JumpReluTrainerConfig(
                **base_config,
                trainer=JumpReluTrainer,
                dict_class=JumpReluAutoEncoder,
                sparsity_warmup_steps=sparsity_warmup_steps,
                lr=learning_rate,
                dict_size=dict_size,
                seed=seed,
                target_l0=target_l0,
                wandb_name=f"JumpReluTrainer-{model_name}-{submodule_name}",
            )
            trainer_configs.append(asdict(config))

    return trainer_configs

def get_single_trainer_config(
    architecture: str,
    learning_rate: str,
    seed: int,
    activation_dim: int,
    dict_size: int,
    model_name: str,
    device: str,
    layer: str,
    submodule_name: str,
    steps: int,
    warmup_steps: int = WARMUP_STEPS,
    sparsity_warmup_steps: int = SPARSITY_WARMUP_STEPS,
    decay_start_fraction=DECAY_START_FRACTION,
    **kwargs: dict,
) -> dict:
    """
    Create configuration for a specific trainer type based on architecture.
    
    Validates that required parameters exist for each architecture type:
    - P_ANNEAL, STANDARD, STANDARD_NEW, GATED require sparsity_penalty
    - TOP_K, BATCH_TOP_K, Matryoshka_BATCH_TOP_K, JUMP_RELU require target_l0
    
    Returns:
        dict: Configuration dictionary for the specified trainer architecture.
    """
    # Define which architectures require which parameters
    architectures_requiring_sparsity_penalty = [
        TrainerType.P_ANNEAL.value,
        TrainerType.STANDARD.value,
        TrainerType.STANDARD_NEW.value,
        TrainerType.GATED.value,
    ]
    
    architectures_requiring_target_l0 = [
        TrainerType.TOP_K.value,
        TrainerType.BATCH_TOP_K.value,
        TrainerType.Matryoshka_BATCH_TOP_K.value,
        TrainerType.JUMP_RELU.value,
    ]
    
    # Validate required parameters based on architecture
    sparsity_penalty = kwargs.get("sparsity_penalty") if kwargs else None
    target_l0 = kwargs.get("target_l0") if kwargs else None
    if architecture in architectures_requiring_sparsity_penalty and sparsity_penalty is None:
        raise ValueError(f"Architecture {architecture} requires sparsity_penalty parameter")
    
    if architecture in architectures_requiring_target_l0 and target_l0 is None:
        raise ValueError(f"Architecture {architecture} requires target_l0 parameter")
    
    # Calculate decay start based on total steps
    decay_start = int(steps * decay_start_fraction)
    
    # Common configuration for all trainer types
    base_config = {
        "activation_dim": activation_dim,
        "steps": steps,
        "warmup_steps": warmup_steps,
        "decay_start": decay_start,
        "device": device,
        "layer": layer,
        "lm_name": model_name,
        "submodule_name": submodule_name,
    }
    
    # Define configuration mapping for different trainer architectures
    trainer_configs = {
        TrainerType.P_ANNEAL.value: {
            "trainer": PAnnealTrainer,
            "dict_class": AutoEncoder,
            "sparsity_warmup_steps": sparsity_warmup_steps,
            "lr": learning_rate,
            "dict_size": dict_size,
            "seed": seed,
            "initial_sparsity_penalty": sparsity_penalty,
            "wandb_name": f"PAnnealTrainer-{model_name}-{submodule_name}-lr{learning_rate}-ds{dict_size}-sp{sparsity_penalty}-{seed}",
            "config_class": PAnnealTrainerConfig,
        },
        TrainerType.STANDARD.value: {
            "trainer": StandardTrainer,
            "dict_class": AutoEncoder,
            "sparsity_warmup_steps": sparsity_warmup_steps,
            "lr": learning_rate,
            "dict_size": dict_size,
            "seed": seed,
            "l1_penalty": sparsity_penalty,
            "wandb_name": f"StandardTrainer-{model_name}-{submodule_name}-lr{learning_rate}-ds{dict_size}-sp{sparsity_penalty}-{seed}",
            "config_class": StandardTrainerConfig,
        },
        TrainerType.STANDARD_NEW.value: {
            "trainer": StandardTrainerAprilUpdate,
            "dict_class": AutoEncoder,
            "sparsity_warmup_steps": sparsity_warmup_steps,
            "lr": learning_rate,
            "dict_size": dict_size,
            "seed": seed,
            "l1_penalty": sparsity_penalty,
            "wandb_name": f"StandardTrainerNew-{model_name}-{submodule_name}-lr{learning_rate}-ds{dict_size}-sp{sparsity_penalty}-{seed}",
            "config_class": StandardNewTrainerConfig,
        },
        TrainerType.GATED.value: {
            "trainer": GatedSAETrainer,
            "dict_class": GatedAutoEncoder,
            "sparsity_warmup_steps": sparsity_warmup_steps,
            "lr": learning_rate,
            "dict_size": dict_size,
            "seed": seed,
            "l1_penalty": sparsity_penalty,
            "wandb_name": f"GatedTrainer-{model_name}-{submodule_name}-lr{learning_rate}-ds{dict_size}-sp{sparsity_penalty}-{seed}",
            "config_class": GatedTrainerConfig,
        },
        TrainerType.TOP_K.value: {
            "trainer": TopKTrainer,
            "dict_class": AutoEncoderTopK,
            "lr": learning_rate,
            "dict_size": dict_size,
            "seed": seed,
            "k": target_l0,
            "wandb_name": f"TopKTrainer-{model_name}-{submodule_name}-lr{learning_rate}-ds{dict_size}-k{target_l0}-{seed}",
            "config_class": TopKTrainerConfig,
        },
        TrainerType.BATCH_TOP_K.value: {
            "trainer": BatchTopKTrainer,
            "dict_class": BatchTopKSAE,
            "lr": learning_rate,
            "dict_size": dict_size,
            "seed": seed,
            "k": target_l0,
            "wandb_name": f"BatchTopKTrainer-{model_name}-{submodule_name}-lr{learning_rate}-ds{dict_size}-k{target_l0}-{seed}",
            "config_class": TopKTrainerConfig,
        },
        TrainerType.Matryoshka_BATCH_TOP_K.value: {
            "trainer": MatryoshkaBatchTopKTrainer,
            "dict_class": MatryoshkaBatchTopKSAE,
            "lr": learning_rate,
            "dict_size": dict_size,
            "seed": seed,
            "k": target_l0,
            "wandb_name": f"MatryoshkaBatchTopKTrainer-{model_name}-{submodule_name}-lr{learning_rate}-ds{dict_size}-k{target_l0}-{seed}",
            "config_class": MatryoshkaBatchTopKTrainerConfig,
        },
        TrainerType.JUMP_RELU.value: {
            "trainer": JumpReluTrainer,
            "dict_class": JumpReluAutoEncoder,
            "sparsity_warmup_steps": sparsity_warmup_steps,
            "lr": learning_rate,
            "dict_size": dict_size,
            "seed": seed,
            "target_l0": target_l0,
            "wandb_name": f"JumpReluTrainer-{model_name}-{submodule_name}-lr{learning_rate}-ds{dict_size}-k{target_l0}-{seed}",
            "config_class": JumpReluTrainerConfig,
        },
    }
    
    # Get the trainer configuration or raise error if architecture not found
    trainer_config = trainer_configs.get(architecture)
    if not trainer_config:
        raise ValueError(f"Unknown architecture: {architecture}")
    
    # Extract config class and remove it from the parameters
    config_class = trainer_config.pop("config_class")
    
    # Create and return the configuration
    return asdict(config_class(**base_config, **trainer_config))