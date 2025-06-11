# Real LLM Activations

An example run for `EleutherAI/pythia-160m-deduped` model with default hyperparameter sweep using slurm job is provided in [`pythia-160m-ds16k.slurm`](./pythia-160m-ds16k.slurm).

Make sure to edit the paths in the scripts. This will reproduce the main experiment in Figure 7.

Details about the default hyperparameter sweep such as `sparsity penalities` and `target L0s` can be found in [`demo_config.py`](./demo_config.py)

After the training, the MCC can be calculated with [`evaluate_mcc.py`](./evaluate_mcc.py).