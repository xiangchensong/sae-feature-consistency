from datasets import load_dataset
import zstandard as zstd
import io
import json
import os
from nnsight import LanguageModel

from .trainers.top_k import AutoEncoderTopK
from .trainers.batch_top_k import BatchTopKSAE
from .trainers.matryoshka_batch_top_k import MatryoshkaBatchTopKSAE
from .dictionary import (
    AutoEncoder,
    GatedAutoEncoder,
    AutoEncoderNew,
    JumpReluAutoEncoder,
)


def hf_dataset_to_generator(dataset_name, split="train", streaming=True):
    dataset = load_dataset(dataset_name, split=split, streaming=streaming)

    def gen():
        for x in iter(dataset):
            yield x["text"]

    return gen()


def zst_to_generator(data_path):
    """
    Load a dataset from a .jsonl.zst file.
    The jsonl entries is assumed to have a 'text' field
    """
    compressed_file = open(data_path, "rb")
    dctx = zstd.ZstdDecompressor()
    reader = dctx.stream_reader(compressed_file)
    text_stream = io.TextIOWrapper(reader, encoding="utf-8")

    def generator():
        for line in text_stream:
            yield json.loads(line)["text"]

    return generator()


def get_nested_folders(path: str) -> list[str]:
    """
    Recursively get a list of folders that contain an ae.pt file, starting the search from the given path
    """
    folder_names = []

    for root, dirs, files in os.walk(path):
        if "ae.pt" in files:
            folder_names.append(root)

    return folder_names


def load_dictionary(base_path: str, device: str) -> tuple:
    ae_path = f"{base_path}/ae.pt"
    config_path = f"{base_path}/config.json"

    with open(config_path, "r") as f:
        config = json.load(f)

    dict_class = config["trainer"]["dict_class"]

    if dict_class == "AutoEncoder":
        dictionary = AutoEncoder.from_pretrained(ae_path, device=device)
    elif dict_class == "GatedAutoEncoder":
        dictionary = GatedAutoEncoder.from_pretrained(ae_path, device=device)
    elif dict_class == "AutoEncoderNew":
        dictionary = AutoEncoderNew.from_pretrained(ae_path, device=device)
    elif dict_class == "AutoEncoderTopK":
        k = config["trainer"]["k"]
        dictionary = AutoEncoderTopK.from_pretrained(ae_path, k=k, device=device)
    elif dict_class == "BatchTopKSAE":
        k = config["trainer"]["k"]
        dictionary = BatchTopKSAE.from_pretrained(ae_path, k=k, device=device)
    elif dict_class == "MatryoshkaBatchTopKSAE":
        k = config["trainer"]["k"]
        dictionary = MatryoshkaBatchTopKSAE.from_pretrained(ae_path, k=k, device=device)
    elif dict_class == "JumpReluAutoEncoder":
        dictionary = JumpReluAutoEncoder.from_pretrained(ae_path, device=device)
    else:
        raise ValueError(f"Dictionary class {dict_class} not supported")

    return dictionary, config


def get_submodule(model: LanguageModel, layer: int):
    """Gets the residual stream submodule"""
    model_name = model._model_key

    if "pythia" in model_name:
        return model.gpt_neox.layers[layer]
    elif "gemma" in model_name:
        return model.model.layers[layer]
    else:
        raise ValueError(f"Please add submodule for model {model_name}")


from scipy.optimize import linear_sum_assignment
import numpy as np
import warnings
def MCC(A_1, A_2, dict_size = None):
    """
    Compute the Matthews correlation coefficient between two matrices.
    """
    assert A_1.shape == A_2.shape, "The two matrices must have the same shape."
        
    if dict_size is not None:
        # Ensure A_1 and A_2 are of the same size and the size match the dict_size
        assert A_1.shape[0] == dict_size, f"A_1 and A_2 must have the {dict_size} rows"

    # Normalize columns of both matrices
    A_1_norm = np.linalg.norm(A_1, axis=1, keepdims=True)
    A_2_norm = np.linalg.norm(A_2, axis=1, keepdims=True)
    
    # Avoid division by zero
    if np.any(A_1_norm == 0) or np.any(A_2_norm == 0):
        warnings.warn("A_1 and A_2 have zero columns, which may affect the MCC calculation.")
        A_1_norm[A_1_norm == 0] = 1
        A_2_norm[A_2_norm == 0] = 1
    
    A_1 = A_1 / A_1_norm
    A_2 = A_2 / A_2_norm

    # Calculate the cost matrix using inner products
    cost_matrix = 1 - np.matmul(A_1, A_2.T)

    # Solve the assignment problem using Hungarian algorithm
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    # Calculate the MCC
    mcc = 1 - cost_matrix[row_ind, col_ind].sum() / len(row_ind)
    return mcc