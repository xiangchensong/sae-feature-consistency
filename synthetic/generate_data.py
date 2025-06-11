# %%
import pickle
import numpy as np
import os
# %%
def generate_synthetic_data(n, m, N, k, distribution='Gaussian', seed=None):
    if seed is not None:
        np.random.seed(seed)
    
    A_true = np.random.randn(m, n)
    # Normalize columns to unit l2 norm
    A_true = A_true / np.linalg.norm(A_true, axis=1, keepdims=True)
    S_true = np.zeros((N, m))
    
    if distribution == 'Gaussian':
        for i in range(N):
            # Randomly select k indices to be non-zero
            indices = np.random.choice(m, k, replace=False)
            # Assign random values to the selected indices
            values = np.random.randn(k)
            values = np.abs(values)  # Ensure non-negativity
            S_true[i, indices] = values
    elif distribution == 'Zipfian':
        raise NotImplementedError("Zipfian distribution is not implemented yet.")
    else:
        raise ValueError("Unsupported distribution.")

    X = np.dot(S_true, A_true)  # Generate observed data matrix

    return {
        'A': A_true,  # Dictionary matrix
        'S': S_true,  # Sparse coefficient matrix
        'X': X        # Observed data matrix
    }
# %%
config = {
    'k': 3,             # sparsity parameter
    'activation_dim': 16,            # input/output dimension
    'dict_size': 32,            # latent dimension
    'N': 50_000,           # number of data points
    'seed': 42,  # random seed
}
# %%
save_file_name = f"data/synthetic_data_n{config['activation_dim']}_m{config['dict_size']}_N{config['N']}_k{config['k']}_seed{config['seed']}.pkl"
if os.path.exists(save_file_name):
    print(f"File {save_file_name} already exists. Loading data...")
else:
    data = generate_synthetic_data(n = config['activation_dim'], m = config['dict_size'], N = config['N'], k = config['k'], seed=config['seed'])
    if not os.path.exists("data"):
        os.makedirs("data")
    with open(save_file_name, "wb") as f:
        pickle.dump(data, f)
# %%