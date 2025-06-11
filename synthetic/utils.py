import torch
import numpy as np
from scipy.optimize import linear_sum_assignment
import warnings
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def plot_loss_plotly(losses, labels=None):
    # Determine if we're dealing with multiple plots
    is_multiple = isinstance(losses[0], list) or isinstance(losses[0], np.ndarray)
    
    if is_multiple:
        # Get the titles for each subplot
        titles = labels if isinstance(labels, list) else [labels] * len(losses)
        
        # Create a figure with subplots (one row for each metric)
        fig = make_subplots(
            rows=len(losses), 
            cols=1,
            subplot_titles=titles,
            vertical_spacing=0.1
        )
        
        # Add each metric to its own subplot
        for i, (loss_data, plot_title) in enumerate(zip(losses, titles)):
            # Convert to list if numpy array for easier indexing
            loss_list = loss_data.tolist() if isinstance(loss_data, np.ndarray) else loss_data
            
            # Generate x-values (indexes)
            x_values = list(range(len(loss_list)))
            
            # Main line trace
            fig.add_trace(
                go.Scatter(
                    x=x_values,
                    y=loss_list,
                    mode='lines',
                    name=plot_title
                ),
                row=i+1, 
                col=1
            )
            
            # Find max value and its index
            max_val = max(loss_list)
            max_idx = loss_list.index(max_val)
            
            # Find min value and its index
            min_val = min(loss_list)
            min_idx = loss_list.index(min_val)
            
            # Add annotations for max and min points
            fig.update_layout(
                annotations=fig.layout.annotations + (
                    # Max annotation - FLIPPED: now points downward
                    dict(
                        x=max_idx,
                        y=max_val,
                        xref=f"x{i+1}" if i > 0 else "x",
                        yref=f"y{i+1}" if i > 0 else "y",
                        text=f"Max: {max_val:.4f}",
                        showarrow=True,
                        arrowhead=4,
                        ax=0,
                        ay=40  # Changed from -40 to 40 (pointing down)
                    ),
                    # Min annotation - FLIPPED: now points upward
                    dict(
                        x=min_idx,
                        y=min_val,
                        xref=f"x{i+1}" if i > 0 else "x",
                        yref=f"y{i+1}" if i > 0 else "y",
                        text=f"Min: {min_val:.4f}",
                        showarrow=True,
                        arrowhead=4,
                        ax=0,
                        ay=-40  # Changed from 40 to -40 (pointing up)
                    )
                )
            )
            
        # Adjust height based on number of subplots
        subplot_height = 250  # Height per subplot
        total_height = subplot_height * len(losses)
            
        # Update layout
        fig.update_layout(
            height=total_height,
            width=900,
            title="Training Metrics",
            showlegend=False
        )
    
    else:
        # Single plot case
        # Convert to list if numpy array for easier indexing
        loss_list = losses.tolist() if isinstance(losses, np.ndarray) else losses
        
        # Generate x-values (indexes)
        x_values = list(range(len(loss_list)))
        
        fig = go.Figure()
        # Main line trace
        fig.add_trace(go.Scatter(
            x=x_values,
            y=loss_list,
            mode='lines',
            name=labels if labels else 'Loss'
        ))
        
        # Find max value and its index
        max_val = max(loss_list)
        max_idx = loss_list.index(max_val)
        
        # Find min value and its index
        min_val = min(loss_list)
        min_idx = loss_list.index(min_val)
        
        # Add annotations for max and min points
        fig.update_layout(
            annotations=[
                # Max annotation - FLIPPED: now points downward
                dict(
                    x=max_idx,
                    y=max_val,
                    xref="x",
                    yref="y",
                    text=f"Max: {max_val:.4f}",
                    showarrow=True,
                    arrowhead=4,
                    ax=0,
                    ay=20  # Changed from -40 to 40 (pointing down)
                ),
                # Min annotation - FLIPPED: now points upward
                dict(
                    x=min_idx,
                    y=min_val,
                    xref="x",
                    yref="y",
                    text=f"Min: {min_val:.4f}",
                    showarrow=True,
                    arrowhead=4,
                    ax=0,
                    ay=-20  # Changed from 40 to -40 (pointing up)
                )
            ]
        )
        
        # Update layout
        fig.update_layout(
            height=500,
            width=900,
            showlegend=False
        )
    
    fig.show()



# def check_data(data, n, m, N, k):
#     """Check the generated data for correctness.
    
#     Parameters
#     ----------
#     data : dict
#         Dictionary containing:
#         - 'A': Dictionary matrix (n x m)
#         - 'S': Sparse coefficient matrix (N x n)
#         - 'X': Data matrix (N x m)
#     n : int
#         input/output dimension
#     m : int
#         Feature dimension
#     N : int
#         Number of data points
#     k : int
#         Sparsity level (non-zero coefficients per data point)
    
#     Returns
#     -------
#     bool
#         True if all checks pass, False otherwise.
#     """
#     assert data['A'].shape == (n, m), "Shape of A is incorrect"
#     assert data['S'].shape == (N, m), "Shape of S is incorrect"
#     assert data['X'].shape == (N, n), "Shape of X is incorrect"
    
#     # Check if X = S @ A^T
#     assert torch.allclose(data['X'], torch.matmul(data['S'], data['A'].T), atol=1e-6), "X does not equal S @ A^T"
    
#     # Check sparsity of S
#     assert torch.all(torch.sum(data['S'] != 0, dim=1) == k).item(), "S is not k-sparse for each row"
#     return True




def MCC(A, A_est, dict_size=None, return_dict = False):
    """Calculate the Matching Coefficient (MCC) between two matrices.
    
    Parameters
    ----------
    A : numpy.ndarray or torch.Tensor
        True dictionary matrix (m x n)
    A_est : numpy.ndarray or torch.Tensor
        Estimated dictionary matrix (m x n)
    
    Returns
    -------
    float
        The Matching Coefficient (MCC) between A and A_est.
    """
    # Convert torch tensors to numpy arrays if needed
    if isinstance(A, torch.Tensor):
        A = A.cpu().numpy()
    if isinstance(A_est, torch.Tensor):
        A_est = A_est.cpu().numpy()
    
    if dict_size is not None:
        # Ensure A and A_est are of the same size
        assert A.shape == A_est.shape, "A and A_est must have the same shape"
        assert A.shape[0] == dict_size, f"A and A_est must have the {dict_size} rows"

    # Normalize columns of both matrices
    A_norm = np.linalg.norm(A, axis=1, keepdims=True, ord=2)
    A_est_norm = np.linalg.norm(A_est, axis=1, keepdims=True,  ord=2)
    
    # Avoid division by zero
    if any(A_norm == 0) or any(A_est_norm == 0):
        warnings.warn("A and A_est have zero columns, which may affect the MCC calculation.")
        A_norm[A_norm == 0] = 1
        A_est_norm[A_est_norm == 0] = 1
    
    A = A / A_norm
    A_est = A_est / A_est_norm

    # Calculate the cost matrix using inner products
    cost_matrix = 1 - np.matmul(A, A_est.T)

    # Solve the assignment problem using Hungarian algorithm
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    # Calculate the MCC
    mcc = 1 - cost_matrix[row_ind, col_ind].sum() / len(row_ind)
    
    A_est_projected = A_est[col_ind]
    error = np.sum((A - A_est_projected) ** 2)
    if return_dict:
        return {
            'mcc': mcc,
            'error': error,
            'col_ind': col_ind,
        }
    return mcc

@torch.no_grad()
def set_decoder_norm_to_unit_norm(
    W_dec_DF: torch.nn.Parameter, activation_dim: int, d_sae: int
) -> torch.Tensor:
    """There's a major footgun here: we use this with both nn.Linear and nn.Parameter decoders.
    nn.Linear stores the decoder weights in a transposed format (d_model, d_sae). So, we pass the dimensions in
    to catch this error."""

    D, F = W_dec_DF.shape

    assert D == activation_dim
    assert F == d_sae

    eps = torch.finfo(W_dec_DF.dtype).eps
    norm = torch.norm(W_dec_DF.data, dim=0, keepdim=True)
    W_dec_DF.data /= norm + eps
    return W_dec_DF.data


if __name__ == "__main__":
    # Test generate_synthetic_data function
    # Set parameters
    n = 50          # input/output dimension
    m= 100          # latent dimension
    N = 1000        # number of data points
    k = 5           # sparsity parameter

    # data = generate_synthetic_data(n=n, m=m, N=N, k=k, distribution='Gaussian')
    # if check_data(data, n, m, N, k):
    #     print("Gaussian data check passed!")
    
    A = torch.randn(m, n) + 1
    A_est = torch.randn(m, n) + 1
    # Calculate MCC
    mcc = MCC(A, A_est)
    print(f"random two As MCC: {mcc:.4f}")
    # Now test if A_est is a permutation and scale of A
    # Generate a random permutation matrix
    P = torch.eye(m)[torch.randperm(m)]
    A_est = P @ torch.diag(torch.rand(m)) @ A
    # Calculate MCC
    mcc = MCC(A, A_est)
    print(f"MCC A and A_perm: {mcc:.4f}")