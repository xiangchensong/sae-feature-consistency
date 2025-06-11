"""
Defines the dictionary classes
"""

from abc import ABC, abstractmethod
import torch as t
import torch.nn as nn
import torch.nn.init as init
import einops
from utils import set_decoder_norm_to_unit_norm

class Dictionary(ABC, nn.Module):
    """
    A dictionary consists of a collection of vectors, an encoder, and a decoder.
    """

    dict_size: int  # number of features in the dictionary
    activation_dim: int  # dimension of the activation vectors

    @abstractmethod
    def encode(self, x):
        """
        Encode a vector x in the activation space.
        """
        pass

    @abstractmethod
    def decode(self, f):
        """
        Decode a dictionary vector f (i.e. a linear combination of dictionary elements)
        """
        pass

    @classmethod
    @abstractmethod
    def from_pretrained(cls, path, device=None, **kwargs) -> "Dictionary":
        """
        Load a pretrained dictionary from a file.
        """
        pass

class AutoEncoder(Dictionary, nn.Module):
    """
    A one-layer autoencoder.
    """

    def __init__(self, activation_dim, dict_size):
        super().__init__()
        self.activation_dim = activation_dim
        self.dict_size = dict_size
        self.bias = nn.Parameter(t.zeros(activation_dim))
        self.encoder = nn.Linear(activation_dim, dict_size, bias=True)
        self.decoder = nn.Linear(dict_size, activation_dim, bias=False)

        # initialize encoder and decoder weights
        w = t.randn(activation_dim, dict_size)
        ## normalize columns of w
        w = w / w.norm(dim=0, keepdim=True) * 0.1
        ## set encoder and decoder weights
        self.encoder.weight = nn.Parameter(w.clone().T)
        self.decoder.weight = nn.Parameter(w.clone())

    def encode(self, x):
        return nn.ReLU()(self.encoder(x - self.bias))

    def decode(self, f):
        return self.decoder(f) + self.bias

    def forward(self, x, output_features=False, ghost_mask=None):
        """
        Forward pass of an autoencoder.
        x : activations to be autoencoded
        output_features : if True, return the encoded features as well as the decoded x
        ghost_mask : if not None, run this autoencoder in "ghost mode" where features are masked
        """
        if ghost_mask is None:  # normal mode
            f = self.encode(x)
            x_hat = self.decode(f)
            if output_features:
                return x_hat, f
            else:
                return x_hat

        else:  # ghost mode
            f_pre = self.encoder(x - self.bias)
            f_ghost = t.exp(f_pre) * ghost_mask.to(f_pre)
            f = nn.ReLU()(f_pre)

            x_ghost = self.decoder(
                f_ghost
            )  # note that this only applies the decoder weight matrix, no bias
            x_hat = self.decode(f)
            if output_features:
                return x_hat, x_ghost, f
            else:
                return x_hat, x_ghost

    def scale_biases(self, scale: float):
        self.encoder.bias.data *= scale
        self.bias.data *= scale

    def normalize_decoder(self):
        norms = t.norm(self.decoder.weight, dim=0).to(dtype=self.decoder.weight.dtype, device=self.decoder.weight.device)

        if t.allclose(norms, t.ones_like(norms)):
            return
        print("Normalizing decoder weights")

        test_input = t.randn(10, self.activation_dim)
        initial_output = self(test_input)

        self.decoder.weight.data /= norms

        new_norms = t.norm(self.decoder.weight, dim=0)
        assert t.allclose(new_norms, t.ones_like(new_norms))

        self.encoder.weight.data *= norms[:, None]
        self.encoder.bias.data *= norms

        new_output = self(test_input)

        # Errors can be relatively large in larger SAEs due to floating point precision
        assert t.allclose(initial_output, new_output, atol=1e-4)


    @classmethod
    def from_pretrained(cls, path, dtype=t.float, device=None, normalize_decoder=True):
        """
        Load a pretrained autoencoder from a file.
        """
        state_dict = t.load(path)
        dict_size, activation_dim = state_dict["encoder.weight"].shape
        autoencoder = cls(activation_dim, dict_size)
        autoencoder.load_state_dict(state_dict)

        # This is useful for doing analysis where e.g. feature activation magnitudes are important
        # If training the SAE using the April update, the decoder weights are not normalized
        if normalize_decoder:
            autoencoder.normalize_decoder()

        if device is not None:
            autoencoder.to(dtype=dtype, device=device)

        return autoencoder


class CoherenceAutoEncoder(AutoEncoder):
    """
    A one-layer autoencoder with a coherence constraint.
    """

    def __init__(self, activation_dim, dict_size):
        super().__init__(activation_dim, dict_size)
    
    def calculate_coherence_loss(self):
        """
        Calculate the mutual coherence of the decoder weight matrix.
        
        Mutual coherence measures the maximum absolute normalized inner product
        between any two distinct columns of the matrix. Lower coherence values
        are generally desirable in sparse coding and dictionary learning.
        
        Returns:
            torch.Tensor: A scalar tensor with the mutual coherence value
        """
        # Get decoder weight matrix
        A_est = self.decoder.weight  # (activation_dim x dict_size)
        
        # Normalize columns to unit l2 norm
        col_norms = t.norm(A_est, dim=0, keepdim=True, p=2)  # (1 x dict_size)
        eps = t.finfo(A_est.dtype).eps
        A_normalized = A_est / (col_norms + eps)  # (activation_dim x dict_size)
        
        # Compute Gram matrix (matrix of inner products)
        G = t.matmul(A_normalized.T, A_normalized)  # (dict_size x dict_size)
        
        # Create mask to exclude self-correlations
        mask = t.triu(t.ones(G.shape[0], G.shape[0], device=G.device), diagonal=1)
        
        # Apply mask and find maximum absolute correlation
        G_masked = G * mask
        mutual_coherence = t.max(t.abs(G_masked))
        
        return mutual_coherence

class TopKAutoEncoder(Dictionary, nn.Module):
    """
    The top-k autoencoder architecture and initialization used in https://arxiv.org/abs/2406.04093
    NOTE: (From Adam Karvonen) There is an unmaintained implementation using Triton kernels in the topk-triton-implementation branch.
    We abandoned it as we didn't notice a significant speedup and it added complications, which are noted
    in the AutoEncoderTopK class docstring in that branch.

    With some additional effort, you can train a Top-K SAE with the Triton kernels and modify the state dict for compatibility with this class.
    Notably, the Triton kernels currently have the decoder to be stored in nn.Parameter, not nn.Linear, and the decoder weights must also
    be stored in the same shape as the encoder.
    """

    def __init__(self, activation_dim: int, dict_size: int, k: int):
        super().__init__()
        self.activation_dim = activation_dim
        self.dict_size = dict_size

        assert isinstance(k, int) and k > 0, f"k={k} must be a positive integer"
        self.register_buffer("k", t.tensor(k, dtype=t.int))
        self.register_buffer("threshold", t.tensor(-1.0, dtype=t.float32))

        self.decoder = nn.Linear(dict_size, activation_dim, bias=False)
        self.decoder.weight.data = set_decoder_norm_to_unit_norm(
            self.decoder.weight, activation_dim, dict_size
        )

        self.encoder = nn.Linear(activation_dim, dict_size)
        self.encoder.weight.data = self.decoder.weight.T.clone()
        self.encoder.bias.data.zero_()

        self.b_dec = nn.Parameter(t.zeros(activation_dim))

    def encode(self, x: t.Tensor, return_topk: bool = False, use_threshold: bool = False):
        post_relu_feat_acts_BF = nn.functional.relu(self.encoder(x - self.b_dec))

        if use_threshold:
            encoded_acts_BF = post_relu_feat_acts_BF * (post_relu_feat_acts_BF > self.threshold)
            if return_topk:
                post_topk = post_relu_feat_acts_BF.topk(self.k, sorted=False, dim=-1)
                return encoded_acts_BF, post_topk.values, post_topk.indices, post_relu_feat_acts_BF
            else:
                return encoded_acts_BF

        post_topk = post_relu_feat_acts_BF.topk(self.k, sorted=False, dim=-1)

        # We can't split immediately due to nnsight
        tops_acts_BK = post_topk.values
        top_indices_BK = post_topk.indices

        buffer_BF = t.zeros_like(post_relu_feat_acts_BF)
        encoded_acts_BF = buffer_BF.scatter_(dim=-1, index=top_indices_BK, src=tops_acts_BK)

        if return_topk:
            return encoded_acts_BF, tops_acts_BK, top_indices_BK, post_relu_feat_acts_BF
        else:
            return encoded_acts_BF

    def decode(self, x: t.Tensor) -> t.Tensor:
        return self.decoder(x) + self.b_dec

    def forward(self, x: t.Tensor, output_features: bool = False):
        encoded_acts_BF = self.encode(x)
        x_hat_BD = self.decode(encoded_acts_BF)
        if not output_features:
            return x_hat_BD
        else:
            return x_hat_BD, encoded_acts_BF

    def scale_biases(self, scale: float):
        self.encoder.bias.data *= scale
        self.b_dec.data *= scale
        if self.threshold >= 0:
            self.threshold *= scale

    def from_pretrained(path, k= None, device=None):
        """
        Load a pretrained autoencoder from a file.
        """
        state_dict = t.load(path)
        dict_size, activation_dim = state_dict["encoder.weight"].shape

        if k is None:
            k = state_dict["k"].item()
        elif "k" in state_dict and k != state_dict["k"].item():
            raise ValueError(f"k={k} != {state_dict['k'].item()}=state_dict['k']")

        autoencoder = TopKAutoEncoder(activation_dim, dict_size, k)
        autoencoder.load_state_dict(state_dict)
        if device is not None:
            autoencoder.to(device)
        return autoencoder

class CoherenceTopKAutoEncoder(TopKAutoEncoder):
    """
    A one-layer autoencoder with a coherence constraint.
    """

    def __init__(self, activation_dim, dict_size, k):
        super().__init__(activation_dim, dict_size, k)
    
    def calculate_coherence_loss(self):
        """
        Calculate the mutual coherence of the decoder weight matrix.
        
        Mutual coherence measures the maximum absolute normalized inner product
        between any two distinct columns of the matrix. Lower coherence values
        are generally desirable in sparse coding and dictionary learning.
        
        Returns:
            torch.Tensor: A scalar tensor with the mutual coherence value
        """
        # Get decoder weight matrix
        A_est = self.decoder.weight  # (activation_dim x dict_size)
        
        # Normalize columns to unit l2 norm
        col_norms = t.norm(A_est, dim=0, keepdim=True, p=2)  # (1 x dict_size)
        eps = t.finfo(A_est.dtype).eps
        A_normalized = A_est / (col_norms + eps)  # (activation_dim x dict_size)
        
        # Compute Gram matrix (matrix of inner products)
        G = t.matmul(A_normalized.T, A_normalized)  # (dict_size x dict_size)
        
        # Create mask to exclude self-correlations
        mask = t.triu(t.ones(G.shape[0], G.shape[0], device=G.device), diagonal=1)
        
        # Apply mask and find maximum absolute correlation
        G_masked = G * mask
        mutual_coherence = t.max(t.abs(G_masked))
        
        return mutual_coherence