import torch
import numpy as np


def sim(z_i, z_j):
    """Normalized dot product between two vectors.

    Inputs:
    - z_i: 1xD tensor.
    - z_j: 1xD tensor.
    
    Returns:
    - A scalar value that is the normalized dot product between z_i and z_j.
    """
    norm_dot_product = None
    ##############################################################################
    # TODO: Start of your code.                                                  #
    #                                                                            #
    # HINT: torch.linalg.norm might be helpful.                                  #
    ##############################################################################

    # compute normalized dot product (cosine similarity)
    # sim(z_i, z_j) = (z_i · z_j) / (||z_i|| * ||z_j||)
    norm_dot_product = torch.dot(z_i.squeeze(), z_j.squeeze()) / (torch.linalg.norm(z_i) * torch.linalg.norm(z_j))
    
    
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    
    return norm_dot_product


def simclr_loss_naive(out_left, out_right, tau):
    """Compute the contrastive loss L over a batch (naive loop version).
    
    Input:
    - out_left: NxD tensor; output of the projection head g(), left branch in SimCLR model.
    - out_right: NxD tensor; output of the projection head g(), right branch in SimCLR model.
    Each row is a z-vector for an augmented sample in the batch. The same row in out_left and out_right form a positive pair. 
    In other words, (out_left[k], out_right[k]) form a positive pair for all k=0...N-1.
    - tau: scalar value, temperature parameter that determines how fast the exponential increases.
    
    Returns:
    - A scalar value; the total loss across all positive pairs in the batch. See notebook for definition.
    """
    N = out_left.shape[0]  # total number of training examples
    
     # Concatenate out_left and out_right into a 2*N x D tensor.
    out = torch.cat([out_left, out_right], dim=0)  # [2*N, D]
    
    total_loss = 0
    for k in range(N):  # loop through each positive pair (k, k+N)
        z_k, z_k_N = out[k], out[k+N]
        
        ##############################################################################
        # TODO: Start of your code.                                                  #
        #                                                                            #
        # Hint: Compute l(k, k+N) and l(k+N, k).                                     #
        ##############################################################################

        # compute l(k, k+N): loss for positive pair (k, k+N)
        # l(i,j) = -log[ exp(sim(z_i, z_j)/tau) / sum_{m!=i} exp(sim(z_i, z_m)/tau) ]

        # numerator for l(k, k+N): similarity between positive pair
        numerator_k = torch.exp(sim(z_k, z_k_N) / tau)

        # denominator for l(k, k+N): sum of similarities with all other samples except k
        denominator_k = 0
        for i in range(2 * N):
            if i != k:
                denominator_k += torch.exp(sim(z_k, out[i]) / tau)

        # loss for l(k, k+N)
        loss_k = -torch.log(numerator_k / denominator_k)

        # compute l(k+N, k): loss for positive pair (k+N, k)
        # numerator for l(k+N, k): similarity between positive pair
        numerator_k_N = torch.exp(sim(z_k_N, z_k) / tau)

        # denominator for l(k+N, k): sum of similarities with all other samples except k+N
        denominator_k_N = 0
        for i in range(2 * N):
            if i != k + N:
                denominator_k_N += torch.exp(sim(z_k_N, out[i]) / tau)

        # loss for l(k+N, k)
        loss_k_N = -torch.log(numerator_k_N / denominator_k_N)

        # add both losses to total
        total_loss += loss_k + loss_k_N

        
        ##############################################################################
        #                               END OF YOUR CODE                             #
        ##############################################################################
    
    # In the end, we need to divide the total loss by 2N, the number of samples in the batch.
    total_loss = total_loss / (2*N)
    return total_loss


def sim_positive_pairs(out_left, out_right):
    """Normalized dot product between positive pairs.

    Inputs:
    - out_left: NxD tensor; output of the projection head g(), left branch in SimCLR model.
    - out_right: NxD tensor; output of the projection head g(), right branch in SimCLR model.
    Each row is a z-vector for an augmented sample in the batch.
    The same row in out_left and out_right form a positive pair.
    
    Returns:
    - A Nx1 tensor; each row k is the normalized dot product between out_left[k] and out_right[k].
    """
    pos_pairs = None
    
    ##############################################################################
    # TODO: Start of your code.                                                  #
    #                                                                            #
    # HINT: torch.linalg.norm might be helpful.                                  #
    ##############################################################################

    # compute element-wise dot product between corresponding rows
    # out_left and out_right are both NxD
    dot_products = (out_left * out_right).sum(dim=1)  # [N]

    # compute norms for each row
    norm_left = torch.linalg.norm(out_left, dim=1)  # [N]
    norm_right = torch.linalg.norm(out_right, dim=1)  # [N]

    # compute normalized dot product (cosine similarity) for all pairs
    pos_pairs = dot_products / (norm_left * norm_right)  # [N]

    # reshape to Nx1
    pos_pairs = pos_pairs.view(-1, 1)
    
    
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return pos_pairs


def compute_sim_matrix(out):
    """Compute a 2N x 2N matrix of normalized dot products between all pairs of augmented examples in a batch.

    Inputs:
    - out: 2N x D tensor; each row is the z-vector (output of projection head) of a single augmented example.
    There are a total of 2N augmented examples in the batch.
    
    Returns:
    - sim_matrix: 2N x 2N tensor; each element i, j in the matrix is the normalized dot product between out[i] and out[j].
    """
    sim_matrix = None
    
    ##############################################################################
    # TODO: Start of your code.                                                  #
    ##############################################################################

    # compute all pairwise dot products: out @ out.T gives a 2N x 2N matrix
    dot_product_matrix = out @ out.T  # [2N, 2N]

    # compute norms for each vector
    norms = torch.linalg.norm(out, dim=1)  # [2N]

    # create outer product of norms to get denominator matrix
    # norms.unsqueeze(1) is [2N, 1], norms.unsqueeze(0) is [1, 2N]
    # their product is [2N, 2N] where element [i,j] = norm[i] * norm[j]
    norm_matrix = norms.unsqueeze(1) * norms.unsqueeze(0)  # [2N, 2N]

    # compute normalized dot products (cosine similarity matrix)
    sim_matrix = dot_product_matrix / norm_matrix  # [2N, 2N]
    
    
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return sim_matrix


def simclr_loss_vectorized(out_left, out_right, tau, device='cuda'):
    """Compute the contrastive loss L over a batch (vectorized version). No loops are allowed.
    
    Inputs and output are the same as in simclr_loss_naive.
    """
    N = out_left.shape[0]
    
    # Concatenate out_left and out_right into a 2*N x D tensor.
    out = torch.cat([out_left, out_right], dim=0)  # [2*N, D]
    
    # Compute similarity matrix between all pairs of augmented examples in the batch.
    sim_matrix = compute_sim_matrix(out)  # [2*N, 2*N]
    
    ##############################################################################
    # TODO: Start of your code. Follow the hints.                                #
    ##############################################################################
    
    # Step 1: Use sim_matrix to compute the denominator value for all augmented samples.
    # Hint: Compute e^{sim / tau} and store into exponential, which should have shape 2N x 2N.
    exponential = torch.exp(sim_matrix / tau)  # [2N, 2N]
    
    # This binary mask zeros out terms where k=i.
    mask = (torch.ones_like(exponential, device=device) - torch.eye(2 * N, device=device)).to(device).bool()
    
    # We apply the binary mask.
    exponential = exponential.masked_select(mask).view(2 * N, -1)  # [2*N, 2*N-1]
    
    # Hint: Compute the denominator values for all augmented samples. This should be a 2N x 1 vector.
    denom = exponential.sum(dim=1, keepdim=True)  # [2N, 1]

    # Step 2: Compute similarity between positive pairs.
    # You can do this in two ways: 
    # Option 1: Extract the corresponding indices from sim_matrix. 
    # Option 2: Use sim_positive_pairs().
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    sim_pos = sim_positive_pairs(out_left, out_right)  # [N, 1]

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    # Step 3: Compute the numerator value for all augmented samples.
    numerator = None
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # for each augmented sample, we need exp(sim(z_i, z_j) / tau) where j is its positive pair
    # sim_pos contains similarities for pairs (0, N), (1, N+1), ..., (N-1, 2N-1)
    # we need to replicate this for both directions: l(k, k+N) and l(k+N, k)
    # since sim is symmetric, sim_pos works for both

    # create numerator for all 2N samples: [N positive pairs, then same N pairs in reverse]
    numerator_exp = torch.exp(sim_pos / tau)  # [N, 1]
    numerator = torch.cat([numerator_exp, numerator_exp], dim=0)  # [2N, 1]

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    # Step 4: Now that you have the numerator and denominator for all augmented samples, compute the total loss.
    loss = None
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # compute the contrastive loss: -log(numerator / denominator)
    # average over all 2N samples
    loss = -torch.log(numerator / denom)  # [2N, 1]
    loss = loss.mean()  # scalar

    
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    
    return loss


def rel_error(x,y):
    return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))