import torch
import torch.nn.functional as F

def slerp(val, low, high):
    """
    Spherical linear interpolation between low and high.
    Args:
        val (float): interpolation value between 0 and 1
        low (Tensor): starting vector
        high (Tensor): ending vector
    Returns:
        Tensor: interpolated vector
    """
    low_norm = F.normalize(low, p=2, dim=-1)
    high_norm = F.normalize(high, p=2, dim=-1)
    omega = torch.acos(torch.clamp((low_norm * high_norm).sum(-1), -1, 1))
    so = torch.sin(omega)
    if torch.any(so == 0):
        # If the vectors are too close, return linear interpolation
        return (1.0 - val) * low + val * high
    return (torch.sin((1.0 - val) * omega) / so).unsqueeze(-1) * low + \
           (torch.sin(val * omega) / so).unsqueeze(-1) * high

def merge_vectors_slerp(tensor, val=0.1):
    """
    Merge vectors along dimension 2 using slerp.
    Args:
        tensor (Tensor): input tensor of shape (1, 8, 5, 128)
        val (float): interpolation value between 0 and 1
    Returns:
        Tensor: tensor with merged vectors along dimension 2
    """
    val = 0.5
    # Initialize a list to hold the merged vectors
    merged_vectors = []

    s1, s2 = tensor.size(1), tensor.size(3)
    
    # Iterate over the first two dimensions
    for i in range(tensor.size(0)):
        for j in range(tensor.size(1)):
            # Select the vectors along dimension 2
            vectors = tensor[i, j, :, :]
            # Initialize the merged vector with the first vector
            merged_vector = vectors[0]
            # Apply slerp sequentially to merge the vectors
            for k in range(1, vectors.size(0)):
                merged_vector = slerp(val, merged_vector, vectors[k])
            merged_vectors.append(merged_vector)
    
    # Stack the merged vectors to form the output tensor
    merged_tensor = torch.stack(merged_vectors).view(1, s1, 1, s2)
    return merged_tensor
