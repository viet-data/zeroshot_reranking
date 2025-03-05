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

import torch

def exp_map_sphere(mu, v, eps=1e-8):
    """
    Exponential map on the unit sphere S^{d-1}.
    
    Given a base point mu (unit vector) and a tangent vector v at mu,
    returns exp_mu(v) which is a point on the sphere.
    """
    norm_v = v.norm(p=2, dim=-1, keepdim=True)
    # Avoid division by zero:
    direction = v / (norm_v + eps)
    return torch.cos(norm_v) * mu + torch.sin(norm_v) * direction

def log_map_sphere(mu, x, eps=1e-8):
    """
    Logarithm map on the unit sphere S^{d-1}.
    
    Given a base point mu (unit vector) and a point x on the sphere,
    returns the tangent vector at mu pointing toward x.
    """
    # Compute the cosine of the angle between mu and x.
    cos_theta = (mu * x).sum(dim=-1, keepdim=True).clamp(-1.0, 1.0)
    theta = torch.acos(cos_theta)
    
    # Compute the orthogonal component: x - (mu^T x) mu
    diff = x - cos_theta * mu
    norm_diff = diff.norm(p=2, dim=-1, keepdim=True)
    
    # For very small theta, return zero to avoid division by zero.
    # Otherwise, scale the unit tangent vector by theta.
    factor = theta / (norm_diff + eps)
    # When norm_diff is zero (i.e. when x == mu) the returned vector is zero.
    v = factor * diff
    return v

def karcher_mean_sphere(points, max_iter=100, tol=1e-6):
    """
    Computes the Karcher (geodesic) mean of points on the unit sphere S^{d-1}.
    
    Args:
        points (Tensor): Tensor of shape (N, d) where each row is a unit-norm vector.
        max_iter (int): Maximum number of iterations.
        tol (float): Tolerance for convergence (if the norm of the mean update is below tol, stop).
        
    Returns:
        Tensor: The computed Karcher mean (unit vector of shape (d,)).
    """
    # Ensure points are normalized
    points = points / (points.norm(p=2, dim=1, keepdim=True) + 1e-8)
    
    # Initialize mean as the first point
    mu = points[0].clone()
    
    for i in range(max_iter):
        # Compute tangent vectors for all points at the current mean.
        # This uses the log map on the sphere.
        v = log_map_sphere(mu.unsqueeze(0).expand_as(points), points)  # (N, d)
        
        # Compute the average tangent vector
        v_mean = v.mean(dim=0, keepdim=True)  # (1, d)
        norm_v_mean = v_mean.norm(p=2)
        
        if norm_v_mean < tol:
            # Convergence: the mean update is very small.
            break
        
        # Update the mean by moving along the geodesic defined by v_mean.
        mu = exp_map_sphere(mu.unsqueeze(0), v_mean).squeeze(0)
        # Renormalize (to guard against numerical drift)
        mu = mu / (mu.norm(p=2) + 1e-8)
        
    return mu
