import torch
import torch.nn.functional as F


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

import torch

def slerp(v0, v1, t, eps=1e-8):
    """
    Spherical linear interpolation (slerp) between two vectors v0 and v1.
    
    Args:
        v0 (Tensor): starting vector (or batch of vectors) of shape (..., v)
        v1 (Tensor): ending vector (or batch of vectors) of same shape as v0.
        t (float or Tensor): interpolation parameter (0.0 -> v0, 1.0 -> v1).
        eps (float): small constant to avoid division by zero.
        
    Returns:
        Tensor: interpolated vector(s), same shape as input.
    """
    # Normalize the input vectors (ensure they lie on the unit sphere)
    v0_norm = v0 / (v0.norm(p=2, dim=-1, keepdim=True) + eps)
    v1_norm = v1 / (v1.norm(p=2, dim=-1, keepdim=True) + eps)
    
    # Compute the cosine of the angle between them and clamp for safety.
    dot = (v0_norm * v1_norm).sum(dim=-1, keepdim=True)
    dot = torch.clamp(dot, -1.0, 1.0)
    
    # Angle between v0 and v1.
    theta = torch.acos(dot)
    sin_theta = torch.sin(theta)
    
    # If theta is very small, use linear interpolation (to avoid division by zero).
    factor0 = torch.sin((1 - t) * theta) / (sin_theta + eps)
    factor1 = torch.sin(t * theta) / (sin_theta + eps)
    
    return factor0 * v0_norm + factor1 * v1_norm

def merge_vectors_slerp(vectors):
    """
    Merges a list (tensor) of vectors (shape: [n, v]) by iteratively slerping them.
    
    Args:
        vectors (Tensor): tensor of shape (n, v) where n is the number of vectors.
        
    Returns:
        Tensor: merged vector of shape (v,), approximately representing the spherical mean.
    """
    n = vectors.shape[0]
    if n == 0:
        raise ValueError("Input tensor must contain at least one vector.")
    
    # Start with the first vector, normalized
    merged = vectors[0] / (vectors[0].norm(p=2) + 1e-8)
    
    # Iteratively merge each vector into the running merged result.
    for i in range(1, n):
        current = vectors[i] / (vectors[i].norm(p=2) + 1e-8)
        # Choose an interpolation parameter; here we use t = 1/(i+1) so that later vectors
        # have less influence, but you may change this weighting.
        t = 1.0 / (i + 1)
        merged = slerp(merged, current, t)
    
    # Optionally, re-normalize the final merged vector.
    merged = merged / (merged.norm(p=2) + 1e-8)
    return merged
