import math

import torch
import torch.nn.functional as F

def calculate_certainty(seg_probs):
    """Calculate the uncertainty of segmentation probability

    Args:
        seg_probs (torch.Tensor): B x C x H x W
            probability map of segmentation

    Returns:
        torch.Tensor: B x 1 x H x W
            uncertainty of input probability
    """
    top2_scores, indices = torch.topk(seg_probs, k=2, dim=1)
    res = (top2_scores[:, 0] - top2_scores[:, 1]).unsqueeze(1)
    # import pdb; pdb.set_trace()
    # res[torch.logical_and(res > 0.0, res < 0.5)] = 0.5
    # res = torch.clamp(res, 0.2, 0.8)
    res = (res - res.min()) / (res.max() - res.min())
    res = res * 0.3
    res = res + 0.5
    return res


def get_uncertain_point_coords_on_grid(uncertainty_map, num_points):
    """Find `num_points` most uncertain points from `uncertainty_map` grid.

    Args:
        uncertainty_map (Tensor): A tensor of shape (N, 1, H, W) that contains uncertainty
            values for a set of points on a regular H x W grid.
        num_points (int): The number of points P to select.

    Returns:
        point_indices (Tensor): A tensor of shape (N, P) that contains indices from
            [0, H x W) of the most uncertain points.
        point_coords (Tensor): A tensor of shape (N, P, 2) that contains [0, 1] x [0, 1] normalized
            coordinates of the most uncertain points from the H x W grid.
    """
    R, _, H, W = uncertainty_map.shape
    h_step = 1.0 / float(H)
    w_step = 1.0 / float(W)

    num_points = min(H * W, num_points)

    uncertainty_map = uncertainty_map.view(R, H * W)

    if num_points == H * W:
        point_indices = torch.arange(start=0, end=H * W, device=uncertainty_map.device).view(1, -1).repeat(R, 1)
    else:
        point_indices = torch.topk(uncertainty_map, k=num_points, dim=1, sorted=False)[1]

    point_coords = torch.zeros(R, num_points, 2, dtype=torch.float, device=uncertainty_map.device)

    point_coords[:, :, 0] = w_step / 2.0 + (point_indices % W).to(torch.float) * w_step
    point_coords[:, :, 1] = h_step / 2.0 + (point_indices // W).to(torch.float) * h_step

    # Sort indices
    sorted_values = torch.sqrt(point_coords[:, :, 0] ** 2 + point_coords[:, :, 1] ** 2)
    indices = torch.argsort(sorted_values, 1)

    for i in range(R):
        point_coords[i] = point_coords[i].gather(0, torch.stack([indices[i], indices[i]], dim=1))
        point_indices[i] = point_indices[i].gather(0, indices[i])

    return point_indices, point_coords

def point_sample(input, point_coords, **kwargs):
    """A wrapper around :function:`torch.nn.functional.grid_sample` to support 3D point_coords tensors.
    Unlike :function:`torch.nn.functional.grid_sample` it assumes `point_coords` to lie inside
    [0, 1] x [0, 1] square.

    Args:
        input (Tensor): A tensor of shape (N, C, H, W) that contains features map on a H x W grid.
        point_coords (Tensor): A tensor of shape (N, P, 2) or (N, Hgrid, Wgrid, 2) that contains
        [0, 1] x [0, 1] normalized point coordinates.

    Returns:
        output (Tensor): A tensor of shape (N, C, P) or (N, C, Hgrid, Wgrid) that contains
            features for points in `point_coords`. The features are obtained via bilinear
            interplation from `input` the same way as :function:`torch.nn.functional.grid_sample`.
    """
    add_dim = False
    if point_coords.dim() == 3:
        add_dim = True
        point_coords = point_coords.unsqueeze(2)
    output = F.grid_sample(input, 2.0 * point_coords - 1.0, **kwargs)
    if add_dim:
        output = output.squeeze(3)
    return output