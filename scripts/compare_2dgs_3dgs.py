"""
Panorama Gaussian Splatting implemented in python with a few lines of code

This code implements the renderer of the paper "GS-LiDAR: Generating Realistic LiDAR Point Clouds with Panoramic Gaussian Splatting".

paper: https://arxiv.org/pdf/2501.13971

homepage: https://github.com/LogosIntelligenceLab/GS-LiDAR

The cuda code is efficient and good, but it would be more readable with a pure pytorch/python code so readers can have better understanding without needing looking into the cuda implementation. Reader can also implement it with other preferred programming language.

This code is built upon many great repos:

torch-splatting: https://github.com/hbb1/torch-splatting

3DGS: https://github.com/graphdeco-inria/gaussian-splatting

2DGS: https://github.com/hbb1/2d-gaussian-splatting

gsplat: https://github.com/nerfstudio-project/gsplat
"""

import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from utils.general_utils import seed_everything


def build_rotation(r):
    norm = torch.sqrt(r[:, 0] * r[:, 0] + r[:, 1] * r[:, 1] + r[:, 2] * r[:, 2] + r[:, 3] * r[:, 3])

    q = r / norm[:, None]

    R = torch.zeros((q.size(0), 3, 3), device='cuda')

    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    R[:, 0, 0] = 1 - 2 * (y * y + z * z)
    R[:, 0, 1] = 2 * (x * y - r * z)
    R[:, 0, 2] = 2 * (x * z + r * y)
    R[:, 1, 0] = 2 * (x * y + r * z)
    R[:, 1, 1] = 1 - 2 * (x * x + z * z)
    R[:, 1, 2] = 2 * (y * z - r * x)
    R[:, 2, 0] = 2 * (x * z - r * y)
    R[:, 2, 1] = 2 * (y * z + r * x)
    R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def build_scaling_rotation(s, r):
    L = torch.zeros((s.shape[0], 3, 3), dtype=torch.float, device="cuda")
    R = build_rotation(r)

    L[:, 0, 0] = s[:, 0]
    L[:, 1, 1] = s[:, 1]
    L[:, 2, 2] = s[:, 2]

    L = R @ L
    return L


def getProjectionMatrix(znear, zfar, fovX, fovY):
    import math
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = torch.zeros(4, 4)

    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P


def focal2fov(focal, pixels):
    import math
    return 2 * math.atan(pixels / (2 * focal))


def homogeneous(points):
    """
    homogeneous points
    :param points: [..., 3]
    """
    return torch.cat([points, torch.ones_like(points[..., :1])], dim=-1)


def homogeneous_vec(vec):
    """
    homogeneous points
    :param points: [..., 3]
    """
    return torch.cat([vec, torch.zeros_like(vec[..., :1])], dim=-1)


"""## Surface splatting (2D Gaussian splatting)"""


def build_panorama_covariance_2d(
        mean3d, cov3d, viewmatrix, vfov, hfov, width, height
):
    VFOV_max = torch.pi / 2 - vfov[0] * torch.pi / 180
    VFOV_min = torch.pi / 2 - vfov[1] * torch.pi / 180
    HFOV_max = hfov[1] * torch.pi / 180
    HFOV_min = hfov[0] * torch.pi / 180

    t = (mean3d @ viewmatrix[:3, :3]) + viewmatrix[-1:, :3]
    tx = t[:, 0]
    ty = t[:, 1]
    tz = t[:, 2]
    r_xz = torch.sqrt(tx * tx + tz * tz)
    r_xyz = torch.sqrt(tx * tx + ty * ty + tz * tz)
    xy = tx * ty
    yz = ty * tz
    Wpi = width / (HFOV_max - HFOV_min)
    Hrange = height / (VFOV_max - VFOV_min)

    # perspective transform is not affine so we approximate with first-order taylor expansion
    # notice that we multiply by the intrinsic so that the variance is at the sceen space
    J = torch.zeros(mean3d.shape[0], 3, 3).to(mean3d)
    J[..., 1, 0] = -Hrange * xy / (r_xz * r_xyz * r_xyz)
    J[..., 1, 1] = Hrange * r_xz / (r_xyz * r_xyz)
    J[..., 1, 2] = -Hrange * yz / (r_xz * r_xyz * r_xyz)
    J[..., 0, 0] = Wpi * tz / (r_xz * r_xz)
    J[..., 0, 2] = -Wpi * tx / (r_xz * r_xz)
    W = viewmatrix[:3, :3].T  # transpose to correct viewmatrix
    cov2d = J @ W @ cov3d @ W.T @ J.permute(0, 2, 1)

    # add low pass filter here according to E.q. 32
    filter = torch.eye(2, 2).to(cov2d) * 0.0
    return cov2d[:, :2, :2] + filter[None]


# Surface splatting (2D Gaussian Splatting)
def setup(means3D, scales, quats, opacities, colors, viewmat, projmat, vfov, hfov):
    VFOV_max = torch.pi / 2 - vfov[0] * torch.pi / 180
    VFOV_min = torch.pi / 2 - vfov[1] * torch.pi / 180
    HFOV_max = hfov[1] * torch.pi / 180
    HFOV_min = hfov[0] * torch.pi / 180
    intrins = projmat[:3, :3].transpose(0, 1)
    W, H = (intrins[0, -1] * 2).long(), (intrins[1, -1] * 2).long()
    W, H = W.item(), H.item()

    rotations = build_scaling_rotation(scales, quats).permute(0, 2, 1)

    # 1. Viewing transform
    p_view = (means3D @ viewmat[:3, :3]) + viewmat[-1:, :3]
    uv_view = (rotations @ viewmat[:3, :3])
    M = torch.cat([homogeneous_vec(uv_view[:, :2, :]), homogeneous(p_view.unsqueeze(1))], dim=1)

    T = M[..., :3, :3]

    x = p_view[:, [0]]
    y = p_view[:, [1]]
    z = p_view[:, [2]]
    phi = torch.atan2(x, z)
    theta = torch.atan2(torch.sqrt(x ** 2 + z ** 2), -y)
    r = torch.sqrt(x ** 2 + y ** 2 + z ** 2)

    point_image = torch.cat([(phi - HFOV_min) * W / (HFOV_max - HFOV_min),
                             (theta - VFOV_min) * H / (VFOV_max - VFOV_min),
                             torch.ones_like(theta)], dim=-1)

    depth = r[:, 0]

    # 2. Compute AABB
    sample_num = 12
    sample_theta = 2 * torch.pi * torch.arange(0, 1, 1 / sample_num)
    sigma = 1.0
    sample_points = torch.cat([sigma * torch.sin(sample_theta).unsqueeze(1),
                               sigma * torch.cos(sample_theta).unsqueeze(1),
                               torch.ones_like(sample_theta).unsqueeze(1)], dim=1).to(T)
    sample_point_image = sample_points @ T
    sample_phi = torch.atan2(sample_point_image[..., 0], sample_point_image[..., 2])
    sample_theta = torch.atan2(torch.sqrt(sample_point_image[..., 0] ** 2 + sample_point_image[..., 2] ** 2), -sample_point_image[..., 1])
    radii = torch.cat([
        (phi - sample_phi.min(dim=-1, keepdim=True)[0]) * W / (HFOV_max - HFOV_min),
        (sample_phi.max(dim=-1, keepdim=True)[0] - phi) * W / (HFOV_max - HFOV_min),
        (theta - sample_theta.min(dim=-1, keepdim=True)[0]) * H / (VFOV_max - VFOV_min),
        (sample_theta.max(dim=-1, keepdim=True)[0] - theta) * H / (VFOV_max - VFOV_min)
    ], dim=-1)

    center = point_image

    # 3. Perform Sorting
    index = depth.sort()[1]
    T = T[index]
    colors = colors[index]
    center = center[index]
    depth = depth[index]
    radii = radii[index]
    opacities = opacities[index]
    return T, colors, opacities, center, depth, radii


def surface_splatting(means3D, scales, quats, colors, opacities, intrins, viewmat, projmat):
    vfov = (-20, 20)
    hfov = (-90, 90)

    VFOV_max = torch.pi / 2 - vfov[0] * torch.pi / 180
    VFOV_min = torch.pi / 2 - vfov[1] * torch.pi / 180
    HFOV_max = hfov[1] * torch.pi / 180
    HFOV_min = hfov[0] * torch.pi / 180

    # Rasterization setup
    projmat = torch.zeros(4, 4).cuda()
    projmat[:3, :3] = intrins
    projmat[-1, -2] = 1.0
    projmat = projmat.T
    T, colors, opacities, center, depth, radii = setup(means3D, scales, quats, opacities, colors, viewmat, projmat, vfov, hfov)

    # Rasterization
    # 1. Generate pixels
    W, H = (intrins[0, -1] * 2).long(), (intrins[1, -1] * 2).long()
    W, H = W.item(), H.item()
    pix = torch.stack(torch.meshgrid(torch.arange(W),
                                     torch.arange(H), indexing='xy'), dim=-1).to('cuda')

    # 2. Compute ray splat intersection
    x = pix.reshape(-1, 1, 2)[..., :1]
    y = pix.reshape(-1, 1, 2)[..., 1:]

    phi = x * (HFOV_max - HFOV_min) / W + HFOV_min
    theta = y * (VFOV_max - VFOV_min) / H + VFOV_min

    k = torch.cos(phi) * T[None][..., 0] - torch.sin(phi) * T[None][..., 2]
    l = torch.cos(theta) * torch.sin(phi) * T[None][..., 0] + torch.sin(theta) * T[None][..., 1] + torch.cos(theta) * torch.cos(phi) * T[None][..., 2]
    points = torch.cross(k, l, dim=-1)
    s = points[..., :2] / points[..., -1:]

    # 3. add low pass filter
    # when a point (2D Gaussian) viewed from a far distance or from a slended angle
    # the 2D Gaussian will falls between pixels and no fragment is used to rasterize the Gaussian
    # so we should add a low pass filter to handle such aliasing.
    dist3d = (s * s).sum(dim=-1)
    filtersze = np.sqrt(2) / 2
    dist2d = (1 / filtersze) ** 2 * (torch.cat([x, y], dim=-1) - center[None, :, :2]).norm(dim=-1) ** 2
    # min of dist2 is equal to max of Gaussian exp(-0.5 * dist2)
    dist2 = dist3d  # torch.min(dist3d, dist2d)
    depth_acc = ((homogeneous(s) * T[None, ..., 0]).sum(dim=-1) * torch.sin(theta)[..., 0] * torch.sin(phi)[..., 0]
                 - (homogeneous(s) * T[None, ..., 1]).sum(dim=-1) * torch.cos(theta)[..., 0]
                 + (homogeneous(s) * T[None, ..., 2]).sum(dim=-1) * torch.sin(theta)[..., 0] * torch.cos(phi)[..., 0])

    # 4. accumulate 2D gaussians through alpha blending
    image, depthmap = alpha_blending_with_gaussians(dist2, colors, opacities, depth_acc, H, W)
    return image, depthmap, center, radii, dist2


"""## Volume splatting (3D Gaussian Splatting)"""


def build_covariance_3d(s, r):
    L = build_scaling_rotation(s, r).permute(0, 2, 1)
    actual_covariance = L @ L.transpose(1, 2)
    return actual_covariance


def get_radius(cov2d):
    det = cov2d[:, 0, 0] * cov2d[:, 1, 1] - cov2d[:, 0, 1] * cov2d[:, 1, 0]
    mid = 0.5 * (cov2d[:, 0, 0] + cov2d[:, 1, 1])
    lambda1 = mid + torch.sqrt((mid ** 2 - det).clip(min=0.1))
    lambda2 = mid - torch.sqrt((mid ** 2 - det).clip(min=0.1))
    return 3.0 * torch.sqrt(torch.max(lambda1, lambda2)).ceil()


def volume_splatting(means3D, scales, quats, colors, opacities, intrins, viewmat, projmat):
    vfov = (-20, 20)
    hfov = (-90, 90)

    VFOV_max = torch.pi / 2 - vfov[0] * torch.pi / 180
    VFOV_min = torch.pi / 2 - vfov[1] * torch.pi / 180
    HFOV_max = hfov[1] * torch.pi / 180
    HFOV_min = hfov[0] * torch.pi / 180
    W, H = (intrins[0, -1] * 2).long(), (intrins[1, -1] * 2).long()
    W, H = W.item(), H.item()

    p_view = (means3D @ viewmat[:3, :3]) + viewmat[-1:, :3]

    x = p_view[:, [0]]
    y = p_view[:, [1]]
    z = p_view[:, [2]]
    phi = torch.atan2(x, z)
    theta = torch.atan2(torch.sqrt(x ** 2 + z ** 2), -y)
    r = torch.sqrt(x ** 2 + y ** 2 + z ** 2)

    depths = r[:, 0]
    means2D = torch.cat([(phi - HFOV_min) * W / (HFOV_max - HFOV_min),
                         (theta - VFOV_min) * H / (VFOV_max - VFOV_min)], dim=-1)
    cov3d = build_covariance_3d(scales, quats)

    cov2d = build_panorama_covariance_2d(means3D, cov3d, viewmat, vfov, hfov, W, H)
    radii = get_radius(cov2d)

    # Rasterization
    # generate pixels
    pix = torch.stack(torch.meshgrid(torch.arange(W), torch.arange(H), indexing='xy'), dim=-1).to('cuda').flatten(0, -2)
    sorted_conic = cov2d.inverse()  # inverse of variance
    dx = (pix[:, None, :] - means2D[None, :])  # B P 2
    dist2 = dx[:, :, 0] ** 2 * sorted_conic[:, 0, 0] + dx[:, :, 1] ** 2 * sorted_conic[:, 1, 1] + dx[:, :, 0] * dx[:, :, 1] * sorted_conic[:, 0, 1] + dx[:, :, 0] * dx[:, :, 1] * sorted_conic[:, 1, 0]
    depth_acc = depths[None].expand_as(dist2)

    image, depthmap = alpha_blending_with_gaussians(dist2, colors, opacities, depth_acc, H, W)
    return image, depthmap, means2D, radii, dist2


"""## Rendering utils"""


def alpha_blending(alpha, colors):
    T = torch.cat([torch.ones_like(alpha[-1:]), (1 - alpha).cumprod(dim=0)[:-1]], dim=0)
    image = (T * alpha * colors).sum(dim=0).reshape(-1, colors.shape[-1])
    alphamap = (T * alpha).sum(dim=0).reshape(-1, 1)
    return image, alphamap


def alpha_blending_with_gaussians(dist2, colors, opacities, depth_acc, H, W):
    colors = colors.reshape(-1, 1, colors.shape[-1])
    depth_acc = depth_acc.T[..., None]
    depth_acc = depth_acc.repeat(1, 1, 1)

    # evaluate gaussians
    # just for visualization, the actual cut off can be 3 sigma!
    cutoff = 1 ** 2
    dist2 = dist2.T
    gaussians = torch.exp(-0.5 * dist2) * (dist2 < cutoff)
    gaussians = gaussians[..., None]
    alpha = opacities.unsqueeze(1) * gaussians

    # accumulate gaussians
    image, _ = alpha_blending(alpha, colors)
    depthmap, alphamap = alpha_blending(alpha, depth_acc)
    # depthmap = depthmap / alphamap
    depthmap = torch.nan_to_num(depthmap, 0, 0)
    return image.reshape(H, W, -1), depthmap.reshape(H, W, -1)


"""## Utils for inputs and cameras"""


def get_inputs(num_points=8):
    length = 1.5
    x = np.linspace(-1, 1, num_points) * length
    y = np.linspace(-1, 1, num_points) * length
    x, y = np.meshgrid(x, y)
    means3D = torch.from_numpy(np.stack([x, y, 0 * np.random.rand(*x.shape)], axis=-1).reshape(-1, 3)).cuda().float()
    quats = torch.zeros(1, 4).repeat(len(means3D), 1).cuda()
    quat = torch.randn(4)
    quat = quat / quat.norm()
    quats[..., :] = quat
    rot = build_rotation(quat[None])[0]
    means3D = means3D @ rot.T
    scale = length / (num_points - 1)
    scales = torch.zeros(1, 3).repeat(len(means3D), 1).fill_(scale).cuda()
    return means3D, scales, quats


def get_cameras():
    width, height = 2500, 600  # 512.0 * 2, 512.0
    intrins = torch.tensor([[711.1111, 0.0000, width / 2, 0.0000],
                            [0.0000, 711.1111, height / 2, 0.0000],
                            [0.0000, 0.0000, 1.0000, 0.0000],
                            [0.0000, 0.0000, 0.0000, 1.0000]]).cuda()

    factor = 1
    c2w = torch.tensor([[-8.6086e-01, 3.7950e-01, -3.3896e-01, 6.7791e-01 * factor],
                        [5.0884e-01, 6.4205e-01, -5.7346e-01, 1.1469e+00 * factor],
                        [1.0934e-08, -6.6614e-01, -7.4583e-01, 1.4917e+00 * factor],
                        [0.0000e+00, 0.0000e+00, 0.0000e+00, 1.0000e+00]]).cuda()

    focal_x, focal_y = intrins[0, 0], intrins[1, 1]
    viewmat = torch.linalg.inv(c2w).permute(1, 0)
    FoVx = focal2fov(focal_x, width)
    FoVy = focal2fov(focal_y, height)
    projmat = getProjectionMatrix(znear=0.2, zfar=1000, fovX=FoVx, fovY=FoVy).transpose(0, 1).cuda()
    projmat = viewmat @ projmat
    return intrins, viewmat, projmat, height, width


if __name__ == '__main__':
    ## Visualization of the 2DGS v.s 3DGS
    save_path = "eval_output/compare_2dgs_3dgs"
    os.makedirs(save_path, exist_ok=True)
    ################################################## Case 1 ##################################################
    seed_everything(2023)
    # Make inputs
    num_points1 = 8
    all_points = num_points1 ** 2
    means3D, scales, quats = get_inputs(num_points=num_points1)
    scales[:, -1] = 0e-6
    intrins, viewmat, projmat, height, width = get_cameras()
    intrins = intrins[:3, :3]
    colors = matplotlib.colormaps['Accent'](np.random.randint(1, all_points, all_points) / all_points)[..., :3]
    colors = torch.from_numpy(colors).cuda()

    opacity = torch.ones_like(means3D[:, :1])
    image1, depthmap1, center1, radii1, dist1 = surface_splatting(means3D, scales, quats, colors, opacity, intrins, viewmat, projmat)
    image2, depthmap2, center2, radii2, dist2 = volume_splatting(means3D, scales, quats, colors, opacity, intrins, viewmat, projmat)

    point_image = center1.cpu().detach().numpy()
    half_extend = radii1.cpu().numpy()  # only show one sigma
    lb = np.floor(point_image[:, :2] - half_extend[:, [0, 2]])
    hw = np.ceil(half_extend[..., [0, 2]] + half_extend[:, [1, 3]])

    from matplotlib.patches import Rectangle

    plt.figure()
    fig, ax = plt.subplots(1)
    ax.imshow(image1.cpu().numpy())
    # visualize AABB
    for k in range(len(half_extend)):
        ax.add_patch(Rectangle(lb[k], hw[k, 0], hw[k, 1], facecolor='none',
                               linewidth=0.1, edgecolor='white'))
    plt.savefig(os.path.join(save_path, 'case1_bbox.png'), dpi=400)

    from torchvision.utils import save_image, make_grid
    from utils.general_utils import visualize_depth

    img1 = image1.permute(2, 0, 1)
    img2 = image2.permute(2, 0, 1)

    img3 = visualize_depth(depthmap1.permute(2, 0, 1), 1.0, 6.0)
    img4 = visualize_depth(depthmap2.permute(2, 0, 1), 1.0, 6.0)

    grid = [img1, img2, img3, img4]
    grid = make_grid(grid, nrow=2)
    save_image(grid, os.path.join(save_path, 'case1.png'))

    ################################################## Case 2 ##################################################
    seed_everything(2023)
    # reduce num of points to give a close look
    num_points2 = 2
    means3D, scales, quats = get_inputs(num_points=num_points2)
    scales[:, -1] = 0e-6
    colors = torch.cat([colors[:num_points2, :], colors[num_points1:num_points1 + num_points2, :]], dim=0)

    opacity = torch.ones_like(means3D[:, :1])

    image1, depthmap1, center1, radii1, dist1 = surface_splatting(means3D, scales, quats, colors, opacity, intrins, viewmat, projmat)
    image2, depthmap2, center2, radii2, dist2 = volume_splatting(means3D, scales, quats, colors, opacity, intrins, viewmat, projmat)

    point_image = center1.cpu().detach().numpy()
    half_extend = radii1.cpu().numpy()
    lb = np.floor(point_image[:, :2] - half_extend[:, [0, 2]])
    hw = np.ceil(half_extend[..., [0, 2]] + half_extend[:, [1, 3]])

    plt.figure()
    fig, ax = plt.subplots(1)
    ax.imshow(image1.cpu().numpy())
    # visualize AABB
    for k in range(len(half_extend)):
        ax.add_patch(Rectangle(lb[k], hw[k, 0], hw[k, 1], facecolor='none',
                               linewidth=1, edgecolor='white'))
    plt.savefig(os.path.join(save_path, 'case2_bbox.png'))

    img1 = image1.permute(2, 0, 1)
    img2 = image2.permute(2, 0, 1)

    img3 = visualize_depth(depthmap1.permute(2, 0, 1), 1.0, 6.0)
    img4 = visualize_depth(depthmap2.permute(2, 0, 1), 1.0, 6.0)

    grid = [img1, img2, img3, img4]
    grid = make_grid(grid, nrow=2)
    save_image(grid, os.path.join(save_path, 'case2.png'), dpi=400)

    """you can see that the flatten 3d Gaussian has perspective distortion and the depth is constant within a splat."""
