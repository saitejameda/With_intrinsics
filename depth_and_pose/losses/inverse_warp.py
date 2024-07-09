from __future__ import division
import torch
import torch.nn.functional as F
from kornia.geometry.depth import depth_to_3d
from vidar.vidar.geometry.camera_ucm import UCMCamera
import numpy as np 
import cv2

def save_tensor_as_image(tensor, file_path):
    # Convert tensor to numpy array
    tensor_np = tensor.cpu().detach().numpy()
    
    # Normalize if needed
    tensor_np = (tensor_np - tensor_np.min()) / (tensor_np.max() - tensor_np.min())
    
    # Scale to [0, 255]
    tensor_np *= 255
    
    # Convert to uint8
    tensor_np = tensor_np.astype(np.uint8)
    
    # Save as image
    cv2.imwrite(file_path, tensor_np)
    
def euler2mat(angle):
    """Convert euler angles to rotation matrix.
     Reference: https://github.com/pulkitag/pycaffe-utils/blob/master/rot_utils.py#L174
    Args:
        angle: rotation angle along 3 axis (in radians) -- size = [B, 3]
    Returns:
        Rotation matrix corresponding to the euler angles -- size = [B, 3, 3]
    """
    B = angle.size(0)
    x, y, z = angle[:, 0], angle[:, 1], angle[:, 2]

    cosz = torch.cos(z)
    sinz = torch.sin(z)

    zeros = z.detach()*0
    ones = zeros.detach()+1
    zmat = torch.stack([cosz, -sinz, zeros,
                        sinz,  cosz, zeros,
                        zeros, zeros,  ones], dim=1).reshape(B, 3, 3)

    cosy = torch.cos(y)
    siny = torch.sin(y)

    ymat = torch.stack([cosy, zeros,  siny,
                        zeros,  ones, zeros,
                        -siny, zeros,  cosy], dim=1).reshape(B, 3, 3)

    cosx = torch.cos(x)
    sinx = torch.sin(x)

    xmat = torch.stack([ones, zeros, zeros,
                        zeros,  cosx, -sinx,
                        zeros,  sinx,  cosx], dim=1).reshape(B, 3, 3)

    rotMat = xmat @ ymat @ zmat
    return rotMat


def quat2mat(quat):
    """Convert quaternion coefficients to rotation matrix.
    Args:
        quat: first three coeff of quaternion of rotation. fourht is then computed to have a norm of 1 -- size = [B, 3]
    Returns:
        Rotation matrix corresponding to the quaternion -- size = [B, 3, 3]
    """
    norm_quat = torch.cat([quat[:, :1].detach()*0 + 1, quat], dim=1)
    norm_quat = norm_quat/norm_quat.norm(p=2, dim=1, keepdim=True)
    w, x, y, z = norm_quat[:, 0], norm_quat[:,
                                            1], norm_quat[:, 2], norm_quat[:, 3]

    B = quat.size(0)

    w2, x2, y2, z2 = w.pow(2), x.pow(2), y.pow(2), z.pow(2)
    wx, wy, wz = w*x, w*y, w*z
    xy, xz, yz = x*y, x*z, y*z

    rotMat = torch.stack([w2 + x2 - y2 - z2, 2*xy - 2*wz, 2*wy + 2*xz,
                          2*wz + 2*xy, w2 - x2 + y2 - z2, 2*yz - 2*wx,
                          2*xz - 2*wy, 2*wx + 2*yz, w2 - x2 - y2 + z2], dim=1).reshape(B, 3, 3)
    return rotMat

def intrinsics_mat(original_tensor,b):
    
    reshaped_tensor = torch.zeros((b, 3, 3))

    # Set the first two elements of the diagonal
    reshaped_tensor[:, 0, 0] = original_tensor[:, 0]
    reshaped_tensor[:, 1, 1] = original_tensor[:, 1]

    # Set the first two elements of the last column
    reshaped_tensor[:, 0, 2] = original_tensor[:, 2]
    reshaped_tensor[:, 1, 2] = original_tensor[:, 3]

    # Set 1 as the last element of the last column
    reshaped_tensor[:, 2, 2] = 1
    
    return reshaped_tensor

    

def pose_vec2mat(vec, rotation_mode='euler'):
    """
    Convert 6DoF parameters to transformation matrix.
    Args:s
        vec: 6DoF parameters in the order of tx, ty, tz, rx, ry, rz -- [B, 6]
    Returns:
        A transformation matrix -- [B, 3, 4]
    """
    translation = vec[:, :3].unsqueeze(-1)  # [B, 3, 1]
    rot = vec[:, 3:]
    if rotation_mode == 'euler':
        rot_mat = euler2mat(rot)  # [B, 3, 3]
    elif rotation_mode == 'quat':
        rot_mat = quat2mat(rot)  # [B, 3, 3]
    transform_mat = torch.cat([rot_mat, translation], dim=2)  # [B, 3, 4]
    return transform_mat

def inverse_warp_w1(img, depth, ref_depth, pose, intrinsics, padding_mode='zeros',hparams=None):
    """
    Inverse warp a source image to the target image plane.
    Args:
        img: the source image (where to sample pixels) -- [B, 3, H, W]
        depth: depth map of the target image -- [B, 1, H, W]
        ref_depth: the source depth map (where to sample depth) -- [B, 1, H, W] 
        pose: 6DoF pose parameters from target to source -- [B, 6]
        intrinsics: camera intrinsic matrix -- [B, 3, 3]
    Returns:
        projected_img: Source image warped to the target image plane
        projected_depth: sampled depth from source image  
        computed_depth: computed depth of source image using the target depth
    """
    B, _, H, W = img.size()
    #print('pose',pose.shape)

    K = intrinsics_mat(intrinsics,B)
    T = pose_vec2mat(pose)  # [B,3,4]
    #print(T)
    camera_ucm = UCMCamera(intrinsics, Tcw=T)
    P = torch.matmul(K.to(T.device), T)[:, :3, :]
    world_points = camera_ucm.reconstruct(depth = depth,frame='c') # B 3 H W
    #print('world_points',world_points.shape)
    #world_points = depth_to_3d(depth, intrinsics) # B 3 H W
    world_points_cam = torch.cat([world_points, torch.ones(B,1,H,W).type_as(img)], 1)
    cam_points = torch.matmul(P, world_points_cam.view(B, 4, -1))

    # pix_coords = cam_points[:, :2, :] / (cam_points[:, 2, :].unsqueeze(1) + 1e-7)
    # pix_coords = pix_coords.view(B, 2, H, W)
    # pix_coords = pix_coords.permute(0, 2, 3, 1)
    # pix_coords[..., 0] /= W - 1
    # pix_coords[..., 1] /= H - 1
    # pix_coords = (pix_coords - 0.5) * 2
    pix_coords = camera_ucm.project(X =world_points,frame='w')

    computed_depth = cam_points[:, 2, :].unsqueeze(1).view(B, 1, H, W)

    projected_img = F.grid_sample(img, pix_coords, padding_mode=padding_mode, align_corners=False)
    projected_depth = F.grid_sample(ref_depth, pix_coords, padding_mode=padding_mode, align_corners=False)
    #print("projected_depth", projected_depth)
    save_tensor_as_image(computed_depth[0, 0], '/home/meda/Thesis/train_outputs/computed_depth.png')
    save_tensor_as_image(projected_depth[0, 0], '/home/meda/Thesis/train_outputs/projected_depth.png')
    save_tensor_as_image(projected_img[0,0], '/home/meda/Thesis/train_outputs/projected_img.png')

    return projected_img, projected_depth, computed_depth

def inverse_warp(img, depth, ref_depth, pose, intrinsics, padding_mode='zeros',hparams=None):
    """
    Inverse warp a source image to the target image plane.
    Args:
        img: the source image (where to sample pixels) -- [B, 3, H, W]
        depth: depth map of the target image -- [B, 1, H, W]
        ref_depth: the source depth map (where to sample depth) -- [B, 1, H, W] 
        pose: 6DoF pose parameters from target to source -- [B, 6]
        intrinsics: camera intrinsic matrix -- [B, 3, 3]
    Returns:
        projected_img: Source image warped to the target image plane
        projected_depth: sampled depth from source image  
        computed_depth: computed depth of source image using the target depth
    """
    B, _, H, W = img.size()
    #print('pose_wrap',pose.shape)
    if hparams.model_version == 'L':
        T = pose.to(intrinsics.device).to(torch.float32)  # [B,3,4]
    else:
        T = pose_vec2mat(pose)  # [B,3,4]
    #print('T,intrinsics',T.dtype,intrinsics.dtype,T.shape,intrinsics.shape)
    P = torch.matmul(intrinsics.to(torch.float32), T)[:, :3, :]

    world_points = depth_to_3d(depth, intrinsics) # B 3 H W
    world_points = torch.cat([world_points, torch.ones(B,1,H,W).type_as(img)], 1)
    cam_points = torch.matmul(P, world_points.view(B, 4, -1))

    pix_coords = cam_points[:, :2, :] / (cam_points[:, 2, :].unsqueeze(1) + 1e-7)
    pix_coords = pix_coords.view(B, 2, H, W)
    pix_coords = pix_coords.permute(0, 2, 3, 1)
    pix_coords[..., 0] /= W - 1
    pix_coords[..., 1] /= H - 1
    pix_coords = (pix_coords - 0.5) * 2

    computed_depth = cam_points[:, 2, :].unsqueeze(1).view(B, 1, H, W)

    projected_img = F.grid_sample(img, pix_coords, padding_mode=padding_mode, align_corners=False)
    projected_depth = F.grid_sample(ref_depth, pix_coords, padding_mode=padding_mode, align_corners=False)
    save_tensor_as_image(computed_depth[0, 0], '/home/meda/Thesis/train_outputs_v2/computed_depth.png')
    save_tensor_as_image(projected_depth[0, 0], '/home/meda/Thesis/train_outputs_v2/projected_depth.png')
    save_tensor_as_image(projected_img[0,0], '/home/meda/Thesis/train_outputs_v2/projected_img.png')
    return projected_img, projected_depth, computed_depth


def inverse_rotation_warp(img, rot, intrinsics, padding_mode='zeros'):

    B, _, H, W = img.size()

    R = euler2mat(rot)  # [B, 3, 3]
    P = torch.matmul(intrinsics, R)

    world_points = depth_to_3d(torch.ones(B, 1, H, W).type_as(img), intrinsics) # B 3 H W
    cam_points = torch.matmul(P, world_points.view(B, 3, -1))

    pix_coords = cam_points[:, :2, :] / (cam_points[:, 2, :].unsqueeze(1) + 1e-7)
    pix_coords = pix_coords.view(B, 2, H, W)
    pix_coords = pix_coords.permute(0, 2, 3, 1)
    pix_coords[..., 0] /= W - 1
    pix_coords[..., 1] /= H - 1
    pix_coords = (pix_coords - 0.5) * 2  

    projected_img = F.grid_sample(img, pix_coords, padding_mode=padding_mode, align_corners=True)

    return projected_img
