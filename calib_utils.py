import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def torch_inverse_batch(deltp):
    # deltp must be [K,2]
    assert deltp.dim() == 3 and deltp.size(1) == 2 and deltp.size(2) == 2, 'The deltp format is not right : {}'.format(
        deltp.size())
    a, b, c, d = deltp[:, 0, 0], deltp[:, 0, 1], deltp[:, 1, 0], deltp[:, 1, 1]
    a = a + np.finfo(float).eps
    d = d + np.finfo(float).eps
    divide = a * d - b * c + np.finfo(float).eps
    inverse = torch.stack([d, -b, -c, a], dim=1) / divide.unsqueeze(1)
    return inverse.view(-1, 2, 2)


def warp_feature_batch(feature, pts_location, patch_size):
    # feature must be [1,C,H,W] and pts_location must be [Num-Pts, (x,y)]
    _, C, H, W = list(feature.size())
    num_pts = pts_location.size(0)
    assert isinstance(patch_size, int) and feature.size(0) == 1 and pts_location.size(
        1) == 2, 'The shapes of feature or points are not right : {} vs {}'.format(feature.size(), pts_location.size())
    assert W > 1 and H > 1, 'To guarantee normalization {}, {}'.format(W, H)

    def normalize(x, L):
        return -1. + 2. * x / (L - 1)

    crop_box = torch.cat([pts_location - patch_size, pts_location + patch_size], 1).float()
    crop_box[:, [0, 2]] = normalize(crop_box[:, [0, 2]], W)
    crop_box[:, [1, 3]] = normalize(crop_box[:, [1, 3]], H)

    # crop_box[:, [0,2]] = int(normalize(crop_box[:, [0,2]], W))
    # crop_box[:, [1,3]] = int(normalize(crop_box[:, [1,3]], H))

    affine_parameter = [(crop_box[:, 2] - crop_box[:, 0]) / 2, crop_box[:, 0] * 0,
                        (crop_box[:, 2] + crop_box[:, 0]) / 2,
                        crop_box[:, 0] * 0, (crop_box[:, 3] - crop_box[:, 1]) / 2,
                        (crop_box[:, 3] + crop_box[:, 1]) / 2]
    # affine_parameter = [(crop_box[:,2]-crop_box[:,0])/2, MU.np2variable(torch.zeros(num_pts),feature.is_cuda,False), (crop_box[:,2]+crop_box[:,0])/2,
    #                    MU.np2variable(torch.zeros(num_pts),feature.is_cuda,False), (crop_box[:,3]-crop_box[:,1])/2, (crop_box[:,3]+crop_box[:,1])/2]
    theta = torch.stack(affine_parameter, 1).view(num_pts, 2, 3)
    feature = feature.expand(num_pts, C, H, W)
    grid_size = torch.Size([num_pts, 1, 2 * patch_size + 1, 2 * patch_size + 1])
    grid = F.affine_grid(theta, grid_size, align_corners=True)
    sub_feature = F.grid_sample(feature, grid, align_corners=True)
    return sub_feature

def Generate_Weight(patch_size, sigma=None):
    assert isinstance(patch_size, list) or isinstance(patch_size, tuple)
    assert patch_size[0] > 0 and patch_size[1] > 0, 'the patch size must > 0 rather :{}'.format(patch_size)
    center = [(patch_size[0] - 1.) / 2, (patch_size[1] - 1.) / 2]
    maps = np.fromfunction(lambda x, y: (x - center[0]) ** 2 + (y - center[1]) ** 2, (patch_size[0], patch_size[1]),
                           dtype=int)
    if sigma is None: sigma = min(patch_size[0], patch_size[1]) / 2.
    maps = np.exp(maps / -2.0 / sigma / sigma)
    maps[0, :] = maps[-1, :] = maps[:, 0] = maps[:, -1] = 0
    return maps.astype(np.float32)

class SobelConv(nn.Module):
    def __init__(self, tag, dtype):
        super(SobelConv, self).__init__()
        if tag == 'x':
            Sobel = np.array([[-1. / 8, 0, 1. / 8], [-2. / 8, 0, 2. / 8], [-1. / 8, 0, 1. / 8]])
        elif tag == 'y':
            Sobel = np.array([[-1. / 8, -2. / 8, -1. / 8], [0, 0, 0], [1. / 8, 2. / 8, 1. / 8]])
        else:
            raise NameError('Do not know this tag for Sobel Kernel : {}'.format(tag))
        Sobel = torch.from_numpy(Sobel).type(dtype)
        Sobel = Sobel.view(1, 1, 3, 3)
        self.register_buffer('weight', Sobel)
        self.tag = tag
    def forward(self, input):
        weight = self.weight.expand(input.size(1), 1, 3, 3).contiguous()
        return F.conv2d(input, weight, groups=input.size(1), padding=1)
    def __repr__(self):
        return ('{name}(tag={tag})'.format(name=self.__class__.__name__, **self.__dict__))


def lk_track(feature_old, feature_new, pts_locations, patch_size, max_step):
    # feature[old,new] : 4-D tensor [1, C, H, W]
    # pts_locations is a 2-D tensor [Num-Pts, (Y,X)]
    if feature_new.dim() == 3:
        feature_new = feature_new.unsqueeze(0)
    if feature_old is not None and feature_old.dim() == 3:
        feature_old = feature_old.unsqueeze(0)
    assert feature_new.dim() == 4, 'The dimension of feature-new is not right : {}.'.format(feature_new.dim())
    BB, C, H, W = list(feature_new.size())
    assert isinstance(patch_size, int), 'The format of lk-parameters are not right : {}'.format(patch_size)
    num_pts = pts_locations.size(0)
    device = feature_new.device
    # feature_T should be a [num_pts, C, patch, patch] tensor
    feature_T = warp_feature_batch(feature_old, pts_locations, patch_size)
    assert feature_T.size(2) == patch_size * 2 + 1 and feature_T.size(
        3) == patch_size * 2 + 1, 'The size of feature-template is not ok : {}'.format(feature_T.size())

    weight_map = Generate_Weight([patch_size * 2 + 1, patch_size * 2 + 1])  # [H, W]
    with torch.no_grad():
        weight_map = torch.tensor(weight_map).view(1, 1, 1, patch_size * 2 + 1, patch_size * 2 + 1).to(device)
        sobelconvx = SobelConv('x', feature_new.dtype).to(device)
        sobelconvy = SobelConv('y', feature_new.dtype).to(device)

    gradiant_x = sobelconvx(feature_T)
    gradiant_y = sobelconvy(feature_T)
    J = torch.stack([gradiant_x, gradiant_y], dim=1)
    weightedJ = J * weight_map
    """
      J is 68*2*C*H*W.
      weight_map = 1*1*C*H*W
      J*w refer to the element-wise multiply, which using the broadcast machanism.
    """

    H = torch.bmm(weightedJ.view(num_pts, 2, -1), J.view(num_pts, 2, -1).transpose(2, 1))
    inverseH = torch_inverse_batch(H)
    for step in range(max_step):
        feature_I = warp_feature_batch(feature_new, pts_locations, patch_size)
        r = feature_I - feature_T
        sigma = torch.bmm(weightedJ.view(num_pts, 2, -1), r.view(num_pts, -1, 1))
        deltap = torch.bmm(inverseH, sigma).squeeze(-1)
        pts_locations = pts_locations - deltap
    return pts_locations

def track_bidirectional(faces, locations):
    patch_size = 15
    max_step = 10
    frames_num = len(faces)

    forward_pts = [locations[0].copy()]
    for i in range(1, frames_num):
        # HWC(CV) to CHW(Torch)
        feature_old = torch.tensor(faces[i-1]).transpose(0, 2).transpose(1, 2).float() / 255
        feature_new = torch.tensor(faces[i]).transpose(0, 2).transpose(1, 2).float() / 255
        location_old = torch.tensor(forward_pts[i - 1])
        forward_pt = lk_track(feature_old, feature_new, location_old, patch_size, max_step).numpy()
        forward_pts.append(forward_pt)

    feedback_pts = [None] * (frames_num - 1) + [forward_pts[-1].copy()]
    for i in range(frames_num - 2, -1, -1):
        feature_old = torch.tensor(faces[i+1]).transpose(0, 2).transpose(1, 2).float() / 255
        feature_new = torch.tensor(faces[i]).transpose(0, 2).transpose(1, 2).float() / 255
        location_old = torch.tensor(feedback_pts[i + 1])
        feedback_pt = lk_track(feature_old, feature_new, location_old, patch_size, max_step).numpy()
        feedback_pts[i] = feedback_pt

    return forward_pts, feedback_pts