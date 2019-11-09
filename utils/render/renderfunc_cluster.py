# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

from __future__ import print_function
from __future__ import division

import torch
import torch.nn
import torch.autograd
import torch.nn as nn
from torch.autograd import Function


################################################################
def p2f(points_bxpx3, faces_fx3, cameras):
    # perspective, use just one camera
    camera_rot_bx3x3, camera_pos_bx3, camera_proj_3x1 = cameras
    cameratrans_rot_bx3x3 = camera_rot_bx3x3.permute(0, 2, 1)
    
    # follow pixel2mesh!!!
    # new_p = cam_mat * (old_p - cam_pos)
    points_bxpx3 = points_bxpx3 - camera_pos_bx3.view(-1, 1, 3)
    points_bxpx3 = torch.matmul(points_bxpx3, cameratrans_rot_bx3x3)
    
    camera_proj_bx1x3 = camera_proj_3x1.view(-1, 1, 3)
    xy_bxpx3 = points_bxpx3 * camera_proj_bx1x3
    xy_bxpx2 = xy_bxpx3[:, :, :2] / xy_bxpx3[:, :, 2:3]

    ##########################################################
    # 1 points
    pf0_bxfx3 = points_bxpx3[:, faces_fx3[:, 0], :]
    pf1_bxfx3 = points_bxpx3[:, faces_fx3[:, 1], :]
    pf2_bxfx3 = points_bxpx3[:, faces_fx3[:, 2], :]
    points3d_bxfx9 = torch.cat((pf0_bxfx3, pf1_bxfx3, pf2_bxfx3), dim=2)
    
    xy_f0 = xy_bxpx2[:, faces_fx3[:, 0], :]
    xy_f1 = xy_bxpx2[:, faces_fx3[:, 1], :]
    xy_f2 = xy_bxpx2[:, faces_fx3[:, 2], :]
    points2d_bxfx6 = torch.cat((xy_f0, xy_f1, xy_f2), dim=2)
    
    ######################################################
    # 2 normals
    v01_bxfx3 = pf1_bxfx3 - pf0_bxfx3
    v02_bxfx3 = pf2_bxfx3 - pf0_bxfx3
    
    # bs cannot be 3, if it is 3, we must specify dim
    normal_bxfx3 = torch.cross(v01_bxfx3, v02_bxfx3, dim=2)
    normalz_bxfx1 = normal_bxfx3[:, :, 2:3]
    # normalz_bxfx1 = torch.abs(normalz_bxfx1)
    
    normallen_bxfx1 = torch.sqrt(torch.sum(normal_bxfx3 ** 2, dim=2, keepdim=True))
    normal1_bxfx3 = normal_bxfx3 / (normallen_bxfx1 + 1e-15)
    
    return points3d_bxfx9, points2d_bxfx6, normalz_bxfx1, normal1_bxfx3


###################################################################

def rendermeshcolor(points_bxpx3, tfcolor_bxpx3, faces_fx3, cameras, func):
    
    points3d_bxfx9, points2d_bxfx6, normalz_bxfx1, normal1_bxfx3 = p2f(points_bxpx3, faces_fx3, cameras)
    
    ######################################################
    # 3 colors
    c0 = tfcolor_bxpx3[:, faces_fx3[:, 0], :]
    c1 = tfcolor_bxpx3[:, faces_fx3[:, 1], :]
    c2 = tfcolor_bxpx3[:, faces_fx3[:, 2], :]

    tfcolor_bxfx9 = torch.cat((c0, c1, c2), dim=2)
    
    return func(points3d_bxfx9, points2d_bxfx6, normalz_bxfx1, tfcolor_bxfx9), normal1_bxfx3


class RenderMeshColor(nn.Module):
    def __init__(self, func):
        super(RenderMeshColor, self).__init__()
        self.func = func

    def forward(self, points_bxpx3, tfcolor_bxpx3, faces_fx3, cameras_1, cameras_2, cameras_3):
        faces_fx3 = faces_fx3.squeeze(0)
        cameras = [cameras_1, cameras_2, cameras_3.squeeze(0)]
        points3d_bxfx9, points2d_bxfx6, normalz_bxfx1, normal1_bxfx3 = p2f(points_bxpx3, faces_fx3, cameras)

        ######################################################
        # 3 colors
        c0 = tfcolor_bxpx3[:, faces_fx3[:, 0], :]
        c1 = tfcolor_bxpx3[:, faces_fx3[:, 1], :]
        c2 = tfcolor_bxpx3[:, faces_fx3[:, 2], :]
        tfcolor_bxfx9 = torch.cat((c0, c1, c2), dim=2)

        return self.func(points3d_bxfx9, points2d_bxfx6, normalz_bxfx1, tfcolor_bxfx9), normal1_bxfx3