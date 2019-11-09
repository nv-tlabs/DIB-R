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

import torch

from torch.utils.cpp_extension import load
cd = load(name="chamfer",
          sources=["chamfer_dist/chamfer.cpp",
                   "chamfer_dist/chamfer.cu"])

class ChamferFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, xyz1, xyz2):
        batchsize, n, _ = xyz1.size()
        _, m, _ = xyz2.size()

        dist1 = torch.zeros(batchsize, n).float().to(xyz1.device)
        dist2 = torch.zeros(batchsize, m).float().to(xyz1.device)

        idx1 = torch.zeros(batchsize, n).type(torch.IntTensor).to(xyz1.device)
        idx2 = torch.zeros(batchsize, m).type(torch.IntTensor).to(xyz1.device)

        cd.forward_cuda(xyz1, xyz2, dist1, dist2, idx1, idx2)
        return idx1, idx2, dist1, dist2

    @staticmethod
    def backward(ctx, graddist1, graddist2):
        ints = ctx.saved_tensors
        gradxyz1 = torch.zeros(ints.size())
        return gradxyz1, gradxyz1

class Chamfer(torch.nn.Module):
    def forward(self, points1, points2):
        return ChamferFunction.apply(points1, points2)