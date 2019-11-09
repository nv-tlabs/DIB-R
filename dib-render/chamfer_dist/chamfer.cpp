/*
 * Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#include <torch/torch.h>
#include <iostream>
using namespace std;

void ChamferKernelLauncher(
    const float* xyz1,
    const float* xyz2,
    float* dist1,
    float* dist2,
    int* idx1,
    int* idx2,
    int b, int n, int m);

void chamfer_forward_cuda(
    const at::Tensor xyz1,
    const at::Tensor xyz2,
    at::Tensor dist1,
    at::Tensor dist2,
    at::Tensor idx1,
    at::Tensor idx2)
{
    int batch_size = xyz1.size(0);
    int n = xyz1.size(1);
    int m = xyz2.size(1);
    ChamferKernelLauncher(xyz1.data<float>(), xyz2.data<float>(),
                                            dist1.data<float>(), dist2.data<float>(),
                                            idx1.data<int>(), idx2.data<int>(), batch_size, n, m);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward_cuda", &chamfer_forward_cuda, "Chamfer forward (CUDA)");
}
