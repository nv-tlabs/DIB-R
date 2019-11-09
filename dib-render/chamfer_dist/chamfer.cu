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

#include <ATen/ATen.h>
#include <iostream>
using namespace std;
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void ChamferKernel(
	const float* xyz1,
    const float* xyz2,
    float* dist,
    int* idx, int batch_size, int n, int m)
{
    // bidx * height + heiidx
	int presentthread = blockIdx.x * blockDim.x + threadIdx.x;
	int n_idx = presentthread % n;
	int b_idx = (presentthread - n_idx) / n;

	if (b_idx >= batch_size || n_idx >= n) {
		return;
	}
	int min_idx = 0;
	float min_dist = 10000.0;
	float cur_x = xyz1[b_idx * n * 3 + n_idx * 3];
	float cur_y = xyz1[b_idx * n * 3 + n_idx * 3 + 1];
	float cur_z = xyz1[b_idx * n * 3 + n_idx * 3 + 2];
	float next_x, next_y, next_z;
	float diff_x, diff_y, diff_z;
	float tmp_dist;
    for (int i = 0; i < m; i++){
        next_x = xyz2[b_idx * m * 3 + i * 3];
        next_y = xyz2[b_idx * m * 3 + i * 3 + 1];
        next_z = xyz2[b_idx * m * 3 + i * 3 + 2];

        diff_x = cur_x - next_x;
        diff_y = cur_y - next_y;
        diff_z = cur_z - next_z;

        tmp_dist = diff_x * diff_x + diff_y * diff_y + diff_z * diff_z;
        tmp_dist = sqrt(tmp_dist);
        if (tmp_dist < min_dist){
            min_dist = tmp_dist;
            min_idx = i;
        }
    }
    dist[b_idx * n + n_idx] = min_dist;
    idx[b_idx * n + n_idx] = min_idx;
}

void ChamferKernelLauncher(
    const float* xyz1,
    const float* xyz2,
    float* dist1,
    float* dist2,
    int* idx1,
    int* idx2,
    int batch_size, int n, int m){

    const int threadnum = 1024;
	const int totalthread = batch_size * n;
	const int blocknum = totalthread / threadnum + 1;

	const dim3 threads(threadnum, 1, 1);
	const dim3 blocks(blocknum, 1, 1);

	ChamferKernel<<<blocks, threads>>>(xyz1, xyz2, dist1, idx1, batch_size, n, m);
	const int totalthread2 = batch_size * m;
	const int blocknum2 = totalthread2 / threadnum + 1;

    const dim3 threads2(threadnum, 1, 1);
	const dim3 blocks2(blocknum2, 1, 1);
	ChamferKernel<<<blocks2, threads2>>>(xyz2, xyz1, dist2, idx2, batch_size, m, n);

	cudaError_t err = cudaGetLastError();
}

