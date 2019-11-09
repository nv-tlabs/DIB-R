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

import os
import numpy as np
import glob
import binvox_rw
import multiprocessing
from functools import partial
import argparse

def get_args():
    parser = argparse.ArgumentParser(description='check iou score')
    parser.add_argument('--folder', type=str, default='debug',
                        help='prediction folder')
    parser.add_argument('--gt_folder', type=str, default='debug',
                        help='gt folder')
    args = parser.parse_args()
    return args

def evaluate_instance(file_name, gtfolder=None):
    if 'imview' in file_name or '_1' in file_name:
        return [0]

    prefl = file_name
    names = prefl.split('/')
    cat = names[-3]
    md5 = names[-2]

    gtfl = '%s/%s/%s/model-0.45.binvox' % (gtfolder, cat, md5)

    try:
        with open(prefl, 'rb') as f:
            data = binvox_rw.read_as_3d_array(f)
        with open(gtfl, 'rb') as f:
            data2 = binvox_rw.read_as_3d_array(f)
    except:
        print('Error in read data')
        print(prefl)
        print(gtfl)
        return [0]

    iouall = data.data | data2.data
    iouoverlap = data.data & data2.data
    iouthis = np.sum(iouoverlap) / (np.sum(iouall) + 1e-8)
    iouthisgt = np.sum(iouoverlap) / (np.sum(data2.data) + 1e-8)
    iouthispre = np.sum(iouoverlap) / (np.sum(data.data) + 1e-8)

    return [cat, iouthis, iouthisgt, iouthispre]


def filter(file_list):
    new_list = []
    for name in file_list:
        if '_1' in file_list:
            continue
        new_list.append(name)
    return new_list

def main():
    args = get_args()
    folder = args.folder
    print(folder)

    voxelfiles = glob.glob('%s/*/*/*.binvox' % folder)
    print('Length files: ', len(voxelfiles))

    iou = {}
    iougt = {}
    ioupre = {}
    catenum = {}
    cates = '02691156,02828884,02933112,02958343,03001627,03211117,03636649,03691459,04090263,04256520,04379243,04401088,04530566'
    cates = cates.split(',')

    for ca in cates:
        iou[ca] = 0
        iougt[ca] = 0
        ioupre[ca] = 0
        catenum[ca] = 0

    gtfolder = args.gt_folder

    pool = multiprocessing.Pool(multiprocessing.cpu_count())
    print("Using %i cpus" % multiprocessing.cpu_count())

    results = []

    for x in pool.imap_unordered(partial(evaluate_instance, gtfolder=gtfolder), voxelfiles):
        if x[0] != 0:
            results.append(x)

    pool.close()
    pool.join()

    for i in range(len(results)):
        result = results[i]
        if result[0] == 0:
            continue
        if len(result) == 1:
            continue
        cat, iouthis, iouthisgt, iouthispre = result
        iou[cat] += iouthis
        iougt[cat] += iouthisgt
        ioupre[cat] += iouthispre
        catenum[cat] += 1

    re = []
    for ca in cates:
        iou[ca] /= catenum[ca] + 1e-8
        iougt[ca] /= catenum[ca] + 1e-8
        ioupre[ca] /= catenum[ca] + 1e-8
        print('{}, {} {} {}'.format(ca, iou[ca], ioupre[ca], iougt[ca], catenum[ca]))
        re.append([int(ca), iou[ca], ioupre[ca], iougt[ca], catenum[ca]])

    re = np.array(re, dtype=np.float32)
    path = '%s-iou.npy' % folder
    np.save(file=path, arr=re)

    meanval = np.mean(re, axis=0)
    print('{}, {} {} {}'.format('mean', meanval[1], meanval[2], meanval[3]))
    iou_format = ['%.4f'%(t) for t in re[:,1]]
    print('&'.join(iou_format))
    print('Category Num: ', re[:, -1])

if __name__ == '__main__':
    main()
