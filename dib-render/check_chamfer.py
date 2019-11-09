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

import glob
import os
import torch
import argparse
import sys
sys.path.append('../utils/')
from utils.utils_mesh import loadobjcolor, loadobj
import numpy as np
from chamfer_dist.chamfer import Chamfer
torch.random.manual_seed(12345)
np.random.seed(123456)
import random
random.seed(1234567)
from collections import defaultdict

def get_args():
    parser = argparse.ArgumentParser(description='check chamfer-dist and F-score')
    parser.add_argument('--folder', type=str, default='debug',
                        help='prediction folder')
    parser.add_argument('--gt_folder', type=str, default='debug',
                        help='gt folder')
    args = parser.parse_args()
    return args

def sample(verts, faces, num=10000, ret_choice = False):
    dist_uni = torch.distributions.Uniform(torch.tensor([0.0]).cuda(), torch.tensor([1.0]).cuda())
    x1,x2,x3 = torch.split(torch.index_select(verts, 0, faces[:,0]) - torch.index_select(verts, 0, faces[:,1]), 1, dim = 1)
    y1,y2,y3 = torch.split(torch.index_select(verts, 0, faces[:,1]) - torch.index_select(verts, 0, faces[:,2]), 1, dim = 1)
    a = (x2*y3 - x3*y2)**2
    b = (x3*y1 - x1*y3)**2
    c = (x1*y2 - x2*y1)**2
    Areas = torch.sqrt(a+b+c)/2
    Areas = Areas / torch.sum(Areas)
    cat_dist = torch.distributions.Categorical(Areas.view(-1))
    choices = cat_dist.sample_n(num)
    select_faces = faces[choices]
    xs = torch.index_select(verts, 0,select_faces[:,0])
    ys = torch.index_select(verts, 0,select_faces[:,1])
    zs = torch.index_select(verts, 0,select_faces[:,2])
    u = torch.sqrt(dist_uni.sample_n(num))
    v = dist_uni.sample_n(num)
    points = (1- u)*xs + (u*(1-v))*ys + u*v*zs
    if ret_choice:
        return points, choices
    else:
        return points

def main() :
    args = get_args()
    folder = args.folder
    print('==> get all predictions')
    print(folder)
    meshfiles = glob.glob('%s/*/*/*.obj' % folder)
    print('Length mesh files: ', len(meshfiles))
    gt_folder = args.gt_folder
    print ('==> starting ')
    chamfer = Chamfer()
    cates = '02691156,02828884,02933112,02958343,03001627,03211117,03636649,03691459,04090263,04256520,04379243,04401088,04530566'
    cates = cates.split(',')
    dist_cate = defaultdict(list)
    F_score = defaultdict(list)
    random.shuffle(meshfiles)
    toa = [0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05]
    for i, fl in enumerate(meshfiles):
        tmp_name = fl
        names = fl.split('/')
        cat = names[-3]
        md5 = names[-2]
        gt_name = os.path.join(gt_folder, cat, md5, 'model-0.45.obj')
        vertices, faces = loadobjcolor(tmp_name)
        if vertices.shape[0] == 0:
            print('Error in read shape')
            print(tmp_name)
            continue
        gt_vertices, gt_faces = loadobj(gt_name)
        if gt_vertices.shape[0] == 0:
            print('Error in read shape')
            print(tmp_name)
            continue
        gt_vertices = torch.from_numpy(gt_vertices).float().cuda()
        gt_faces = torch.from_numpy(gt_faces).cuda()
        try:
            sample_gt = sample(gt_vertices, gt_faces, num=2500)
        except:
            print('Error in sample:')
            print(fl)
            continue
        sample_gt = sample_gt.unsqueeze(0)

        vertices = torch.from_numpy(vertices).float().cuda()
        faces = torch.from_numpy(faces).cuda()
        try:
            sample_p = sample(vertices, faces, num=2500)
        except:
            print('Error in sample:')
            print(fl)
            continue
        sample_p = sample_p.unsqueeze(0)

        _, _, dist1, dist2 = chamfer(sample_p, sample_gt)

        cf = (dist1.mean() + dist2.mean()) / 2
        f_score_list = []
        for t in toa:
            fp = (dist1 > t).float().sum()
            tp = (dist1 <= t).float().sum()
            precision = tp / (tp + fp)
            tp = (dist2 <= t).float().sum()
            fn = (dist2 > t).float().sum()
            recall = tp / (tp + fn)
            f_score = 2 * (precision * recall) / (precision + recall + 1e-8)
            f_score_list.append(f_score.item())
        F_score[cat].append(f_score_list)
        dist_cate[cat].append(cf.item())
        if i % 1000 == 999:
            print('-'*50)
            print('Step: ', i)
            print('==> chamfer')
            for c in cates:
                print('%s: %.10f' %(c, np.mean(dist_cate[c])))
            print('Mean of ALL: %.10f'%(np.mean([np.mean(dist_cate[c]) for c in cates])))
            print('==> F')
            mean_score_list = []
            for c in cates:
                print(c, end='')
                s = F_score[c]
                s = np.asarray(s)
                mean_s = np.mean(s, axis=0)
                mean_score_list.append(mean_s)
                for i in range(len(toa)):
                    print(' %.10f' % mean_s[i], end='')
                print('')
            all_mean = np.asarray(mean_score_list)
            all_mean = np.mean(all_mean, axis=0)
            print('ALL Mean:', end='')
            for i in range(len(toa)):
                print(' %.10f' % all_mean[i], end='')
            print('\n')

if __name__ == '__main__':
    main()