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

import argparse


def get_args():
    parser = argparse.ArgumentParser(description='dib-r')
    
    parser.add_argument('--filelist', type=str, default='train_list.txt',
                        help='filelist name')
    parser.add_argument('--thread', type=int, default=8,
                        help='num of workers')
    parser.add_argument('--svfolder', type=str, default='experiment',
                        help='save folder for experiments ')
    parser.add_argument('--g_model_dir', type=str, default='pretrained_model',
                        help='save path for pretrained model')
    parser.add_argument('--data_folder', type=str, default='data',
                        help='data folder')
    parser.add_argument('--iter', type=int, default=-1,
                        help='start iteration')
    
    parser.add_argument('--loss', type=str, default='iou',
                        help='loss type')
    parser.add_argument('--camera', type=str, default='per',
                        help='camera mode')
    parser.add_argument('--view', type=int, default=2,
                        help='view number')
    
    parser.add_argument('--img_dim', type=int, default=64,
                        help='dim of image')
    parser.add_argument('--img_channels', type=int, default=4,
                        help='image channels')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='batch size')
    
    parser.add_argument('--epoch', type=int, default=1000,
                        help='training epoch')
    parser.add_argument('--iter_log', type=int, default=50,
                        help='iterations per log')
    parser.add_argument('--iter_sample', type=int, default=1000,
                        help='iterations per sample')
    parser.add_argument('--iter_model', type=int, default=10000,
                        help='iterations per model saving')

    parser.add_argument('--sil_lambda', type=float, default=1,
                        help='hyperparamter for sil')
    
    args = parser.parse_args()

    return args

