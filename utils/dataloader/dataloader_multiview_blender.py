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
import cv2

from torch.utils.data import Dataset, DataLoader

#######################################################
class DataProvider(Dataset):
    def __init__(self, file_list,
                 imsz=-1,
                 viewnum=1,
                 mode='train',
                 datadebug=False,
                 classes=None,
                 data_folder=None):

        self.mode = mode
        self.datadebug = datadebug
        
        self.imsz = imsz
        self.viewum = viewnum

        assert self.viewum >= 1

        self.folder = data_folder
        self.camfolder = data_folder
        print(self.folder)

        self.pkl_list = []
        with open(file_list, 'r') as f:
            while True:
                line = f.readline().strip()
                if not line:
                    break
                self.pkl_list.append(line)

        if not classes is None:
            self.filter_class(classes)
            self.imnum = len(self.pkl_list)
            print('imnum {}'.format(self.imnum))

        self.imnum = len(self.pkl_list)
        print(self.pkl_list[0])
        print(self.pkl_list[-1])
        print('imnum {}'.format(self.imnum))
    
    def __len__(self):
        return self.imnum

    def __getitem__(self, idx):
        return self.prepare_instance(idx)

    def filter_class(self, classes):
        new_pkl_list = []
        for pkl in self.pkl_list:
            for cls in classes:
                if cls in pkl:
                    new_pkl_list.append(pkl)
                    break
        self.pkl_list = new_pkl_list

    def load_im_cam(self, pkl_path, catagory, md5name, num):
        imname = '%s/%s/%s/%d.png' % (self.folder, catagory, md5name, num)
        img = cv2.imread(imname, cv2.IMREAD_UNCHANGED)
        img = cv2.resize(img, (self.imsz, self.imsz))
        im_hxwx4 = img.astype('float32') / 255.0

        rotntxname = '%s/%s/%s/%d.npy' % (self.camfolder, catagory, md5name, num)
        rotmtx_4x4 = np.load(rotntxname).astype(np.float32)
        rotmx = rotmtx_4x4[:3, :3]
        transmtx = rotmtx_4x4[:3, 3:4]
        transmtx = -np.matmul(rotmx.T, transmtx)
        renderparam = (rotmx, transmtx)

        return im_hxwx4, renderparam
    
    def prepare_instance(self, idx):
        re = {}
        re['valid'] = True

        pkl_path = self.pkl_list[idx]
        _, fname = os.path.split(pkl_path)
        fname, _ = os.path.splitext(fname)
        catagory, md5name, numname = fname.split('_')
        re['cate'] = catagory
        re['md5'] = md5name
        
        try:
            if self.viewum == 1:
                num = int(numname)
                im_hxwx4, renderparam = self.load_im_cam(pkl_path, catagory, md5name, num)
                
                i = 0
                re['view%d' % i] = {}
                re['view%d' % i]['camrot'] = renderparam[0]
                re['view%d' % i]['campos'] = renderparam[1]
                re['view%d' % i]['im'] = im_hxwx4
                re['view%d' % i]['ori_im'] = np.copy(im_hxwx4)
                re['view%d' % i]['num'] = num
            else:
                for i in range(self.viewum):
                    # 24 views in total
                    num = np.random.randint(24)
                    im_hxwx4, renderparam = self.load_im_cam(pkl_path, catagory, md5name, num)
                    re['view%d' % i] = {}
                    re['view%d' % i]['camrot'] = renderparam[0]
                    re['view%d' % i]['campos'] = renderparam[1][:, 0]
                    re['view%d' % i]['im'] = im_hxwx4
                    re['view%d' % i]['ori_im'] = np.copy(im_hxwx4)
                    re['view%d' % i]['num'] = num
        except:
            re['valid'] = False
            return re
        
        return re


def collate_fn(batch_list):
    collated = {}
    batch_list = [data for data in batch_list if data['valid']]
    if len(batch_list) == 0:
        return None

    keys = ['cate', 'md5']
    for key in keys:
        val = [item[key] for item in batch_list]
        collated[key] = val
    
    viewnum = len(batch_list[0].keys()) - 3
    keys = ['im', 'camrot', 'campos', 'num']
    for i in range(viewnum):
        collated['view%d' % i] = {}
        for key in keys:
            val = [item['view%d' % i][key] for item in batch_list]
            val = np.stack(val, axis=0)
            collated['view%d' % i][key] = val

    return collated


def get_data_loaders(filelist, imsz, viewnum, mode, bs, numworkers, classes=None, data_folder=None):
    print('Building dataloaders')
    dataset_train = DataProvider(filelist, imsz, viewnum,
                                 mode=mode, datadebug=False, classes=classes, data_folder=data_folder)

    if mode == 'test':
        shuffle = False
    else:
        shuffle = True
    
    train_loader = DataLoader(dataset_train, batch_size=bs,
                              shuffle=shuffle, num_workers=numworkers, collate_fn=collate_fn)
    
    print('train num {}'.format(len(dataset_train)))
    print('train iter'.format(len(train_loader)))
    
    return train_loader