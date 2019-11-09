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
import torch.nn as nn
import torch.nn.functional as TorchF


#################################
class Ecoder(nn.Module):
    def __init__(self, N_CHANNELS, N_KERNELS, \
                 BATCH_SIZE, IMG_DIM, VERBOSE=False, pred_cam=False):
        super(Ecoder, self).__init__()
        block1 = self.convblock(N_CHANNELS, 64, N_KERNELS, stride=2, pad=2)
        block2 = self.convblock(64, 128, N_KERNELS, stride=2, pad=2)
        block3 = self.convblock(128, 256, N_KERNELS, stride=2, pad=2)
        
        linear1 = self.linearblock(16384, 1024)
        linear2 = self.linearblock(1024, 1024)
        self.linear3 = nn.Linear(1024, 1024)
        self.pred_cam = pred_cam
        if self.pred_cam:
            self.pred_cam_linear_1 = nn.Linear(1024, 128)
            self.pred_cam_linear_2 = nn.Linear(128, 9 + 3)
        
        linear4 = self.linearblock(1024, 1024)
        linear5 = self.linearblock(1024, 2048)
        self.linear6 = nn.Linear(2048, 1926)
        
        linear42 = self.linearblock(1024, 1024)
        linear52 = self.linearblock(1024, 2048)
        self.linear62 = nn.Linear(2048, 1926)
        
        #################################################
        all_blocks = block1 + block2 + block3
        self.encoder1 = nn.Sequential(*all_blocks)
        
        all_blocks = linear1 + linear2
        self.encoder2 = nn.Sequential(*all_blocks)
        
        all_blocks = linear4 + linear5
        self.decoder = nn.Sequential(*all_blocks)
        
        all_blocks = linear42 + linear52
        self.decoder2 = nn.Sequential(*all_blocks)
        
        # Initialize with Xavier Glorot
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d) \
            or isinstance(m, nn.Linear) \
            or isinstance(object, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                nn.init.normal_(m.weight, mean=0, std=0.001)

        # Free some memory
        del all_blocks, block1, block2, block3, \
        linear1, linear2, linear4, linear5, \
        linear42, linear52

        # Print summary
        if VERBOSE:
            self.summary(BATCH_SIZE, N_CHANNELS, IMG_DIM)
    
    def convblock(self, indim, outdim, ker, stride, pad):
        block2 = [
            nn.Conv2d(indim, outdim, ker, stride, pad),
            nn.BatchNorm2d(outdim),
            nn.ReLU()
        ]
        return block2
    
    def linearblock(self, indim, outdim):
        block2 = [
            nn.Linear(indim, outdim),
            nn.BatchNorm1d(outdim),
            nn.ReLU()
        ]
        return block2
        
    def forward(self, x):
        
        for layer in self.encoder1:
            x = layer(x)
        
        bnum = x.shape[0] 
        x = x.view(bnum, -1)  # flatten the encoder1 output
        for layer in self.encoder2:
            x = layer(x)
        x = self.linear3(x)
        if self.pred_cam:
            cam_x = TorchF.relu(self.pred_cam_linear_1(x))
            pred_cam = self.pred_cam_linear_2(cam_x)
        x1 = x
        for layer in self.decoder:
            x1 = layer(x1)
        x1 = self.linear6(x1)
        
        x2 = x
        for layer in self.decoder2:
            x2 = layer(x2)
        x2 = self.linear62(x2)
        if self.pred_cam:
            return x1, x2, pred_cam
        return x1, x2

    def summary(self, BATCH_SIZE, N_CHANNELS, IMG_DIM):
        
        x = torch.zeros(BATCH_SIZE, N_CHANNELS, IMG_DIM, IMG_DIM)
        
        # Print the title in a good design
        # for easy recognition.
        print()
        summary_title = '| {} Summary |'.format(self.__class__.__name__)
        for _ in range(len(summary_title)):
            print('-', end='')
        print()
        print(summary_title)
        for _ in range(len(summary_title)):
            print('-', end='')
        print('\n')
        
        # Run forward pass while not tracking history on
        # tape using `torch.no_grad()` for printing the
        # output shape of each neural layer operation.
        print('Input: {}'.format(x.size()))
        with torch.no_grad():
            
            for layer in self.encoder1:
                x = layer(x)
                print('Out: {} \tLayer: {}'.format(x.size(), layer))
            
            bnum = x.shape[0] 
            x = x.view(bnum, -1)  # flatten the encoder1 output
            print('Out: {} \tlayer: {}'.format(x.size(), 'Reshape: Flatten'))
            
            for layer in self.encoder2:
                x = layer(x)
                print('Out: {} \tLayer: {}'.format(x.size(), layer))
            
            x = self.linear3(x)
            print('Out: {} \tLayer: {}'.format(x.size(), self.linear3))
            
            for layer in self.decoder:
                x = layer(x)
                print('Out: {} \tLayer: {}'.format(x.size(), layer))
            
            x = self.linear6(x)
            print('Out: {} \tLayer: {}'.format(x.size(), self.linear6))


if __name__ == '__main__':
    
    ###################################
    BATCH_SIZE = 64
    IMG_DIM = 64
    N_CHANNELS = 4
    N_KERNELS = 5
    VERBOSE = True

    model = Ecoder(N_CHANNELS, N_KERNELS, BATCH_SIZE, IMG_DIM, VERBOSE)
