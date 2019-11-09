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
import torchvision.utils as vutils

import os
import numpy as np
from config import get_args

import sys
sys.path.append('../utils/')
sys.path.append('./render_cuda')

from dataloader.dataloader_multiview_blender import get_data_loaders

from utils.utils_mesh import loadobj, \
    face2edge, edge2face, face2pneimtx, mtx2tfsparse, savemesh, savemeshcolor

from utils.utils_perspective import camera_info_batch, perspectiveprojectionnp

from model.modelcolor import Ecoder

sys.path.append('../utils/render')
from renderfunc_cluster import rendermeshcolor as rendermesh
from render_cuda.utils_render_color2 import linear

############################################
# Make experiments reproducible
torch.manual_seed(123456)
np.random.seed(123456)
eps = 1e-15

####################
# Make directories #
# - Samples        #
# - Checkpoints    #
####################

args = get_args()
args.img_dim = 64
args.batch_size = 64

FILELIST = args.filelist
IMG_DIM = args.img_dim
N_CHANNELS = args.img_channels
BATCH_SIZE = args.batch_size
TOTAL_EPOCH = args.epoch
ITERS_PER_LOG = args.iter_log
ITERS_PER_SAMPLE = args.iter_sample
ITERS_PER_MODEL = args.iter_model
VERBOSE = True

print('------------------')
print('| Configurations |')
print('------------------')
print('')
print('IMG_DIM:         {}'.format(IMG_DIM))
print('N_CHANNELS:      {}'.format(N_CHANNELS))
print('BATCH_SIZE:      {}'.format(BATCH_SIZE))
print('FILELIST:        {}'.format(FILELIST))
print('TOTAL_EPOCH:     {}'.format(TOTAL_EPOCH))
print('ITERS_PER_LOG:   {}'.format(ITERS_PER_LOG))
print('VERBOSE:         {}'.format(VERBOSE))
print('')

##########################################################
lossname = args.loss
cameramode = args.camera
viewnum = args.view

test_iter_num = args.iter
svfolder = args.svfolder
g_model_dir = args.g_model_dir
data_folder = args.data_folder

####################
# Load the dataset #
####################

# for test, only one view
viewnum = 1

filelist = FILELIST
imsz = IMG_DIM
numworkers = args.thread
data = get_data_loaders(filelist, imsz, viewnum, mode='test',
                        bs=BATCH_SIZE, numworkers=numworkers,data_folder=data_folder)

############################################
# load obj template, the sphere and blender camera
# sphere.obj: A unit sphere with 642 vertices and 1280 faces
pointnp_px3, facenp_fx3 = loadobj('sphere.obj')
edge_ex2 = face2edge(facenp_fx3)
edgef_ex2 = edge2face(facenp_fx3, edge_ex2)
pneimtx = face2pneimtx(facenp_fx3)

pnum = pointnp_px3.shape[0]
fnum = facenp_fx3.shape[0]
enum = edge_ex2.shape[0]

camfovy = 49.13434207744484 / 180.0 * np.pi
camprojmtx = perspectiveprojectionnp(camfovy, 1.0)

################################################
# Define device, neural nets, optimizers, etc. #
################################################

# Automatic GPU/CPU device placement
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Create models
g_model_im2mesh = Ecoder(N_CHANNELS, N_KERNELS=5, BATCH_SIZE=BATCH_SIZE, IMG_DIM=IMG_DIM, VERBOSE=VERBOSE).to(device)

############
# Training #
############

def test(iter_num=-1):
    print('Begin Testing!')
    # Try loading the latest existing checkpoints based on iter_num
    g_model_im2mesh.load_state_dict(torch.load(g_model_dir), strict=True)
    g_model_im2mesh.eval()
    print('Loaded the latest checkpoints from {}'.format(g_model_dir))

    # Directory for test samples
    model_iter = iter_num
    if not os.path.exists(os.path.join(svfolder, 'test-%d' % model_iter)):
        print('Make Save Dir')
        os.makedirs(os.path.join(svfolder, 'test-%d' % model_iter))

    global pointnp_px3, facenp_fx3, edgef_ex2, pneimtx
    global camprojmtx
    global cameramode, lossname

    p = pointnp_px3
    pmax = np.max(p, axis=0, keepdims=True)
    pmin = np.min(p, axis=0, keepdims=True)
    pmiddle = (pmax + pmin) / 2
    p = p - pmiddle

    assert cameramode == 'per', 'now we only support perspective'
    pointnp_px3 = p * 0.35

    tfp_1xpx3 = torch.from_numpy(pointnp_px3).to(device).view(1, pnum, 3)
    tff_fx3 = torch.from_numpy(facenp_fx3).to(device)

    tfcamproj = torch.from_numpy(camprojmtx).to(device)

    iou = {}
    catenum = {}
    cates = '02691156,02828884,02933112,02958343,03001627,03211117,03636649,03691459,04090263,04256520,04379243,04401088,04530566'
    cates = cates.split(',')
    for ca in cates:
        iou[ca] = 0
        catenum[ca] = 0

    iter_num = 0
    for i, da in enumerate(data):
        if da is None:
            continue
        iter_num += 1
        tfims = []
        tfcams = []

        for j in range(viewnum):
            imnp = da['view%d' % j]['im']
            bs = imnp.shape[0]
            imnp_bxcxhxw = np.transpose(imnp, [0, 3, 1, 2])
            tfim_bx4xhxw = torch.from_numpy(imnp_bxcxhxw).to(device)
            tfims.append(tfim_bx4xhxw)
            # camera
            camrot_bx3x3 = da['view%d' % j]['camrot']
            campos_bx3 = da['view%d' % j]['campos']
            tfcamrot = torch.from_numpy(camrot_bx3x3).to(device)
            tfcampos = torch.from_numpy(campos_bx3).to(device)
            tfcameras = [tfcamrot, tfcampos, tfcamproj]
            tfcams.append(tfcameras)

        ########################################3
        with torch.no_grad():
            meshes = []
            meshcolors = []
            meshmovs = []

            # generate j-th mesh
            for j in range(viewnum):
                meshmov_bxp3, mc_bxp3 = g_model_im2mesh(tfims[j][:, :args.img_channels,:,:])
                meshmov_bxpx3 = meshmov_bxp3.view(bs, -1, 3)
                mesh_bxpx3 = meshmov_bxpx3 + tfp_1xpx3
                mc_bxpx3 = mc_bxp3.view(bs, -1, 3)

                # normalize
                mesh_max = torch.max(mesh_bxpx3, dim=1, keepdim=True)[0]
                mesh_min = torch.min(mesh_bxpx3, dim=1, keepdim=True)[0]
                mesh_middle = (mesh_min + mesh_max) / 2
                mesh_bxpx3 = mesh_bxpx3 - mesh_middle

                bs = mesh_bxpx3.shape[0]
                mesh_biggest = torch.max(mesh_bxpx3.view(bs, -1), dim=1)[0]
                mesh_bxpx3 = mesh_bxpx3 / mesh_biggest.view(bs, 1, 1) * 0.45

                meshes.append(mesh_bxpx3)
                meshcolors.append(mc_bxpx3)
                meshmovs.append(meshmov_bxpx3)

            meshesvv = []
            mcvv = []
            tfcamsvv = [[], [], tfcamproj]
            gtvv = []

            # use j-th mesh
            for j in range(viewnum):
                # generate with k-th camera
                for k in range(viewnum):
                    mesh_bxpx3 = meshes[j]
                    mc_bxpx3 = meshcolors[j]
                    meshesvv.append(mesh_bxpx3)
                    mcvv.append(mc_bxpx3)
                    tfcamrot_bx3x3, tfcampos_bx3, _ = tfcams[k]
                    tfcamsvv[0].append(tfcamrot_bx3x3)
                    tfcamsvv[1].append(tfcampos_bx3)
                    # k-th camera, k-th image
                    tfim_bx4xhxw = tfims[k]
                    gtvv.append(tfim_bx4xhxw)

            mesh_vvbxpx3 = torch.cat(meshesvv)
            mc_vvbxpx3 = torch.cat(mcvv)
            tfcamsvv[0] = torch.cat(tfcamsvv[0])
            tfcamsvv[1] = torch.cat(tfcamsvv[1])
            tmp, _ = rendermesh(mesh_vvbxpx3, mc_vvbxpx3, tff_fx3, tfcamsvv, linear)
            impre_vvbxhxwx3, silpred_vvbxhxwx1 = tmp

            # Compute loss
            tfim_vvbx4xhxw = torch.cat(gtvv)

            impre_vvbx3xhxw = impre_vvbxhxwx3.permute(0, 3, 1, 2)
            imgt_vvbx3xhxw = tfim_vvbx4xhxw[:, :3, :, :]
            colloss = 3 * torch.mean(torch.abs(impre_vvbx3xhxw - imgt_vvbx3xhxw))

            silpred_vvbx1xhxw = silpred_vvbxhxwx1.view(viewnum * viewnum * bs, 1, IMG_DIM, IMG_DIM)
            silgt = tfim_vvbx4xhxw[:, 3:4, :, :]

            silmul = silpred_vvbx1xhxw * silgt
            siladd = silpred_vvbx1xhxw + silgt
            silmul = silmul.view(bs, -1)
            siladd = siladd.view(bs, -1)
            iouup = torch.sum(silmul, dim=1)
            ioudown = torch.sum(siladd - silmul, dim=1)
            iouneg = iouup / (ioudown + eps)
            silloss = 1.0 - torch.mean(iouneg)

        iouneg = iouneg.detach().cpu().numpy()
        for cid, ca in enumerate(da['cate']):
            iou[ca] += iouneg[cid]
            catenum[ca] += 1

        if iter_num % 100 == 0:
            # Print statistics
            print('epo: {}, iter: {}, color_loss: {}, iou_loss: {}'. \
                  format(0, iter_num, colloss, silloss))

        if iter_num % ITERS_PER_SAMPLE == ITERS_PER_SAMPLE - 1:
            silpred_vvbx3xhxw = silpred_vvbx1xhxw.repeat(1, 3, 1, 1)
            silgt_vvbx3xhxw = tfim_vvbx4xhxw[:, 3:4, :, :].repeat(1, 3, 1, 1)
            re = torch.cat((imgt_vvbx3xhxw, silgt_vvbx3xhxw, impre_vvbx3xhxw, silpred_vvbx3xhxw), dim=3)
            real_samples_dir = os.path.join(svfolder, 'test-%d' % model_iter, 'real_{:0>7d}.png'.format(iter_num))
            vutils.save_image(re, real_samples_dir, normalize=False)

        meshnp_bxpx3 = mesh_vvbxpx3.detach().cpu().numpy()
        meshcolnp_bxpx3 = mc_vvbxpx3.detach().cpu().numpy()
        meshcolnp_bxpx3[meshcolnp_bxpx3 < 0] = 0
        meshcolnp_bxpx3[meshcolnp_bxpx3 > 1] = 1
        meshcolnp_bxpx3 = meshcolnp_bxpx3[..., ::-1]

        for j, meshnp_px3 in enumerate(meshnp_bxpx3):
            catname, md5name, numname = da['cate'][j], da['md5'][j], da['view0']['num'][j]
            mesh_dir = os.path.join(svfolder, 'test-%d' % model_iter,
                                    '{}/{}/{}.obj'.format(catname, md5name, numname))
            if not os.path.exists(os.path.join(svfolder, 'test-%d' % model_iter, catname, md5name)):
                os.makedirs(os.path.join(svfolder, 'test-%d' % model_iter, catname, md5name))
            tmo = meshnp_px3
            savemeshcolor(tmo, facenp_fx3, mesh_dir, meshcolnp_bxpx3[j])

    re = []
    for ca in cates:
        iou[ca] /= catenum[ca]
        print('{}, {}'.format(ca, iou[ca]))
        re.append([int(ca), iou[ca]])
    re = np.array(re, dtype=np.float32)
    path = os.path.join(svfolder, 'test-%d.npy' % test_iter_num)
    np.save(file=path, arr=re)

###############################################################
if __name__ == '__main__':
    test()