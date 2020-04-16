# DIB-Render
This is the official inference code for:

#### Learning to Predict 3D Objects with an Interpolation-based Differentiable Renderer (NeurIPS 2019)

[Wenzheng Chen](http://www.cs.toronto.edu/~wenzheng/), [Jun Gao\*](http://www.cs.toronto.edu/~jungao/), [Huan Ling\*](http://www.cs.toronto.edu/~linghuan/), [Edward J. Smith\*](), [Jaakko Lehtinen](https://users.aalto.fi/~lehtinj7/), [Alec Jacobson](https://www.cs.toronto.edu/~jacobson/), [Sanja Fidler](http://www.cs.toronto.edu/~fidler/)


**[[Paper](https://arxiv.org/abs/1908.01210)]  [[Project Page](https://nv-tlabs.github.io/DIB-R/)]**

## Usage


#### Install dependencies

This code requires PyTorch 1.1.0 and python 3+. Please install dependencies by
```bash
pip install -r requirements.txt
```

### Compile the DIB-Render
```bash
cd dib-render/cuda_dib_render
python build.py install
```


#### Inference
``` bash
python test-all.py \
 --g_model_dir ./checkpoints/g_model.pth \
 --svfolder ./prediction \
 --data_folder ./dataset \
 --filelist ./test_list.txt
```

To get the evaluation IOU, please first download the tool [Binvox](https://www.patrickmin.com/binvox/) and install it's dependencies,

Voxelize the prediction using Binvox
```bash
python voxelization.py  --folder ./prediction
```

To evaluate the IOU, please first install binvox-rw-py following this [Link](https://github.com/dimatura/binvox-rw-py), then run the script
```bash
python check_iou.py --folder ./prediction  --gt_folder ./dataset 
```

To get the boundary F-score, please run the following script
```bash
python check_chamfer.py --folder ./prediction  --gt_folder ./dataset 
```

### Ciatation
If you use the code, please cite our paper:
```
@inproceedings{chen2019dibrender,
title={Learning to Predict 3D Objects with an Interpolation-based Differentiable Renderer},
author={Wenzheng Chen and Jun Gao and Huan Ling and Edward Smith and Jaakko Lehtinen and Alec Jacobson and Sanja Fidler},
booktitle={Advances In Neural Information Processing Systems},
year={2019}
}
```
