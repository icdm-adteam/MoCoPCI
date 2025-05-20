# MoCoPCI: Inter-frame Motion Correlation Guided Efficient 3D Point Cloud Interpolation
:wave: This is the official repository for *MoCoPCI: Inter-frame Motion Correlation Guided Efficient 3D Point Cloud Interpolation*.

## Introduction
LiDAR point cloud streams provide precise 3D mapping and detailed spatial information but limited temporal resolution due to hardware constraints.  Multi-frame point cloud interpolation, which synthesizes intermediate frames between consecutive point clouds, presents a promising solution. However, previous methods treat point cloud sequence interpolation as an iterative process, resulting in prolonged inference times and making it challenging to maintain the temporal consistency of the predicted point cloud sequence. To address these issues, we propose MoCoPCI, a novel *one-shot* sequence interpolation framework designed for *faster and more accurate* nonlinear predictions. The core of our approach is a motion correlation-based bidirectional flow prediction, which ensures temporal consistency in the predicted point cloud sequence while significantly improving efficiency. Additionally, we introduce a cross-attention mechanism based on extrapolation and injection to capture multi-scale motion information from consecutive frames. Meanwhile, a point-wise compensation mechanism refines the wrapped points, collectively enhancing the network’s ability to model nonlinear and non-rigid motion. Experiments on benchmark outdoor datasets show that MoCoPCI achieves state-of-the-art performance, reducing the Chamfer distance by up to 24.35\% and providing a 2× speedup.


## Network Structure
<p align="center">
  <img src="resources/network.png" width="80%"/>
</p>

## Requirements

Create a conda environment:
```
conda env create -f environment.yaml
conda activate mocopci

cd models/EMD/
python setup.py install
cp build/lib.linux-x86_64-cpython-39/emd_cuda.cpython-39-x86_64-linux-gnu.so .

cd ../pointnet2/
python setup.py install

cd ../../
```

## Dataset preparation

We utilize the NL-Drive dataset, which, following the same processing protocol as FastPCI and NeuralPCI, integrates data from KITTI Odometry, Argoverse2 Sensor, and NuScenes. 
Please download the NL-Drive dataset at [here](https://tongjieducn-my.sharepoint.com/:f:/g/personal/zhengzehan_tongji_edu_cn/Ej4AiwgJWp1MsAFwtWcxIFkBPDwsCW_3bWSRlpYf4XZw-w)
. And put the NL-Drive dataset into `data/NL-Drive` .
We provide the split list of three datasets in `./data/NL-Drive/`.

## Instructions to training and testing

### Training

Training on KITTI Odometry dataset, Argoverse 2 sensor dataset, Nuscenes dataset, run separately:
```
bash train_mocopci_kitti.sh
bash train_mocopci_argoverse2.sh
bash train_mocopci_nuscenes.sh
```

### Testing

Testing on KITTI Odometry dataset, Argoverse 2 sensor dataset, Nuscenes dataset, run separately:
```
bash test_mocopci_kitti.sh
bash test_mocopci_argoverse2.sh
bash test_mocopci_nuscenes.sh
```




## Acknowledgments
We thank the authors of
- [FastPCI](https://github.com/genuszty/FastPCI)
- [PointPWC-Net](https://github.com/DylanWusee/PointPWC)
- [PointINet](https://github.com/ispc-lab/PointINet)
- [NeuralPCI](https://github.com/ispc-lab/NeuralPCI)
- [EMA-VFI](https://github.com/mcg-nju/ema-vfi)

for open sourcing their methods.
