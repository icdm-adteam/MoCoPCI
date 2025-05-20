#!/bin/bash
python train.py  \
--batch_size 2 \
--epochs 250 \
--gpu 0 \
--data_root data/NL-Drive/train/ \
--scene_list data/NL-Drive/train_scene02_list.txt \
--npoints 8192 \
--save_dir experiments/nuscenes/ \
> .log_mocopci_nuscenes_train 2>&1
