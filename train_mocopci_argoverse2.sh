#!/bin/bash
python train.py  \
--batch_size 2 \
--epochs 400 \
--data_root data/NL-Drive/train/ \
--scene_list data/NL-Drive/train_scene01_list.txt \
--npoints 8192 \
--save_dir experiments/argoverse2/ \
> .log_mocopci_argoverse2_train 2>&1
