#!/bin/bash
python train.py  \
--batch_size 2 \
--epochs 400 \
--data_root data/NL-Drive/train/ \
--scene_list data/NL-Drive/train_scene_list.txt \
--npoints 8192 \
--save_dir experiments/ko/ \
> .log_mocopci_ko_train 2>&1
