#!/bin/bash
python test.py  \
--batch_size 2 \
--data_root data/NL-Drive/test/ \
--scene_list data/NL-Drive/test_scene_list.txt \
--npoints 8192 \
--pretrain_model experiments/ko/ckpt_best_***.pth \
> .log_mocopci_ko_test 2>&1
