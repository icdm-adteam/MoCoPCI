from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import os
import numpy as np
import sys
sys.path.append('/home/ubuntu/ypf/MoCoPCI')
from data.no_norm_datasets import NLDriveDataset
from models.m_models.mocopci import MoCoPCI
from models.utils import chamfer_loss
import time
from tqdm import tqdm
import argparse

def parse_args(train_target='scene01'):
    parser = argparse.ArgumentParser(description='MoCoPCI')
    # training setting
    parser.add_argument('--multi-gpu', type=str, default='0, 1')
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=400)
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate.')
    parser.add_argument('--weight_decay', type=float, default=0.0001, help='Weight decay.')
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--resume', type=bool, default=False, help='whether continue the training')
    parser.add_argument('--save_dir', type=str, default='/home/ubuntu/ypf/MoCoPCI/outputs')
    # dataset setting
    parser.add_argument('--data_root', type=str, default='')
    parser.add_argument('--scene_list', type=str, default='')
    parser.add_argument('--interval', type=int, default=4)
    parser.add_argument('--num_frames', type=int, default=4)
    parser.add_argument('--npoints', type=int, default=8192)
    parser.add_argument('--t_begin', type=float, default=0., help='Time stamp of the first input frame.')
    parser.add_argument('--t_end', type=float, default=1., help='Time stamp of the last input frame.')
    parser.add_argument('--experiment_scene', type=str, default='02')
    parser.add_argument('--experiment_name', type=str, default='frame_times01_no_gru')
    return parser.parse_args()


def init_weights(m):
    if isinstance(m, nn.Conv1d) or isinstance(m, nn.Conv2d):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.0)
    elif isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1.0)
        m.bias.data.fill_(0.0)


def get_timestamp(args):
    time_seq = [t for t in np.linspace(args.t_begin, args.t_end, args.num_frames)]
    t_left = time_seq[args.num_frames // 2 - 1]
    t_right = time_seq[args.num_frames // 2]
    time_intp = [t for t in np.linspace(t_left, t_right, args.interval + 1)]
    time_intp = time_intp[1:-1]
    return time_seq, time_intp

def train(args):
    LEARNING_RATE_CLIP = 0.00005
    train_start_time = datetime.now().strftime('%Y-%m-%d-%H-%M')
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu if args.multi_gpu is None else '0,1,2,3'
    scene_list = '/home/ubuntu/FastPCI/data/NL-Drive/train_scene' + args.experiment_scene + '_list.txt'
    train_dataset = NLDriveDataset(args.data_root, scene_list, args.npoints, args.interval, args.num_frames)

    train_loader = DataLoader(train_dataset,
                              batch_size=args.batch_size,
                              num_workers=8,
                              shuffle=True,
                              pin_memory=True,
                              drop_last=True)

    net = MoCoPCI()
    if torch.cuda.is_available():
        num_devices = torch.cuda.device_count()
        print("Obtain the number of available CUDA devices:", num_devices)
        if num_devices > 1:
            torch.backends.cudnn.benchmark = True
            net = torch.nn.DataParallel(net)
            net.cuda()
        else:
            net.cuda()
    else:
        raise EnvironmentError("CUDA is not available.")
    total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print('the number of network parameters: {}'.format(total_params))

    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, net.parameters()), lr=args.lr, betas=(0.9, 0.999),
                            eps=1e-08, weight_decay=args.weight_decay)

    if args.resume:
        experiments = 'xxx'
        checkpoint = torch.load(experiments)
        net.module.load_state_dict(checkpoint['net'])
        start_epoch = checkpoint['epoch']
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.8, last_epoch=start_epoch - 1)
        scheduler.load_state_dict(checkpoint['scheduler'])
    else:
        start_epoch = 0
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.8, last_epoch=start_epoch - 1)

    best_train_loss = float('inf')
    best_train_loss_ls = float('inf')
    _, time_inp = get_timestamp(args)

    for epoch in range(start_epoch, args.epochs):
        lr = max(optimizer.param_groups[0]['lr'], LEARNING_RATE_CLIP)
        print('current learning rate:', lr)

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        optimizer.zero_grad()

        start_time = time.time()
        count = 0
        total_loss = 0
        loss_final = 0
        loss_straight_f = 0
        loss_straight_b = 0
        loss_multi_f = 0
        loss_multi_b = 0

        pbar = tqdm(enumerate(train_loader))
        for i, (input, gt) in pbar:
            for i in range(len(input)):
                input[i] = input[i].permute(0, 2, 1).cuda().contiguous().float()
            for i in range(len(gt)):
                gt[i] = gt[i].permute(0, 2, 1).cuda().contiguous().float()

            net.train()
            frames_lst_f, frames_lst_b, gt_frame, out_lst = net( input[1], input[2], gt, time_inp, True)

            torch.cuda.synchronize()

            loss_f = 0
            loss_m_f = 0
            loss_m_b = 0
            loss_s_f = 0
            loss_s_b = 0
            for frames, gts in zip(out_lst, gt):
                loss_f += chamfer_loss(frames.permute(0, 2, 1), gts)
            alpha = [1.0, 0.8, 0.4, 0.2]
            for frames_f, frames_b, gts in zip(frames_lst_f, frames_lst_b, gt_frame):
                loss_s_f += 0.5*chamfer_loss(frames_f[0].permute(0, 2, 1), gts[0])
                loss_s_b += 0.5*chamfer_loss(frames_b[0].permute(0, 2, 1), gts[0])

                loss_s_f += 0.5*chamfer_loss(frames_f[1].permute(0, 2, 1), gts[0])
                loss_s_b += 0.5*chamfer_loss(frames_b[1].permute(0, 2, 1), gts[0])

                alpha = [1.0, 0.8, 0.4, 0.2]  # , 0.2
                multiscaleloss_f = 0
                multiscaleloss_b = 0
                m_b = [0, 0, 0, 0]
                for l in range(len(alpha) - 1):
                    multiscaleloss_f += alpha[l + 1] * chamfer_loss(frames_f[l+2].permute(0, 2, 1), gts[l + 1])
                    multiscaleloss_b += alpha[l + 1] * chamfer_loss(frames_b[l+2].permute(0, 2, 1), gts[l + 1])
                loss_m_f += multiscaleloss_f
                loss_m_b += multiscaleloss_b

            losssum = loss_f + (loss_s_f+loss_s_b)/2 + 0.25 * loss_m_b + 0.25 * loss_m_f
            losssum.backward()
            if epoch < 30:
                clip_value = 2.0
            else:
                clip_value = 2.0
            torch.nn.utils.clip_grad_norm_(net.parameters(), clip_value)
            optimizer.step()
            optimizer.zero_grad()

            count += 1
            loss_final += loss_f.item()
            loss_straight_f += loss_s_f.item()
            loss_straight_b += loss_s_b.item()
            loss_multi_f += loss_m_f.item()
            loss_multi_b += loss_m_b.item()
            if i % 10 == 0:
                print(
                    'Train Epoch:{}[{}/{}({:.0f}%)]'
                    '\tloss_final: {:.6f}\t'
                    '\tloss_straight_forward: {:.6f}\t'
                    '\tloss_straight_backward: {:.6f}\t'
                    '\tloss_multi_forward: {:.6f}\t'
                    '\tloss_multi_backward: {:.6f}\t'
                    .format(
                        epoch + 1, i,len(train_loader), 100. * i / len(train_loader),
                        loss_final.item(),
                        loss_straight_f.item(),
                        loss_straight_b.item(),
                        loss_multi_f.item(),
                        loss_multi_b.item(),

                    ))

        scheduler.step()
        loss_final = loss_final / count
        loss_straight_f = loss_straight_f / count
        loss_straight_b = loss_straight_b / count
        loss_multi_f = loss_multi_f / count
        loss_multi_b = loss_multi_b / count
        print('Epoch ', epoch + 1, 'finished ', 'loss_f = ', loss_final, 'loss_s_f = ', loss_straight_f, 'loss_s_b = ', loss_straight_b, 'loss_m_f =', loss_multi_f, 'loss_m_b = ', loss_multi_b)

        if loss_final < best_train_loss_ls:
            best_train_loss_ls = loss_final
            if args.multi_gpu is not None:
                best_train_loss = loss_final
                checkpoint = {
                    'net': net.module.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'epoch': epoch
                }
                save_dir = os.path.join(args.save_dir, train_start_time)
                if not os.path.isdir(save_dir):
                    os.mkdir(save_dir)
                torch.save(checkpoint, save_dir + '/ckpt_best_' + str(epoch) + '_' + str(round(best_train_loss_ls,3)) + '.pth')
            else:
                best_train_loss = total_loss
                checkpoint = {
                    'net': net.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'epoch': epoch
                }
                save_dir = os.path.join(args.save_dir, args.experiment_scene, train_start_time + '-' + args.experiment_name)
                if not os.path.isdir(save_dir):
                    os.mkdir(save_dir)
                torch.save(checkpoint, save_dir + '/ckpt_best_' + str(epoch) + '_' + str(round(best_train_loss_ls, 3)) + '.pth')

        print('Best train loss: {:.4f} Best targer loss: {:.4f}'.format(best_train_loss, best_train_loss_ls))
        one_epoch_time = time.time() - start_time
        print('epoch:', epoch, 'one_epoch_time:', one_epoch_time)


if __name__ == '__main__':
    args = parse_args()
    train(args)