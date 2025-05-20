import torch
import torch.nn as nn
from torch.autograd import Function
import numpy as np
import random
import torch.optim as optim

from pytorch3d.loss import chamfer_distance
import emd_cuda
from torch.cuda import amp


def pdist2squared(x, y):
    xx = (x**2).sum(dim=1).unsqueeze(2)
    yy = (y**2).sum(dim=1).unsqueeze(1)
    dist = xx + yy - 2.0 * torch.bmm(x.permute(0, 2, 1), y)
    dist[dist != dist] = 0
    dist = torch.clamp(dist, 0.0, np.inf)
    return dist

class ClippedStepLR(optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, step_size, min_lr, gamma=0.1, last_epoch=-1):
        self.step_size = step_size
        self.min_lr = min_lr
        self.gamma = gamma
        super(ClippedStepLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        return [max(base_lr * self.gamma ** (self.last_epoch // self.step_size), self.min_lr)
                for base_lr in self.base_lrs]

def flow_criterion(pred_flow, flow, mask):
    loss = torch.mean(mask * torch.sum((pred_flow - flow) * (pred_flow - flow), dim=1) / 2.0)
    return loss

def chamfer_loss(pc1, pc2):
    '''
    Input:
        pc1: [B,3,N]
        pc2: [B,3,N]
    '''
    pc1 = pc1.permute(0,2,1)
    pc2 = pc2.permute(0,2,1)
    chamfer_dist, _ = chamfer_distance(pc1, pc2)
    return chamfer_dist

class EarthMoverDistanceFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, xyz1, xyz2):
        xyz1 = xyz1.contiguous()
        xyz2 = xyz2.contiguous()
        assert xyz1.is_cuda and xyz2.is_cuda, "Only support cuda currently."
        match = emd_cuda.approxmatch_forward(xyz1, xyz2)
        cost = emd_cuda.matchcost_forward(xyz1, xyz2, match)
        ctx.save_for_backward(xyz1, xyz2, match)
        return cost

    @staticmethod
    def backward(ctx, grad_cost):
        xyz1, xyz2, match = ctx.saved_tensors
        grad_cost = grad_cost.contiguous()
        grad_xyz1, grad_xyz2 = emd_cuda.matchcost_backward(grad_cost, xyz1, xyz2, match)
        return grad_xyz1, grad_xyz2


def earth_mover_distance(xyz1, xyz2, transpose=True):
    """Earth Mover Distance (Approx)

    Args:
        xyz1 (torch.Tensor): (b, 3, n1)
        xyz2 (torch.Tensor): (b, 3, n1)
        transpose (bool): whether to transpose inputs as it might be BCN format.
            Extensions only support BNC format.

    Returns:
        cost (torch.Tensor): (b)

    """
    if xyz1.dim() == 2:
        xyz1 = xyz1.unsqueeze(0)
    if xyz2.dim() == 2:
        xyz2 = xyz2.unsqueeze(0)
    if transpose:
        xyz1 = xyz1.transpose(1, 2)
        xyz2 = xyz2.transpose(1, 2)
    cost = EarthMoverDistanceFunction.apply(xyz1, xyz2)
    return cost


import torch
from torch.cuda import amp


class OptimizedEMDFunction(torch.autograd.Function):
    @staticmethod
    @amp.autocast(enabled=True)  # 启用自动混合精度
    def forward(ctx, xyz1, xyz2):
        # 预先分配内存并确保连续性
        xyz1 = xyz1.contiguous()
        xyz2 = xyz2.contiguous()

        assert xyz1.is_cuda and xyz2.is_cuda, "Only support cuda currently."

        # 并行计算match和cost
        match = emd_cuda.approxmatch_forward(xyz1, xyz2)
        cost = emd_cuda.matchcost_forward(xyz1, xyz2, match)

        ctx.save_for_backward(xyz1, xyz2, match)
        return cost

    @staticmethod
    @amp.autocast(enabled=True)
    def backward(ctx, grad_cost):
        xyz1, xyz2, match = ctx.saved_tensors
        grad_cost = grad_cost.contiguous()

        # 并行计算梯度
        grad_xyz1, grad_xyz2 = emd_cuda.matchcost_backward(grad_cost, xyz1, xyz2, match)
        return grad_xyz1, grad_xyz2


@torch.jit.script  # JIT编译加速
def prepare_inputs(xyz1, xyz2, transpose: bool):
    if xyz1.dim() == 2:
        xyz1 = xyz1.unsqueeze(0)
    if xyz2.dim() == 2:
        xyz2 = xyz2.unsqueeze(0)

    if transpose:
        # 使用contiguous优化内存访问
        xyz1 = xyz1.transpose(1, 2).contiguous()
        xyz2 = xyz2.transpose(1, 2).contiguous()
    return xyz1, xyz2


def optimized_emd(xyz1, xyz2, transpose=True):
    """High Performance Earth Mover Distance calculation

    Args:
        xyz1 (torch.Tensor): (b, 3, n1)
        xyz2 (torch.Tensor): (b, 3, n1)
        transpose (bool): whether to transpose inputs

    Returns:
        cost (torch.Tensor): (b)
    """
    # 使用JIT编译后的函数处理输入
    xyz1, xyz2 = prepare_inputs(xyz1, xyz2, transpose)

    # 计算批处理大小，用于并行处理
    batch_size = xyz1.size(0)

    # 使用优化后的CUDA实现
    with torch.cuda.amp.autocast():  # 启用自动混合精度
        cost = OptimizedEMDFunction.apply(xyz1, xyz2)

    return cost


# class OptimizedEMDFunction(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, xyz1, xyz2):
#         # Ensure contiguous memory layout
#         xyz1 = xyz1.contiguous()
#         xyz2 = xyz2.contiguous()
#
#         assert xyz1.is_cuda and xyz2.is_cuda, "Only support cuda currently."
#
#         # Pre-compute batch statistics if possible
#         batch_size = xyz1.size(0)
#
#         # Use built-in CUDA operations where possible
#         with amp.autocast(enabled=True):
#             match = emd_cuda.approxmatch_forward(xyz1, xyz2)
#             cost = emd_cuda.matchcost_forward(xyz1, xyz2, match)
#
#         ctx.save_for_backward(xyz1, xyz2, match)
#         return cost
#
#     @staticmethod
#     def backward(ctx, grad_cost):
#         xyz1, xyz2, match = ctx.saved_tensors
#         grad_cost = grad_cost.contiguous()
#
#         # Use mixed precision for backward pass
#         with amp.autocast(enabled=True):
#             grad_xyz1, grad_xyz2 = emd_cuda.matchcost_backward(grad_cost, xyz1, xyz2, match)
#
#         return grad_xyz1, grad_xyz2
#
#
# def optimized_emd(xyz1, xyz2, transpose=True):
#     """Optimized Earth Mover Distance calculation
#
#     Args:
#         xyz1 (torch.Tensor): (b, 3, n1)
#         xyz2 (torch.Tensor): (b, 3, n1)
#         transpose (bool): whether to transpose inputs
#
#     Returns:
#         cost (torch.Tensor): (b)
#     """
#     # Cache the original shapes
#     original_shape = xyz1.shape
#
#     # Batch processing if needed
#     if xyz1.dim() == 2:
#         xyz1 = xyz1.unsqueeze(0)
#     if xyz2.dim() == 2:
#         xyz2 = xyz2.unsqueeze(0)
#
#     if transpose:
#         # Use contiguous to optimize memory access
#         xyz1 = xyz1.transpose(1, 2).contiguous()
#         xyz2 = xyz2.transpose(1, 2).contiguous()
#
#     # Calculate cost using optimized implementation
#     cost = OptimizedEMDFunction.apply(xyz1, xyz2)
#
#     # For the loss calculation
#     return cost / original_shape[1]  # Normalize by point count

def EMD(pc1, pc2):
    '''
    Input:
        pc1: [1,3,M]
        pc2: [1,3,M]
    Ret:
        d: torch.float32
    '''
    pc1 = pc1.permute(0,2,1).contiguous()
    pc2 = pc2.permute(0,2,1).contiguous()
    d = earth_mover_distance(pc1, pc2, transpose=False)
    d = torch.mean(d)/pc1.shape[1]
    return d
