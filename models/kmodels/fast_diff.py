import torch.nn as nn
import torch
import math
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import torch.nn.functional as F
from pytorch3d.ops import knn_points
from models.pointnet2.pointnet2_utils import gather_operation, grouping_operation, furthest_point_sample
from models.pointT_layer2 import FlowRefineNet, TransformerBlock, square_distance  # , index_points
from models.pointconv_util import CrossLayerLightFeatCosine as CrossLayer, FlowEmbeddingLayer, BidirectionalLayerFeatCosine
from models.pointconv_util import SceneFlowEstimatorResidual
from models.pointconv_util import index_points_gather as index_points, index_points_group, Conv1d, square_distance, knn_point_cosine, knn_point
import time# from models.amformer import MotionFormer
# from models.common import gather_points
# from models.utils import chamfer_loss

scale = 1.0


class RecurrentUnitAtt(nn.Module):
    def __init__(self, iters, feat_ch, feat_new_ch, latent_ch, cross_mlp1, cross_mlp2,
                 weightnet=8, flow_channels=[64, 64], flow_mlp=[64, 64]):
        super(RecurrentUnitAtt, self).__init__()
        flow_nei = 32
        self.iters = 3
        self.scale = 1.0

        self.bid = BidirectionalLayerFeatCosine(flow_nei, feat_new_ch + feat_ch, cross_mlp1)
        self.fe = FlowEmbeddingLayer(flow_nei, cross_mlp1[-1], cross_mlp2)
        flow_channels = [latent_ch, latent_ch]
        self.cross_block = MotionFormerBlockTU(
            dim=feat_ch, motion_dim=feat_ch//4, flow_feats=flow_channels,
            mlp_ratio=4., qkv_bias=True, qk_scale=None,
            drop=0.0, attn_drop=0.0, drop_path=0.04, norm_layer=nn.BatchNorm1d)
        self.flow = SceneFlowGRUResidual(latent_ch, cross_mlp2[-1] + feat_ch, channels=flow_channels, mlp=flow_mlp)
        self.downsample = Conv1d(latent_ch, 64)
        self.warping = PointWarping()

    def forward(self, pc1, pc2, feat1_new, feat2_new, feat1, feat2, up_flow, up_feat):
        c_feat1 = torch.cat([feat1, feat1_new], dim=1)
        c_feat2 = torch.cat([feat2, feat2_new], dim=1)

        flows = []
        for i in range(self.iters):
            pc2_warp = self.warping(pc1, pc2, up_flow)
            feat1_new, feat2_new = self.bid(pc1, pc2_warp, c_feat1, c_feat2, feat1, feat2)
            fe = self.fe(pc1, pc2_warp, feat1_new, feat2_new, feat1, feat2)
            # new_feat1 = torch.cat([feat1, fe], dim=1)
            # feat_flow, flow = self.flow(pc1, up_feat, new_feat1, up_flow)
            x = torch.cat([feat1_new, fe, feat2_new], dim=0)
            feat_flow, flow = self.cross_block(x, pc1.shape[-1], pc1.shape[0])
            up_flow = flow
            up_feat = feat_flow
            c_feat1 = torch.cat([feat1, feat1_new], dim=1)
            c_feat2 = torch.cat([feat2, feat2_new], dim=1)
            flows.append(flow)
        feat_flow = self.downsample(feat_flow)
        return flows, feat1_new, feat2_new, feat_flow, fe

class RecurrentUnit(nn.Module):
    def __init__(self, iters, feat_ch, feat_new_ch, latent_ch, cross_mlp1, cross_mlp2,
                 weightnet=8, flow_channels=[64, 64], flow_mlp=[64, 64]):
        super(RecurrentUnit, self).__init__()
        flow_nei = 32
        self.iters = 3
        self.scale = 1.0

        self.bid = BidirectionalLayerFeatCosine(flow_nei, feat_new_ch + feat_ch, cross_mlp1)
        self.fe = FlowEmbeddingLayer(flow_nei, cross_mlp1[-1], cross_mlp2)
        flow_channels = [latent_ch, latent_ch]
        self.cross_block = MotionFormerBlockTU(
            dim=feat_ch, motion_dim=feat_ch//4, flow_feats=flow_channels,
            mlp_ratio=4., qkv_bias=True, qk_scale=None,
            drop=0.0, attn_drop=0.0, drop_path=0.04, norm_layer=nn.BatchNorm1d)
        self.flow = SceneFlowGRUResidual(latent_ch, cross_mlp2[-1] + feat_ch, channels=flow_channels, mlp=flow_mlp)
        self.downsample = Conv1d(latent_ch, 64)
        self.warping = PointWarping()

    def forward(self, pc1, pc2, feat1_new, feat2_new, feat1, feat2, up_flow, up_feat):
        c_feat1 = torch.cat([feat1, feat1_new], dim=1)
        c_feat2 = torch.cat([feat2, feat2_new], dim=1)

        flows = []
        for i in range(self.iters):
            pc2_warp = self.warping(pc1, pc2, up_flow)
            feat1_new, feat2_new = self.bid(pc1, pc2_warp, c_feat1, c_feat2, feat1, feat2)
            fe = self.fe(pc1, pc2_warp, feat1_new, feat2_new, feat1, feat2)
            new_feat1 = torch.cat([feat1, fe], dim=1)
            feat_flow, flow = self.flow(pc1, up_feat, new_feat1, up_flow)
            # x = torch.cat([feat1_new, fe, feat2_new], dim=0)
            # feat_flow, flow = self.cross_block(x, pc1.shape[-1], pc1.shape[0])
            up_flow = flow
            up_feat = feat_flow
            c_feat1 = torch.cat([feat1, feat1_new], dim=1)
            c_feat2 = torch.cat([feat2, feat2_new], dim=1)
            flows.append(flow)
        feat_flow = self.downsample(feat_flow)
        return flows, feat1_new, feat2_new, feat_flow, fe


class GRUMappingNoGCN(nn.Module):
    def __init__(self, nsample, in_channel, latent_channel, mlp, mlp2=None, bn=False, use_leaky=True,
                 return_inter=False, radius=None, use_relu=False):
        super(GRUMappingNoGCN, self).__init__()
        self.radius = radius
        self.nsample = nsample
        self.return_inter = return_inter
        self.mlp_r_convs = nn.ModuleList()
        self.mlp_z_convs = nn.ModuleList()
        self.mlp_h_convs = nn.ModuleList()
        self.mlp_r_bns = nn.ModuleList()
        self.mlp_z_bns = nn.ModuleList()
        self.mlp_h_bns = nn.ModuleList()
        self.mlp2 = mlp2
        self.bn = bn
        self.use_relu = use_relu

        last_channel = in_channel + 3

        self.fuse_r = nn.Conv1d(latent_channel, mlp[0], 1, bias=False)
        self.fuse_r_o = nn.Conv2d(latent_channel, mlp[0], 1, bias=False)
        self.fuse_z = nn.Conv1d(latent_channel, mlp[0], 1, bias=False)

        for out_channel in mlp:
            self.mlp_r_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_z_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_h_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            if bn:
                self.mlp_r_bns.append(nn.BatchNorm2d(out_channel))
                self.mlp_z_bns.append(nn.BatchNorm2d(out_channel))
                self.mlp_h_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel

        if mlp2:
            self.mlp2 = nn.ModuleList()
            for out_channel in mlp2:
                self.mlp2.append(Conv1d(last_channel, out_channel, 1, bias=False, bn=bn))
                last_channel = out_channel

        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU(inplace=True) if not use_leaky else nn.LeakyReLU(0.1, inplace=True)

        if radius is not None:
            self.queryandgroup = pointnet2_utils.QueryAndGroup(radius, nsample, True)

    def forward(self, xyz1, xyz2, points1, points2):
        '''
        add fuse_r_o
        xyz1: joints [B, 3, N1]
        xyz2: local points [B, 3, N2]
        points1: joints features [B, C, N1]
        points2: local features [B, C, N2]
        '''
        B, C, N1 = xyz1.shape
        _, _, N2 = xyz2.shape
        _, D1, _ = points1.shape
        _, D2, _ = points2.shape
        xyz1 = xyz1.permute(0, 2, 1)
        xyz2 = xyz2.permute(0, 2, 1)
        # points1 = points1.permute(0, 2, 1)
        # points2 = points2.permute(0, 2, 1)

        if self.radius is None:
            sqrdists = square_distance(xyz1, xyz2)
            dists, knn_idx = torch.topk(sqrdists, self.nsample, dim=-1, largest=False, sorted=False)
            neighbor_xyz = index_points_group(xyz2, knn_idx)
            direction_xyz = neighbor_xyz - xyz1.view(B, N1, 1, C)

            grouped_points2 = index_points_group(points2.permute(0, 2, 1), knn_idx)  # B, N1, nsample, D2
            new_points = torch.cat([grouped_points2, direction_xyz], dim=-1)  # B, N1, nsample, D1+D2+3
            new_points = new_points.permute(0, 3, 2, 1)  # [B, D2+3, nsample, N1]

        else:
            new_points = self.queryandgroup(xyz2.contiguous(), xyz1.contiguous(), points2.contiguous())
            new_points = new_points.permute(0, 1, 3, 2)

        point1_graph = points1

        # r
        r = new_points
        for i, conv in enumerate(self.mlp_r_convs):
            r = conv(r)
            if i == 0:
                grouped_points1 = self.fuse_r(point1_graph)
                r = r + grouped_points1.view(B, grouped_points1.size(1), 1, N1).repeat(1, 1, self.nsample, 1)
            if self.bn:
                r = self.mlp_r_bns[i](r)
            if i == len(self.mlp_r_convs) - 1:
                r = self.sigmoid(r)
            else:
                r = self.relu(r)

        # z
        z = new_points
        for i, conv in enumerate(self.mlp_z_convs):
            z = conv(z)
            if i == 0:
                grouped_points1 = self.fuse_z(point1_graph)
                z = z + grouped_points1.view(B, grouped_points1.size(1), 1, N1).repeat(1, 1, self.nsample, 1)
            if self.bn:
                z = self.mlp_z_bns[i](z)
            if i == len(self.mlp_z_convs) - 1:
                z = self.sigmoid(z)
                # print('sigmoid', z.shape)
            else:
                z = self.relu(z)
                # print('relu', z.shape)

            if i == len(self.mlp_z_convs) - 2:
                z = torch.max(z, -2)[0].unsqueeze(-2)
                # print('max', z.shape)

        z = z.squeeze(-2)

        point1_graph_expand = point1_graph.view(B, point1_graph.size(1), 1, N1).repeat(1, 1, self.nsample, 1)
        point1_expand = r * point1_graph_expand
        point1_expand = self.fuse_r_o(point1_expand)

        h = new_points
        for i, conv in enumerate(self.mlp_h_convs):
            h = conv(h)
            if i == 0:
                h = h + point1_expand
            if self.bn:
                h = self.mlp_h_bns[i](h)
            if i == len(self.mlp_h_convs) - 1:
                #
                if self.use_relu:
                    h = self.relu(h)
                else:
                    h = self.tanh(h)
            else:
                h = self.relu(h)
            if i == len(self.mlp_h_convs) - 2:
                h = torch.max(h, -2)[0].unsqueeze(-2)

        h = h.squeeze(-2)

        new_points = (1 - z) * points1 + z * h

        if self.mlp2:
            for _, conv in enumerate(self.mlp2):
                new_points = conv(new_points)

        return new_points


class SceneFlowGRUResidual(nn.Module):

    def __init__(self, feat_ch, cost_ch, flow_ch=3, channels=[64, 64], mlp=[64, 64], neighbors=9, clamp=[-200, 200],
                 use_leaky=True):
        super(SceneFlowGRUResidual, self).__init__()
        self.clamp = clamp
        self.use_leaky = use_leaky
        self.pointconv_list = nn.ModuleList()
        # last_channel = feat_ch + cost_ch

        self.gru = GRUMappingNoGCN(neighbors, in_channel=cost_ch, latent_channel=feat_ch, mlp=channels)

        # self.mlp_convs = nn.ModuleList()
        # for _, ch_out in enumerate(mlp):
        #     self.mlp_convs.append(Conv1d(last_channel, ch_out))
        #     last_channel = ch_out

        self.fc = nn.Conv1d(channels[-1], 3, 1)

    def forward(self, xyz, feats, cost_volume, flow=None):
        '''
        feats: B C1 N
        cost_volume: B C2 N
        flow: B 3 N
        '''
        # new_points = torch.cat([feats, cost_volume], dim = 1)

        feats_new = self.gru(xyz, xyz, feats, cost_volume)

        new_points = feats_new - feats
        # for conv in self.mlp_convs:
        #     new_points = conv(new_points)

        flow_local = self.fc(new_points).clamp(self.clamp[0], self.clamp[1])

        if flow is None:
            flow = flow_local
        else:
            flow = flow_local + flow
        return feats_new, flow

class FlowRefineNet_Unet(nn.Module):
    def __init__(self, weightnet=8):
        super(FlowRefineNet_Unet, self).__init__()
        feat_nei = 32
        self.rlevel0 = Conv1d(32, 64)
        self.level1 = PointConvD(2048, feat_nei, 64 + 3, 64, weightnet=weightnet)
        self.level2 = PointConvD(512, feat_nei, 128 + 3, 128, weightnet=weightnet)
        self.level3 = PointConvD(256, feat_nei, 256 + 3, 256, weightnet=weightnet)
        self.shape1 = TransformerBlock(64, 64)
        self.shape2 = TransformerBlock(128, 128)
        self.shape3 = TransformerBlock(256, 256)
        self.upsample = UpsampleFlow()
        self.warping = PointWarping()
        # self.conv4_3 = Conv1d(512, 256)
        self.conv3_2 = Conv1d(256, 128)
        self.conv2_1 = Conv1d(128, 64)
        self.pred = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 3))

    def forward(self, encoder, pc1s, pc2s, feat1s, flows, k, t):
        warped_pc1t_l2 = pc1s[2] + flows[2] * t
        warped_pc1t_l1 = pc1s[1] + flows[1] * t
        warped_pc1t = pc1s[0] + flows[0] * t
        warped_pc2t = pc2s[0] + flows[0] * (1 - t)

        # fused_initial = self.fusion0(warped_pc1t, warped_pc2t, k, t)
        _, _, N = flows[0].shape
        # fps_idx = furthest_point_sample(flows[2].permute(0, 2, 1).contiguous(), int(N / 32))
        # flow3 = (index_points_gather(flows[2].permute(0, 2, 1).contiguous(), fps_idx)).permute(0, 2, 1)
        warped_pc1t_l3 = pc1s[3] + flows[3] * t
        # feat1s_0 = self.rlevel0(feat1s[0])
        warped_feat1t_l0 = feat1s[0] + (
            F.interpolate(flows[0].permute(0, 2, 1), size=feat1s[0].size(1), mode="area")).permute(0, 2, 1) * t
        warped_feat1t_l1 = feat1s[1] + (
            F.interpolate(flows[1].permute(0, 2, 1), size=feat1s[1].size(1), mode="area")).permute(0, 2, 1) * t
        warped_feat1t_l2 = feat1s[2] + (
            F.interpolate(flows[2].permute(0, 2, 1), size=feat1s[2].size(1), mode="area")).permute(0, 2, 1) * t
        warped_feat1t_l3 = feat1s[3] + (
            F.interpolate(flows[3].permute(0, 2, 1), size=feat1s[3].size(1), mode="area")).permute(0, 2, 1) * t

        # shape encode
        # fused_down1, fused_feat1 = self.level1(warped_pc1t, warped_feat1t_l0)
        warped_feat1t_l0 = self.rlevel0(warped_feat1t_l0)
        # fused_down1, fused_feat1, _ = encoder.level1(warped_pc1t, warped_feat1t_l0)
        fused_down1, fused_feat1 = self.level1(warped_pc1t, warped_feat1t_l0)
        fea_shape1 = self.shape1(fused_feat1.permute(0, 2, 1), fused_down1.permute(0, 2, 1))  # [B,64,2048]
        # fused_down2, fused_feat2, _ = encoder.level2(fused_down1, torch.cat([warped_feat1t_l1, fea_shape1], dim=1))
        fused_down2, fused_feat2 = self.level2(fused_down1, torch.cat([warped_feat1t_l1, fea_shape1], dim=1))
        fea_shape2 = self.shape2(fused_feat2.permute(0, 2, 1), fused_down2.permute(0, 2, 1))  # [B,128,512]
        # fused_down3, fused_feat3, _ = encoder.level3(fused_down2, torch.cat([warped_feat1t_l2, fea_shape2], dim=1))
        fused_down3, fused_feat3 = self.level3(fused_down2, torch.cat([warped_feat1t_l2, fea_shape2], dim=1))

        fea_shape3 = self.shape3(fused_feat3.permute(0, 2, 1), fused_down3.permute(0, 2, 1))  # [B,256,256]

        up_pc2 = self.upsample(fused_down2, fused_down3, warped_pc1t_l3)
        up_feat2 = self.upsample(fused_down2, fused_down3, fea_shape3)
        up_feat2 = self.shape2((self.conv3_2(up_feat2)).permute(0, 2, 1), up_pc2.permute(0, 2, 1))

        up_pc1 = self.upsample(fused_down1, fused_down2, warped_pc1t_l2)
        up_feat1 = self.upsample(fused_down1, fused_down2, fea_shape2)
        up_feat1 = self.shape1((self.conv2_1(up_feat1)).permute(0, 2, 1), up_pc1.permute(0, 2, 1))

        up_feat0 = self.upsample(warped_pc1t, fused_down1, fea_shape1)
        refine_out = (self.pred(up_feat0.permute(0, 2, 1))).permute(0, 2, 1)

        return warped_pc1t, warped_pc2t, warped_pc1t_l1, warped_pc1t_l2, warped_pc1t_l3, refine_out


class PointConvEncoder(nn.Module):
    def __init__(self, weightnet=8):
        super(PointConvEncoder, self).__init__()
        feat_nei = 32

        self.level0_lift = Conv1d(3, 32)
        # self.level0 = Conv1d(32, 32)
        self.level0 = PointConv(feat_nei, 32 + 3, 32, weightnet = weightnet) # out
        self.level0_1 = Conv1d(32, 64)

        # self.level1 = PointConvD(2048, feat_nei, 64 + 3, 64, weightnet=weightnet)
        self.level1 = PointConvDE(2048, feat_nei, 64 + 3, 64, weightnet = weightnet)
        self.level1_0 = Conv1d(64, 64)  # out
        self.level1_1 = Conv1d(64, 128)

        self.level2 = PointConvDE(512, feat_nei, 128 + 3, 128, weightnet=weightnet)
        self.level2_0 = Conv1d(128, 128)  # out
        self.level2_1 = Conv1d(128, 256)

        self.level3 = PointConvDE(256, feat_nei, 256 + 3, 256, weightnet=weightnet)
        self.level3_0 = Conv1d(256, 256)  # out
        self.level3_1 = Conv1d(256, 512)

        self.level4 = PointConvDE(64, feat_nei, 512 + 3, 256, weightnet=weightnet)  # out

    def forward(self, xyz, color):
        # 局部邻域特征提取  动态卷积权重计算 特征加权求和 特征投影 批归一化和激活
        pc_0 = xyz
        feat_l0 = self.level0_lift(color)
        feat_l0 = self.level0(pc_0, feat_l0)
        feat_l0_1 = self.level0_1(feat_l0)

        # l1
        # 下采样 局部邻域特征提取 动态卷积权重计算 特征更新和下采样点特征计算 Batch Normalization 和激活
        # pc_l1, feat_l1, fps_l1 = self.level1(xyz, feat_l0_1)
        pc_l1, feat_l1, fps_l1 = self.level1(pc_0, feat_l0_1)
        feat_l1 = self.level1_0(feat_l1)  # 线性变化
        feat_l1_2 = self.level1_1(feat_l1)  # 升维

        # l2
        # 下采样 局部邻域特征提取 动态卷积权重计算 特征更新和下采样点特征计算 Batch Normalization 和激活
        # pc_l2, feat_l2, fps_l2 = self.level2(pc_l1, feat_l1_2)
        pc_l2, feat_l2, fps_l2 = self.level2(pc_l1, feat_l1_2)
        feat_l2 = self.level2_0(feat_l2)  # 线性变化
        feat_l2_3 = self.level2_1(feat_l2)  # 升维

        # l3
        # 下采样 局部邻域特征提取 动态卷积权重计算 特征更新和下采样点特征计算 Batch Normalization 和激活
        # pc_l3, feat_l3, fps_l3 = self.level3(pc_l2, feat_l2_3)
        pc_l3, feat_l3, fps_l3 = self.level3(pc_l2, feat_l2_3)
        feat_l3 = self.level3_0(feat_l3)  # 线性变化
        feat_l3_4 = self.level3_1(feat_l3)  # 升维

        # l4
        # pc_l4, feat_l4, fps_l4 = self.level4(pc_l3, feat_l3_4)
        pc_l4, feat_l4, fps_l4 = self.level4(pc_l3, feat_l3_4)
        # 最后一次下采样 局部邻域特征提取 动态卷积权重计算 特征更新和下采样点特征计算 Batch Normalization 和激活

        return [xyz, pc_l1, pc_l2, pc_l3, pc_l4], \
            [feat_l0, feat_l1, feat_l2, feat_l3, feat_l4], \
            [fps_l1, fps_l2, fps_l3, fps_l4]


class MotionFormerBlockT(nn.Module):
    def __init__(self, dim, motion_dim, num_heads=8, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.PReLU, norm_layer=nn.BatchNorm1d):
        super().__init__()

        # self.shift_size = shift_size
        # if not isinstance(self.shift_size, (tuple, list)):
        #    self.shift_size = to_2tuple(shift_size)
        # self.bidirectional = bidirectional
        self.norm1 = norm_layer(dim)
        self.attn = InterFrameAttentionT(
            dim,
            motion_dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.trans_block = Mlp(in_features=dim*2, hidden_features=mlp_hidden_dim, out_features=64, act_layer=act_layer, drop=drop)
        self.apply(self._init_weights)
        self.mapping_xyz = nn.Linear(64, 3)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        # elif isinstance(m, nn.Conv1d):

    def forward(self, x, N, B):
        nwB = x.shape[0]
        x_norm = (self.norm1(x)).permute(0, 2, 1)  # B,C,N
        x_reverse = torch.cat([x_norm[nwB // 2:], x_norm[:nwB // 2]])
        x_appearence = self.attn(x_norm, x_reverse, N)  # B,C,N
        x_norm = x_norm + self.drop_path(x_appearence)
        x_back = x_norm  # .view(2*B, N, -1)
        x_back = self.norm2(x_back.permute(0, 2, 1))
        x_back = self.drop_path(self.mlp(x_back.permute(0, 2, 1), N))
        x = x + x_back.permute(0, 2, 1)  # self.drop_path(self.mlp(self.norm2(x_back.permute(0,2,1)), N))
        x_f = torch.concat([x[:B], x[:B]], dim=-1)
        x_f = self.trans_block(x_f, N)
        x_o = self.mapping_xyz(x_f)
        return x_f.permute(0, 2, 1), x_o.permute(0, 2, 1)


class MotionFormerBlockTU(nn.Module):
    def __init__(self, dim, motion_dim, flow_feats, num_heads=8, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.PReLU, norm_layer=nn.BatchNorm1d):
        super().__init__()

        # self.shift_size = shift_size
        # if not isinstance(self.shift_size, (tuple, list)):
        #    self.shift_size = to_2tuple(shift_size)
        # self.bidirectional = bidirectional
        self.norm1 = norm_layer(dim)
        self.attn = InterFrameAttentionTU(
            dim,
            motion_dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.apply(self._init_weights)
        self.mapping_xyz = nn.Linear(flow_feats[0], 3)
        self.trans_block = Mlp(in_features=dim*3, hidden_features=mlp_hidden_dim, out_features=flow_feats[0], act_layer=act_layer, drop=drop)
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        # elif isinstance(m, nn.Conv1d):

    def forward(self, x, N, B):
        nwB = x.shape[0]
        x_norm = (self.norm1(x)).permute(0, 2, 1)  # B,C,N
        x_reverse = torch.cat([ x_norm[nwB*2 // 3:], x_norm[nwB // 3:nwB // 3*2], x_norm[:nwB // 3]])
        x_appearence = self.attn(x_norm, x_reverse, N)  # B,C,N
        x_norm = x_norm + self.drop_path(x_appearence)
        x_back = x_norm  # .view(2*B, N, -1)
        x_back = self.norm2(x_back.permute(0, 2, 1))
        x_back = self.drop_path(self.mlp(x_back.permute(0, 2, 1), N))
        x = x + x_back.permute(0, 2, 1)  # self.drop_path(self.mlp(self.norm2(x_back.permute(0,2,1)), N))
        x_f = torch.concat([x[:B], x[B:2*B], x[2*B:]], dim=1).permute(0, 2, 1)
        x_f = self.trans_block(x_f, N)
        x_o = self.mapping_xyz(x_f)
        return x_f.permute(0, 2, 1), x_o.permute(0, 2, 1)


class InterFrameAttentionTU(nn.Module):
    def __init__(self, dim, motion_dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.motion_dim = motion_dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.cor_embed = nn.Linear(3, motion_dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.motion_proj = nn.Linear(motion_dim, motion_dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x1, x2, N, mask=None):
        B, N, C = x1.shape
        q = self.q(x1).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        kv = self.kv(x2).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        if mask is not None:
            nW = mask.shape[0]  # mask: nW, N, N
            attn = attn.view(B // nW, nW, self.num_heads, N, N) + mask.unsqueeze(
                1
            ).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = attn.softmax(dim=-1)
        else:
            attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Estimatier(nn.Module):
    def __init__(self, scale=1.0, iters=3):
        super(Estimatier, self).__init__()
        flow_nei = 32
        weightnet = 8
        self.scale = scale
        self.iters = iters
        # l0: 8192

        self.recurrent0 = RecurrentUnit(iters=iters, feat_ch=32, feat_new_ch=32, latent_ch=64 + 16*4, cross_mlp1=[32, 32], cross_mlp2=[32, 32], weightnet=weightnet, flow_channels = [64, 64], flow_mlp = [32, 32])
        # l1: 2048
        c = 32

        self.rf_block0 = FlowRefineNet(32, 32, c=c)

        self.recurrent1 = RecurrentUnitAtt(iters=iters, feat_ch=64, feat_new_ch=64, latent_ch=64 + 32*4, cross_mlp1=[64, 64],
                                        cross_mlp2=[64, 64], weightnet=weightnet)
        self.rf_block1 = FlowRefineNet(64, 64, c=2 * c)
        # l2: 512
        self.recurrent2 = RecurrentUnitAtt(iters=iters, feat_ch=128, feat_new_ch=128, latent_ch=64  + 64*4, cross_mlp1=[128, 128],
                                        cross_mlp2=[128, 128], weightnet=weightnet)
        self.rf_block2 = FlowRefineNet(128, 128, c=4 * c)

        # l3: 256
        self.cross3 = CrossLayer(flow_nei, 256 + 64, [256, 256], [256, 256])
        self.flow3 = SceneFlowEstimatorResidual(256 + 128*4, 256, channels=[128, 64], mlp=[], weightnet=weightnet)
        self.rf_block3 = FlowRefineNet(256, 256, c=8 * c)
        self.cross_block3 = MotionFormerBlockT(
            dim=256, motion_dim=32,
            mlp_ratio=4., qkv_bias=True, qk_scale=None,
            drop=0.0, attn_drop=0.0, drop_path=0.04, norm_layer=nn.BatchNorm1d)
        # deconv
        self.deconv4_3 = Conv1d(256, 64)
        self.deconv3_2 = Conv1d(256, 128)
        self.deconv2_1 = Conv1d(128, 64)
        self.deconv1_0 = Conv1d(64, 32)

        # warping
        self.warping = PointWarping()

        # upsample
        self.upsample = UpsampleFlow()

    def forward(self, pc1s, pc2s, feat1s, feat2s, af, mf, t, gt=None, train=False):
        # xyz1, xyz2: B, N, 3
        # color1, color2: B, N, 3
        B = pc1s[0].shape[0]
        # torch.cat([feat1_l2, t * mf[2][:B], (1 - t) * mf[2][B:], af[2][:B], af[2][B:]])
        amf = []
        # l4
        feat1_l4_3 = self.upsample(pc1s[3], pc1s[4], feat1s[4])
        feat1_l4_3 = self.deconv4_3(feat1_l4_3)
        feat2_l4_3 = self.upsample(pc2s[3], pc2s[4], feat2s[4])
        feat2_l4_3 = self.deconv4_3(feat2_l4_3)

        # l3
        c_feat1_l3 = torch.cat([feat1s[3], feat1_l4_3], dim=1)
        c_feat2_l3 = torch.cat([feat2s[3], feat2_l4_3], dim=1)
        feat1_new_l3, feat2_new_l3, cross3 = self.cross3(pc1s[3], pc2s[3], c_feat1_l3, c_feat2_l3, feat1s[3], feat2s[3])
        # feat3f, flows3f = self.flow3(pc1s[3], torch.cat( [feat1s[3] , t * mf[3][:B], (1 - t) * mf[3][B:], af[3][:B], af[3][B:]] , dim=1), cross3)
        x = torch.cat([feat1_new_l3, feat2_new_l3], 0)
        feat3, flows3 = self.cross_block3(x, pc1s[3].shape[1], pc1s[3].shape[0])
        flows3 = self.rf_block3(feat1s[3], feat2s[3], cross3, flows3)

        feat1_l3_2 = self.upsample(pc1s[2], pc1s[3], feat1_new_l3)
        feat1_l3_2 = self.deconv3_2(feat1_l3_2)
        feat2_l3_2 = self.upsample(pc2s[2], pc2s[3], feat2_new_l3)
        feat2_l3_2 = self.deconv3_2(feat2_l3_2)
        # l2
        up_flow2 = self.upsample(pc1s[2], pc1s[3], self.scale * flows3)
        up_feat2 = self.upsample(pc1s[2], pc1s[3], feat3)

        flows2, feat1_new_l2, feat2_new_l2, feat2, cost2 = self.recurrent2(pc1s[2], pc2s[2], feat1_l3_2, feat2_l3_2, feat1s[2],
                                                            feat2s[2], up_flow2, torch.cat( [up_feat2, t * mf[2][:B], (1 - t) * mf[2][B:], af[2][:B], af[2][B:]] , dim=1))

        flows2 = flows2[::-1][0]
        flows2 = self.rf_block2(feat1s[2], feat2s[2], cost2, flows2)
        feat1_l2_1 = self.upsample(pc1s[1], pc1s[2], feat1_new_l2)
        feat1_l2_1 = self.deconv2_1(feat1_l2_1)
        feat2_l2_1 = self.upsample(pc2s[1], pc2s[2], feat2_new_l2)
        feat2_l2_1 = self.deconv2_1(feat2_l2_1)

        # l1
        up_flow1 = self.upsample(pc1s[1], pc1s[2], self.scale * flows2)
        up_feat1 = self.upsample(pc1s[1], pc1s[2], feat2)

        flows1, feat1_new_l1, feat2_new_l1, feat1, cost1 = self.recurrent1(pc1s[1], pc2s[1], feat1_l2_1, feat2_l2_1, feat1s[1],
                                                                    feat2s[1], up_flow1, torch.cat( [up_feat1, t * mf[1][:B], (1 - t) * mf[1][B:], af[1][:B], af[1][B:]] , dim=1))
        flows1 = flows1[::-1][0]
        flows1 = self.rf_block1(feat1s[1], feat2s[1], cost1, flows1)

        feat1_l1_0 = self.upsample(pc1s[0], pc1s[1], feat1_new_l1)
        feat1_l1_0 = self.deconv1_0(feat1_l1_0)
        feat2_l1_0 = self.upsample(pc2s[0], pc2s[1], feat2_new_l1)
        feat2_l1_0 = self.deconv1_0(feat2_l1_0)

        # l0
        up_flow0 = self.upsample(pc1s[0], pc1s[1], self.scale * flows1)
        up_feat0 = self.upsample(pc1s[0], pc1s[1], feat1)


        flows0, feat1_new_l0, feat2_new_l0, feat0, cost0 = self.recurrent0(pc1s[0], pc2s[0], feat1_l1_0, feat2_l1_0, feat1s[0],
                                                            feat2s[0], up_flow0, torch.cat( [up_feat0, af[0][:B], af[0][B:]] , dim=1))
        flows0 = flows0[::-1][0]
        flows0 = self.rf_block0(feat1s[0], feat2s[0], cost0, flows0)
        flows = [flows0, flows1, flows2, flows3]

        return flows

class SceneFlowPWC(nn.Module):
    def __init__(self):
        super(SceneFlowPWC, self).__init__()

        flow_nei = 32
        feat_nei = 16
        self.scale = scale

        self.feature_bone = MotionFormer()
        self.encoder = PointConvEncoder()
        self.inference = Estimatier()

        self.refine = FlowRefineNet_Unet()
        self.fusion = PointsFusion(4, [64, 64, 128])

    def forward(self, xyz1, xyz2, t, gt=None, train=False):

        af, mf = self.feature_bone(xyz1, xyz2)
        '''
        af[0]: torch.Size([12, 32, 8192]) af[1]: torch.Size([12, 32, 2048]) af[2] torch.Size([12, 64, 512]) af[3] torch.Size([12, 128, 256])
        mf[0]: <class 'list'> 0
        mf[1]: torch.Size([12, 32, 2048]) mf[2] torch.Size([12, 64, 512]) mf[3] torch.Size([12, 512, 256])
        '''
        #af: b,32,N(8192) | b,32,n//2(2048) | b,64,n//8(512)
        #mf: [] | b,32,N//2(2048) | b,64,n//8(512)
        # encoder
        pc1s, feat1s, idx1s = self.encoder(xyz1, xyz1)
        pc2s, feat2s, idx2s = self.encoder(xyz2, xyz2)


        flows = self.inference(pc1s, pc2s, feat1s, feat2s, af, mf, t, gt, train)
        # refine
        _, _, N = flows[0].shape
        k = 32
        warped_pc1t, warped_pc2t, warped_pc1t_l1, warped_pc1t_l2, warped_pc1t_l3, refine_out = self.refine(self.encoder,
                                                                                                           pc1s, pc2s,
                                                                                                           feat1s,
                                                                                                           flows, k=32,t=t)
        if t > 0.5:
            warped_pc = warped_pc2t
        else:
            warped_pc = warped_pc1t
        # fusion
        fused_points = self.fusion(warped_pc, refine_out, k, t)
        if train:
            gt1 = downsampling(gt, int(N / 4))
            gt2 = downsampling(gt, int(N / 16))
            gt3 = downsampling(gt, int(N / 32))
            gt_list = [gt1, gt2, gt3]

            warped_list = [warped_pc1t, warped_pc1t_l1, warped_pc1t_l2, warped_pc1t_l3]
            return fused_points, warped_list, warped_pc2t, gt_list
        else:
            return fused_points


def downsampling(pc, num):
    # return [B,3,N], [B,3,N/4], [B,3,N/16], [B,3,N/32]
    _, _, N = pc.shape
    idx = furthest_point_sample(pc.permute(0, 2, 1).contiguous(), num)
    gt = (index_points_gather(pc.permute(0, 2, 1).contiguous(), idx)).permute(0, 2, 1)
    return gt


LEAKY_RATE = 0.1
use_bn = False  # True


class Conv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, use_leaky=True, bn=use_bn):
        super(Conv1d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        relu = nn.ReLU(inplace=True) if not use_leaky else nn.LeakyReLU(LEAKY_RATE, inplace=True)

        self.composed_module = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=True),
            nn.BatchNorm1d(out_channels) if bn else nn.Identity(),
            relu
        )

    def forward(self, x):
        x = self.composed_module(x)
        return x


def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.

    src^T * dst = xn * xm + yn * ym + zn * zm;
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst

    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    # device = src.device
    # print('src:', src.shape)
    # print('dst:', dst.shape)
    dst = dst.to(src.device)
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


def knn_point(nsample, xyz, new_xyz):
    """
    Input:
        nsample: max sample number in local region
        xyz: all points, [B, N, C]
        new_xyz: query points, [B, S, C]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    sqrdists = square_distance(new_xyz, xyz)
    _, group_idx = torch.topk(sqrdists, nsample, dim=-1, largest=False, sorted=False)
    return group_idx


def gather_point(points, inds):
    '''

    :param points: shape=(B, N, C)
    :param inds: shape=(B, M) or shape=(B, M, K)
    :return: sampling points: shape=(B, M, C) or shape=(B, M, K, C)
    '''
    device = points.device
    B, N, C = points.shape
    inds_shape = list(inds.shape)
    inds_shape[1:] = [1] * len(inds_shape[1:])
    repeat_shape = list(inds.shape)
    repeat_shape[0] = 1
    batchlists = torch.arange(0, B, dtype=torch.long).to(device).reshape(inds_shape).repeat(repeat_shape)
    inds = inds.type(torch.long)
    return points[batchlists, inds, :]


def index_points_gather(points, fps_idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """

    points_flipped = points.permute(0, 2, 1).contiguous()
    new_points = gather_operation(points_flipped, fps_idx)
    return new_points.permute(0, 2, 1).contiguous()


def index_points_group(points, knn_idx):
    """
    Input:
        points: input points data, [B, N, C]
        knn_idx: sample index data, [B, N, K]
    Return:
        new_points:, indexed points data, [B, N, K, C]
    """
    points_flipped = points.permute(0, 2, 1).contiguous()
    new_points = grouping_operation(points_flipped, knn_idx.int()).permute(0, 2, 3, 1)

    return new_points


def group(nsample, xyz, points):
    """
    Input:
        nsample: scalar
        xyz: input points position data, [B, N, C]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, 1, C]
        new_points: sampled points data, [B, 1, N, C+D]
    """
    B, N, C = xyz.shape
    S = N
    new_xyz = xyz
    idx = knn_point(nsample, xyz, new_xyz)
    grouped_xyz = index_points_group(xyz, idx)  # [B, npoint, nsample, C]
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, C)
    if points is not None:
        grouped_points = index_points_group(points, idx)
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1)  # [B, npoint, nsample, C+D]
    else:
        new_points = grouped_xyz_norm

    return new_points, grouped_xyz_norm


def group_query(nsample, s_xyz, xyz, s_points):
    """
    Input:
        nsample: scalar
        s_xyz: input points position data, [B, N, C]
        s_points: input points data, [B, N, D]
        xyz: input points position data, [B, S, C]
    Return:
        new_xyz: sampled points position data, [B, 1, C]
        new_points: sampled points data, [B, 1, N, C+D]
    """
    B, N, C = s_xyz.shape
    S = xyz.shape[1]
    new_xyz = xyz
    idx = knn_point(nsample, s_xyz, new_xyz)
    grouped_xyz = index_points_group(s_xyz, idx)  # [B, npoint, nsample, C]
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, C)
    if s_points is not None:
        grouped_points = index_points_group(s_points, idx)
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1)  # [B, npoint, nsample, C+D]
    else:
        new_points = grouped_xyz_norm

    return new_points, grouped_xyz_norm


class WeightNet(nn.Module):

    def __init__(self, in_channel, out_channel, hidden_unit=[8, 8], bn=use_bn):
        super(WeightNet, self).__init__()

        self.bn = bn
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        if hidden_unit is None or len(hidden_unit) == 0:
            self.mlp_convs.append(nn.Conv2d(in_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
        else:
            self.mlp_convs.append(nn.Conv2d(in_channel, hidden_unit[0], 1))
            self.mlp_bns.append(nn.BatchNorm2d(hidden_unit[0]))
            for i in range(1, len(hidden_unit)):
                self.mlp_convs.append(nn.Conv2d(hidden_unit[i - 1], hidden_unit[i], 1))
                self.mlp_bns.append(nn.BatchNorm2d(hidden_unit[i]))
            self.mlp_convs.append(nn.Conv2d(hidden_unit[-1], out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))

    def forward(self, localized_xyz):
        # xyz : BxCxKxN

        weights = localized_xyz
        for i, conv in enumerate(self.mlp_convs):
            if self.bn:
                bn = self.mlp_bns[i]
                weights = F.relu(bn(conv(weights)))
            else:
                weights = F.relu(conv(weights))

        return weights


class PointConv(nn.Module):
    def __init__(self, nsample, in_channel, out_channel, weightnet=16, bn=use_bn, use_leaky=True):
        super(PointConv, self).__init__()
        self.bn = bn
        self.nsample = nsample
        self.weightnet = WeightNet(3, weightnet)
        self.linear = nn.Linear(weightnet * in_channel, out_channel)
        if bn:
            self.bn_linear = nn.BatchNorm1d(out_channel)

        self.relu = nn.ReLU(inplace=True) if not use_leaky else nn.LeakyReLU(LEAKY_RATE, inplace=True)

    def forward(self, xyz, points):
        """
        PointConv without strides size, i.e., the input and output have the same number of points.
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """
        B = xyz.shape[0]
        N = xyz.shape[2]
        xyz = xyz.permute(0, 2, 1)
        points = points.permute(0, 2, 1)

        new_points, grouped_xyz_norm = group(self.nsample, xyz, points)

        grouped_xyz = grouped_xyz_norm.permute(0, 3, 2, 1)
        weights = self.weightnet(grouped_xyz)
        new_points = torch.matmul(input=new_points.permute(0, 1, 3, 2), other=weights.permute(0, 3, 2, 1)).view(B, N,
                                                                                                                -1)
        # print('new_points:', new_points.shape, new_points.device)
        new_points = self.linear(new_points)
        # print('---new_points---:', new_points.shape, new_points.device)
        if self.bn:
            new_points = self.bn_linear(new_points.permute(0, 2, 1))
        else:
            new_points = new_points.permute(0, 2, 1)

        new_points = self.relu(new_points)

        return new_points


class PointConvD(nn.Module):
    def __init__(self, npoint, nsample, in_channel, out_channel, weightnet=16, bn=use_bn, use_leaky=True):
        super(PointConvD, self).__init__()
        self.npoint = npoint
        self.bn = bn
        self.nsample = nsample
        self.weightnet = WeightNet(3, weightnet)
        self.linear = nn.Linear(weightnet * in_channel, out_channel)
        if bn:
            self.bn_linear = nn.BatchNorm1d(out_channel)

        self.relu = nn.ReLU(inplace=True) if not use_leaky else nn.LeakyReLU(LEAKY_RATE, inplace=True)

    def forward(self, xyz, points):
        """
        PointConv with downsampling.
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """
        # import ipdb; ipdb.set_trace()
        B = xyz.shape[0]
        N = xyz.shape[2]
        xyz = xyz.permute(0, 2, 1).contiguous()
        points = points.permute(0, 2, 1)

        fps_idx = furthest_point_sample(xyz, self.npoint)
        new_xyz = index_points_gather(xyz, fps_idx)

        new_points, grouped_xyz_norm = group_query(self.nsample, xyz, new_xyz, points)

        grouped_xyz = grouped_xyz_norm.permute(0, 3, 2, 1)
        weights = self.weightnet(grouped_xyz)
        new_points = torch.matmul(input=new_points.permute(0, 1, 3, 2), other=weights.permute(0, 3, 2, 1)).view(B,
                                                                                                                self.npoint,
                                                                                                                -1)
        new_points = self.linear(new_points)
        if self.bn:
            new_points = self.bn_linear(new_points.permute(0, 2, 1))
        else:
            new_points = new_points.permute(0, 2, 1)

        new_points = self.relu(new_points)

        return new_xyz.permute(0, 2, 1), new_points


class PointConvDE(nn.Module):
    def __init__(self, npoint, nsample, in_channel, out_channel, weightnet = 16, bn = use_bn, use_leaky = True):
        super(PointConvDE, self).__init__()
        self.npoint = npoint
        self.bn = bn
        self.nsample = nsample
        self.weightnet = WeightNet(3, weightnet)
        self.linear = nn.Linear(weightnet * in_channel, out_channel)
        if bn:
            self.bn_linear = nn.BatchNorm1d(out_channel)

        self.relu = nn.ReLU(inplace=True) if not use_leaky else nn.LeakyReLU(LEAKY_RATE, inplace=True)

    def forward(self, xyz, points, fps_idx=None, new_xyz=None):
        """
        PointConv with downsampling.
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """
        #import ipdb; ipdb.set_trace()
        B = xyz.shape[0]
        N = xyz.shape[2]
        xyz = xyz.permute(0, 2, 1).contiguous()
        points = points.permute(0, 2, 1)

        if new_xyz is None:
            if fps_idx is None:
                fps_idx = furthest_point_sample(xyz, self.npoint)
            new_xyz = index_points_gather(xyz, fps_idx)
        else:
            new_xyz = new_xyz.permute(0, 2, 1)

        new_points, grouped_xyz_norm = group_query(self.nsample, xyz, new_xyz, points)
        grouped_xyz = grouped_xyz_norm.permute(0, 3, 2, 1)
        weights = self.weightnet(grouped_xyz)
        # B, N, S, C
        new_points = torch.matmul(input=new_points.permute(0, 1, 3, 2), other = weights.permute(0, 3, 2, 1)).view(B, self.npoint, -1)
        new_points = self.linear(new_points)
        if self.bn:
            new_points = self.bn_linear(new_points.permute(0, 2, 1))
        else:
            new_points = new_points.permute(0, 2, 1)

        new_points = self.relu(new_points)

        if fps_idx is not None:
            return new_xyz.permute(0, 2, 1), new_points, fps_idx
        else:
            return new_xyz.permute(0, 2, 1), new_points


class PointConvFlow(nn.Module):
    def __init__(self, nsample, in_channel, mlp, bn=use_bn, use_leaky=True):
        super(PointConvFlow, self).__init__()
        self.nsample = nsample
        self.bn = bn
        self.mlp_convs = nn.ModuleList()
        if bn:
            self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            if bn:
                self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel

        self.weightnet1 = WeightNet(3, last_channel)
        self.weightnet2 = WeightNet(3, last_channel)

        self.relu = nn.ReLU(inplace=True) if not use_leaky else nn.LeakyReLU(LEAKY_RATE, inplace=True)

    def forward(self, xyz1, xyz2, points1, points2):
        """
        Cost Volume layer for Flow Estimation
        Input:
            xyz1: input points position data, [B, C, N1]
            xyz2: input points position data, [B, C, N2]
            points1: input points data, [B, D, N1]
            points2: input points data, [B, D, N2]
        Return:
            new_points: upsample points feature data, [B, D', N1]
        """
        # import ipdb; ipdb.set_trace()
        B, C, N1 = xyz1.shape
        _, _, N2 = xyz2.shape
        _, D1, _ = points1.shape
        _, D2, _ = points2.shape
        xyz1 = xyz1.permute(0, 2, 1)
        xyz2 = xyz2.permute(0, 2, 1)
        points1 = points1.permute(0, 2, 1)
        points2 = points2.permute(0, 2, 1)

        # point-to-patch Volume
        knn_idx = knn_point(self.nsample, xyz2, xyz1)  # B, N1, nsample
        neighbor_xyz = index_points_group(xyz2, knn_idx)
        direction_xyz = neighbor_xyz - xyz1.view(B, N1, 1, C)

        grouped_points2 = index_points_group(points2, knn_idx)  # B, N1, nsample, D2
        grouped_points1 = points1.view(B, N1, 1, D1).repeat(1, 1, self.nsample, 1)
        new_points = torch.cat([grouped_points1, grouped_points2, direction_xyz], dim=-1)  # B, N1, nsample, D1+D2+3
        new_points = new_points.permute(0, 3, 2, 1)  # [B, D1+D2+3, nsample, N1]
        for i, conv in enumerate(self.mlp_convs):
            if self.bn:
                bn = self.mlp_bns[i]
                new_points = self.relu(bn(conv(new_points)))
            else:
                new_points = self.relu(conv(new_points))

        # weighted sum
        weights = self.weightnet1(direction_xyz.permute(0, 3, 2, 1))  # B C nsample N1

        point_to_patch_cost = torch.sum(weights * new_points, dim=2)  # B C N

        # Patch to Patch Cost
        knn_idx = knn_point(self.nsample, xyz1, xyz1)  # B, N1, nsample
        neighbor_xyz = index_points_group(xyz1, knn_idx)
        direction_xyz = neighbor_xyz - xyz1.view(B, N1, 1, C)

        # weights for group cost
        weights = self.weightnet2(direction_xyz.permute(0, 3, 2, 1))  # B C nsample N1
        grouped_point_to_patch_cost = index_points_group(point_to_patch_cost.permute(0, 2, 1),
                                                         knn_idx)  # B, N1, nsample, C
        patch_to_patch_cost = torch.sum(weights * grouped_point_to_patch_cost.permute(0, 3, 2, 1), dim=2)  # B C N

        return patch_to_patch_cost


class PointWarping(nn.Module):

    def forward(self, xyz1, xyz2, flow1=None):
        if flow1 is None:
            return xyz2

        # move xyz1 to xyz2'
        xyz1_to_2 = xyz1 + flow1

        # interpolate flow
        B, C, N1 = xyz1.shape
        _, _, N2 = xyz2.shape
        xyz1_to_2 = xyz1_to_2.permute(0, 2, 1)  # B 3 N1
        xyz2 = xyz2.permute(0, 2, 1)  # B 3 N2
        flow1 = flow1.permute(0, 2, 1)

        knn_idx = knn_point(3, xyz1_to_2, xyz2)
        grouped_xyz_norm = index_points_group(xyz1_to_2, knn_idx) - xyz2.view(B, N2, 1, C)  # B N2 3 C
        dist = torch.norm(grouped_xyz_norm, dim=3).clamp(min=1e-10)
        norm = torch.sum(1.0 / dist, dim=2, keepdim=True)
        weight = (1.0 / dist) / norm

        grouped_flow1 = index_points_group(flow1, knn_idx)
        flow2 = torch.sum(weight.view(B, N2, 3, 1) * grouped_flow1, dim=2)
        warped_xyz2 = (xyz2 - flow2).permute(0, 2, 1)  # B 3 N2

        return warped_xyz2


class UpsampleFlow(nn.Module):
    def forward(self, xyz, sparse_xyz, sparse_flow):
        # import ipdb; ipdb.set_trace()
        B, C, N = xyz.shape
        _, _, S = sparse_xyz.shape

        xyz = xyz.permute(0, 2, 1)  # B N 3
        sparse_xyz = sparse_xyz.permute(0, 2, 1)  # B S 3
        sparse_flow = sparse_flow.permute(0, 2, 1)  # B S 3
        knn_idx = knn_point(3, sparse_xyz, xyz)
        grouped_xyz_norm = index_points_group(sparse_xyz, knn_idx) - xyz.view(B, N, 1, C)
        dist = torch.norm(grouped_xyz_norm, dim=3).clamp(min=1e-10)
        norm = torch.sum(1.0 / dist, dim=2, keepdim=True)
        weight = (1.0 / dist) / norm

        grouped_flow = index_points_group(sparse_flow, knn_idx)
        dense_flow = torch.sum(weight.view(B, N, 3, 1) * grouped_flow, dim=2).permute(0, 2, 1)
        return dense_flow


class SceneFlowEstimatorPointConv(nn.Module):

    def __init__(self, feat_ch, cost_ch, flow_ch=3, channels=[128, 128], mlp=[128, 64], neighbors=9, clamp=[-200, 200],
                 use_leaky=True):
        super(SceneFlowEstimatorPointConv, self).__init__()
        self.clamp = clamp
        self.use_leaky = use_leaky
        self.pointconv_list = nn.ModuleList()
        last_channel = feat_ch + cost_ch + flow_ch

        for _, ch_out in enumerate(channels):
            pointconv = PointConv(neighbors, last_channel + 3, ch_out, bn=True, use_leaky=True)
            # pointconv = PointTransformerLayer(dim = last_channel, out_c = ch_out, pos_mlp_hidden_dim = 64, attn_mlp_hidden_mult = 4)
            self.pointconv_list.append(pointconv)
            last_channel = ch_out

        self.mlp_convs = nn.ModuleList()
        for _, ch_out in enumerate(mlp):
            self.mlp_convs.append(Conv1d(last_channel, ch_out))
            last_channel = ch_out

        self.fc = nn.Conv1d(last_channel, 3, 1)

    def forward(self, xyz, feats, cost_volume, flow=None):
        '''
        feats: B C1 N
        cost_volume: B C2 N
        flow: B 3 N
        '''
        if flow is None:
            new_points = torch.cat([feats, cost_volume], dim=1)
        else:
            new_points = torch.cat([feats, cost_volume, flow], dim=1)

        for _, pointconv in enumerate(self.pointconv_list):
            # print('new_points000:', new_points.shape)
            # print('xyz:', xyz.shape)
            new_points = pointconv(xyz, new_points)
            # new_points = pointconv(new_points.permute(0,2,1), xyz.permute(0,2,1))
            # print('new_points111:', new_points.shape)

        for conv in self.mlp_convs:
            new_points = conv(new_points)

        flow = self.fc(new_points)
        return new_points, flow.clamp(self.clamp[0], self.clamp[1])


class PointsFusion(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(PointsFusion, self).__init__()

        layers = []
        out_channels = [in_channels, *out_channels]
        for i in range(1, len(out_channels)):
            layers += [nn.Conv2d(out_channels[i - 1], out_channels[i], 1, bias=True),
                       nn.BatchNorm2d(out_channels[i], eps=0.001), nn.ReLU()]

        self.conv = nn.Sequential(*layers)
        # self.sample = Sample(N)

    def knn_group(self, points1, points2, k):
        '''
        For each point in points1, query kNN points in points2
        Input:
            points1: [B,3,N]
            points2: [B,3,N]
        Output:
            new_features: [B,4,N]
            nn: [B,3,N]
        '''
        points1 = points1.permute(0, 2, 1).contiguous()
        points2 = points2.permute(0, 2, 1).contiguous()
        _, nn_idx, nn = knn_points(points1, points2, K=k, return_nn=True)
        points_resi = nn - points1.unsqueeze(2).repeat(1, 1, k, 1)
        grouped_dist = torch.norm(points_resi, dim=-1, keepdim=True)
        new_features = torch.cat([points_resi, grouped_dist], dim=-1)

        return new_features.permute(0, 3, 1, 2).contiguous(), \
            nn.permute(0, 3, 1, 2).contiguous()

    def forward(self, points1, points2, k, t):
        # 逐批次处理点云数据
        N = points1.shape[-1]  # 点数
        B = points1.shape[0]  # batch size

        new_features_list = []
        new_grouped_points_list = []

        # 对每个批次分别处理
        for i in range(B):
            # 对每个批次的点云进行K近邻分组
            new_features1, grouped_points1 = self.knn_group(points1[i:i + 1, :, :], points1[i:i + 1, :, :], k)
            new_features2, grouped_points2 = self.knn_group(points1[i:i + 1, :, :], points2[i:i + 1, :, :], k)

            # 拼接特征和点云分组
            new_features = torch.cat((new_features1, new_features2), dim=-1)
            new_grouped_points = torch.cat((grouped_points1, grouped_points2), dim=-1)

            new_features_list.append(new_features)
            new_grouped_points_list.append(new_grouped_points)

        # 重新组装批次数据
        new_features = torch.cat(new_features_list, dim=0)
        new_grouped_points = torch.cat(new_grouped_points_list, dim=0)

        # 特征处理
        new_features = self.conv(new_features)  # 通过卷积层
        new_features = torch.max(new_features, dim=1, keepdim=False)[0]  # 最大池化
        weights = F.softmax(new_features, dim=-1)  # 软注意力权重

        # 加权融合点云
        weights = weights.unsqueeze(1).repeat(1, 3, 1, 1)
        fused_points = torch.sum(torch.mul(weights, new_grouped_points), dim=-1, keepdim=False)

        return fused_points

class PointsFusion2(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(PointsFusion2, self).__init__()

        self.conv = SelfAttention(in_channels, out_channels)

    def knn_group(self, points1, points2, k):
        '''
        For each point in points1, query kNN points in points2
        Input:
            points1: [B,3,N]
            points2: [B,3,N]
        Output:
            new_features: [B,4,N,k]
            nn: [B,3,N,k]
        '''
        points1 = points1.permute(0, 2, 1).contiguous()
        points2 = points2.permute(0, 2, 1).contiguous()
        _, nn_idx, nn = knn_points(points1, points2, K=k, return_nn=True)
        points_resi = nn - points1.unsqueeze(2).repeat(1, 1, k, 1)
        grouped_dist = torch.norm(points_resi, dim=-1, keepdim=True)
        new_features = torch.cat([points_resi, grouped_dist], dim=-1)

        return new_features.permute(0, 3, 1, 2).contiguous(), nn.permute(0, 3, 1, 2).contiguous()

    def forward(self, points1, points2, pc, k, t):
        '''
        Input:
            points1: [B,3,N]
            points2: [B,3,N]
            features1: [B,C,N] (only for inference of additional features)
            features2: [B,C,N] (only for inference of additional features)
            k: int, number of kNN cluster
        Output:
            fused_points: [B,3,N]
        '''
        N = points1.shape[-1]  # 点数
        B = points1.shape[0]  # batch size

        new_features_list = []
        new_grouped_points_list = []

        for i in range(B):
            new_points1 = points1[i:i + 1, :, :]
            new_points2 = points2[i:i + 1, :, :]
            new_points3 = pc[i:i + 1, :, :]

            N2 = int(N * t)  # 设置从warped帧中采样点的个数
            N1 = N - N2

            k2 = int(k * t)
            k1 = k - k2

            randidx1 = torch.randperm(N)[:N1]  # 把N个数打散，取前N1个数
            randidx2 = torch.randperm(N)[:N2]  # 把N个数打散，取前N2个数
            # 从warped_pc1中取N1个点，从warped_pc2中取N2个点，cat起来
            new_points = torch.cat((new_points1[:, :, randidx1], new_points2[:, :, randidx2]), dim=-1)  # [B,3,N]

            new_features1, grouped_points1 = self.knn_group(new_points, new_points1, k1)
            new_features2, grouped_points2 = self.knn_group(new_points, new_points2, k2)
            new_features3, grouped_points3 = self.knn_group(new_points, new_points3, k)

            new_features = torch.cat((new_features1, new_features2, new_features3), dim=-1)
            new_grouped_points = torch.cat((grouped_points1, grouped_points2, grouped_points3), dim=-1)

            new_features_list.append(new_features)
            new_grouped_points_list.append(new_grouped_points)

        new_features = torch.cat(new_features_list, dim=0)  # [B,4,N,k*2]
        new_grouped_points = torch.cat(new_grouped_points_list, dim=0)  # [B,3,N,k*2]

        new_features = self.conv(new_features.permute(0, 2, 3, 1))
        # print('new_features:', new_features.shape)
        new_features = torch.max(new_features.permute(0, 3, 1, 2), dim=1, keepdim=False)[0]  # [B,N,K]
        weights = F.softmax(new_features, dim=-1)

        weights = weights.unsqueeze(1).repeat(1, 3, 1, 1)
        fused_points = torch.sum(torch.mul(weights, new_grouped_points), dim=-1, keepdim=False)  # [B,3,N]

        return fused_points


class DWConv(nn.Module):
    def __init__(self, dim):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv1d(dim, dim, 1, 1, 0, bias=True, groups=dim)

    def forward(self, x, N):
        B, N, C = x.shape
        x = x.transpose(1, 2).reshape(B, C, N)
        x = self.dwconv(x)
        x = x.reshape(B, C, -1).transpose(1, 2)

        return x

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.ReLU, drop=0.05):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.relu = nn.ReLU(inplace=True)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        # elif isinstance(m, nn.Conv1d):
        #    fan_out = m.kernel_size * m.out_channels
        #    fan_out //= m.groups
        #    m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
        #    if m.bias is not None:
        #        m.bias.data.zero_()

    def forward(self, x, N):
        x = self.fc1(x)
        # print('x:', x.shape)
        x = self.dwconv(x, N)
        # print('x:', x.shape)
        x = self.act(x)
        # print('x:', x.shape)
        x = self.drop(x)
        # print('x:', x.shape)
        x = self.fc2(x)
        # print('x:', x.shape)
        x = self.drop(x)
        # print('x:', x.shape)
        return x


class InterFrameAttention(nn.Module):
    def __init__(self, dim, motion_dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.05, proj_drop=0.05):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.motion_dim = motion_dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.cor_embed = nn.Linear(3, motion_dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.motion_proj = nn.Linear(motion_dim, motion_dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x1, x2, cor, N, mask=None):
        B, N, C = x1.shape
        B, N, C_c = cor.shape
        q = self.q(x1).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        kv = self.kv(x2).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        cor_embed_ = self.cor_embed(cor)
        cor_embed = cor_embed_.reshape(B, N, self.num_heads, self.motion_dim // self.num_heads).permute(0, 2, 1, 3)
        k, v = kv[0], kv[1]
        attn = (q @ k.transpose(-2, -1)) * self.scale

        if mask is not None:
            nW = mask.shape[0]  # mask: nW, N, N
            attn = attn.view(B // nW, nW, self.num_heads, N, N) + mask.unsqueeze(
                1
            ).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = attn.softmax(dim=-1)
        else:
            attn = attn.softmax(dim=-1)

        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        c_reverse = (attn @ cor_embed).transpose(1, 2).reshape(B, N, -1)
        motion = self.motion_proj(c_reverse - cor_embed_)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, motion


class InterFrameAttentionT(nn.Module):
    def __init__(self, dim, motion_dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.05, proj_drop=0.05):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.motion_dim = motion_dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.cor_embed = nn.Linear(3, motion_dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.motion_proj = nn.Linear(motion_dim, motion_dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x1, x2, N, mask=None):
        B, N, C = x1.shape
        q = self.q(x1).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        kv = self.kv(x2).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        attn = (q @ k.transpose(-2, -1)) * self.scale

        if mask is not None:
            nW = mask.shape[0]  # mask: nW, N, N
            attn = attn.view(B // nW, nW, self.num_heads, N, N) + mask.unsqueeze(
                1
            ).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = attn.softmax(dim=-1)
        else:
            attn = attn.softmax(dim=-1)

        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MotionFormerBlock(nn.Module):
    def __init__(self, dim, motion_dim, num_heads=8, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0.05, attn_drop=0.05,
                 drop_path=0.05, act_layer=nn.PReLU, norm_layer=nn.BatchNorm1d):
        super().__init__()

        # self.shift_size = shift_size
        # if not isinstance(self.shift_size, (tuple, list)):
        #    self.shift_size = to_2tuple(shift_size)
        # self.bidirectional = bidirectional
        self.norm1 = norm_layer(dim)
        self.attn = InterFrameAttention(
            dim,
            motion_dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        # elif isinstance(m, nn.Conv1d):

    def forward(self, x, cor, N, B):
        # x = x.view(2*B, -1, N)

        # print('x:', x.shape)
        nwB = x.shape[0]
        x_norm = (self.norm1(x)).permute(0, 2, 1)  # B,C,N

        x_reverse = torch.cat([x_norm[nwB // 2:], x_norm[:nwB // 2]])
        # print('x_norm:', x_norm.shape, 'x_reverse:', x_reverse.shape, 'cor:', cor.shape)
        x_appearence, x_motion = self.attn(x_norm, x_reverse, cor, N)  # B,C,N
        x_norm = x_norm + self.drop_path(x_appearence)

        x_back = x_norm  # .view(2*B, N, -1)
        # print('x_back:', x_back.shape, 'x:', x.permute(0,2,1).shape)
        x_back = self.norm2(x_back.permute(0, 2, 1))
        # print('x_back:', x_back.shape)
        x_back = self.drop_path(self.mlp(x_back.permute(0, 2, 1), N))
        # print('x_back:', x_back.shape)
        x = x + x_back.permute(0, 2, 1)  # self.drop_path(self.mlp(self.norm2(x_back.permute(0,2,1)), N))
        return x, x_motion


class ConvBlock(nn.Module):
    def __init__(self, in_dim, out_dim, depths=2, act_layer=nn.PReLU):
        super().__init__()
        layers = []
        for i in range(depths):
            if i == 0:
                layers.append(nn.Conv1d(in_dim, out_dim, 1, 1, 0, bias=True))
            else:
                layers.append(nn.Conv1d(out_dim, out_dim, 1, 1, 0, bias=True))
            layers.extend([
                act_layer(out_dim),
            ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv(x)
        return x


class MotionFormer(nn.Module):
    def __init__(self, in_chans=3, npoints=[8192, 2048, 512, 256], embed_dims=[32, 32, 64, 128], motion_dims=[0, 16, 32, 32],
                 mlp_ratios=[4, 4], qkv_bias=True, qk_scale=None, drop_rate=0.05,
                 attn_drop_rate=0.05, drop_path_rate=0.05, norm_layer=nn.BatchNorm1d,
                 depths=[2, 2, 2, 4], feat_nei=16, **kwarg):  # window_sizes=[11, 11],
        super().__init__()
        self.depths = depths
        self.num_stages = len(embed_dims)  # 5

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0

        # self.conv_stages = self.num_stages - len(num_heads) # 3

        for i in range(self.num_stages):  # 0,1,2,3
            if i == 0:  # 0
                block = ConvBlock(in_chans, embed_dims[i], depths[i])  # inputchannel=3-->outchannel=32, 2 layers
            else:  # 1,2,3
                patch_embed = PointConvD(npoints[i], feat_nei, embed_dims[i - 1] + 3, embed_dims[i])  # downsampling x2

                block = nn.ModuleList([MotionFormerBlock(
                    dim=embed_dims[i], motion_dim=motion_dims[i],
                    mlp_ratio=4., qkv_bias=qkv_bias, qk_scale=qk_scale,
                    drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + j], norm_layer=norm_layer)
                    for j in range(depths[i])])

                norm = norm_layer(embed_dims[i])
                setattr(self, f"norm{i + 1}", norm)
                setattr(self, f"patch_embed{i + 1}", patch_embed)
            cur += depths[i]

            setattr(self, f"block{i + 1}", block)

        self.cor = {}

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def get_cor(self, shape, device):
        k = (str(shape), str(device))
        if k not in self.cor:
            tenHorizontal = torch.linspace(-1.0, 1.0, shape[1], device=device).view(
                1, 1, shape[1]).expand(shape[0], -1, -1).permute(0, 2, 1)
            tenVertical = torch.linspace(-1.0, 1.0, shape[1], device=device).view(
                1, 1, shape[1]).expand(shape[0], -1, -1).permute(0, 2, 1)
            tenZaxis = torch.linspace(-1.0, 1.0, shape[1], device=device).view(
                1, 1, shape[1]).expand(shape[0], -1, -1).permute(0, 2, 1)
            self.cor[k] = torch.cat([tenHorizontal, tenVertical, tenZaxis], -1).to(device)
        return self.cor[k]

    def forward(self, x1, x2):
        B = x1.shape[0]
        x = torch.cat([x1, x2], 0)  # 2B,3,N
        motion_features = []
        appearence_features = []
        xs = []
        for i in range(self.num_stages):
            motion_features.append([])
            patch_embed = getattr(self, f"patch_embed{i + 1}", None)
            block = getattr(self, f"block{i + 1}", None)
            norm = getattr(self, f"norm{i + 1}", None)
            if i == 0:
                fea = block(x)
                xs.append(fea)
            else:
                # print('i:', i, 'x:', x.shape)
                xyz, fea = patch_embed(x, xs[i - 1])
                x = xyz
                xs.append(fea)
                N = xyz.shape[2]
                cor = self.get_cor((xyz.shape[0], N), xyz.device)
                for blk in block:
                    x_, x_motion = blk(fea, cor, N, B)
                    # print('x_motion:', x_motion.shape)
                    motion_features[i].append(x_motion.permute(0, 2, 1).contiguous())  # B,C,N
                    x_ = x_.permute(0, 2, 1)
                # print('x:', x.shape)
                x_ = norm(x_.permute(0, 2, 1))  # B,C,N
                fea = x_.reshape(2 * B, -1, N).contiguous()
                motion_features[i] = torch.cat(motion_features[i], 1)
                # print('motion:', motion_features[i].shape)
            # print('fea:', fea.shape)
            appearence_features.append(fea)
            # print('motion_features:', type(motion_features))
        return appearence_features, motion_features

