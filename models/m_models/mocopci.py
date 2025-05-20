import torch.nn as nn
import torch
import math
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import torch.nn.functional as F
from functools import partial
from pytorch3d.ops import knn_points
from models.pointnet2.pointnet2_utils import gather_operation, grouping_operation, furthest_point_sample
from models.pointT_layer2 import FlowRefineNet, TransformerBlock, square_distance  # , index_points
from models.pointconv_util import CrossLayerLightFeatCosine as CrossLayer, FlowEmbeddingLayer, \
    BidirectionalLayerFeatCosine
from models.pointconv_util import index_points_gather as index_points, index_points_group, Conv1d, square_distance, \
    knn_point_cosine, knn_point
import time  # from models.amformer import MotionFormer


scale = 1.0


class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            act_layer=nn.GELU,
            norm_layer=None,
            bias=True,
            drop=0.,
            use_conv=False,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)
        linear_layer = partial(nn.Conv2d, kernel_size=1) if use_conv else nn.Linear

        self.fc1 = linear_layer(in_features, hidden_features, bias=bias[0])
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.norm = norm_layer(hidden_features) if norm_layer is not None else nn.Identity()
        self.fc2 = linear_layer(hidden_features, out_features, bias=bias[1])
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.norm(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False,
                 attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, c):
        B, N, C = x.shape
        kv = self.kv(c).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)

        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Extractor(nn.Module):
    def __init__(self, dim, num_heads=8, n_points=4, n_levels=1, deform_ratio=1.0,
                 with_cffn=True, cffn_ratio=0.25, drop=0., drop_path=0.,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6), with_cp=False):
        super().__init__()
        self.query_norm = norm_layer(dim)
        self.feat_norm = norm_layer(dim)
        self.attn = CrossAttention(dim, num_heads, qkv_bias=False, attn_drop=drop, proj_drop=drop)
        self.with_cffn = with_cffn
        self.with_cp = with_cp
        if with_cffn:
            self.ffn = Mlp(in_features=dim, hidden_features=int(dim * cffn_ratio), act_layer=nn.GELU, drop=drop)
            self.ffn_norm = norm_layer(dim)
            self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        # self.pj = nn.Linear(2*dim, dim, bias=False)

    def forward(self, x1, x2):

        def _inner_forward(query, feat):

            attn = self.attn(self.query_norm(query), self.feat_norm(feat))  # TODO B N C
            query = query + attn

            query = self.drop_path(self.ffn(self.ffn_norm(query)))
            return query

        res = _inner_forward(x1, x2)
        return res


class Injector(nn.Module):
    def __init__(self, dim, num_heads=6, n_points=4, n_levels=1, deform_ratio=1.0,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6), init_values=0., with_cp=False, drop=0.):
        super().__init__()
        self.with_cp = with_cp
        self.query_norm = norm_layer(dim)
        self.feat_norm = norm_layer(dim)
        self.attn = CrossAttention(dim, num_heads, qkv_bias=False, attn_drop=drop, proj_drop=drop)
        self.gamma = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)

    def forward(self, query, feat):

        def _inner_forward(query, feat):

            attn = self.attn(self.query_norm(query), self.feat_norm(feat))
            return self.gamma * attn

        res = _inner_forward(query, feat)

        return res

class EI_Crossformer(nn.Module):
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.injector = Injector(dim=dim, num_heads=num_heads)
        self.extractor = Extractor(dim=dim, num_heads=num_heads)
        self.pj = nn.Linear(2 * dim, dim, bias=False)

    def forward(self, x1, x2):
        res1 = self.injector(x1, x2)
        res2 = self.extractor(x2, x1)
        features = self.pj(torch.concat([res1, res2], dim=-1))
        return features


class Multiframe_Attention(nn.Module):
    def __init__(self, iters, feat_ch, feat_new_ch, latent_ch, cross_mlp1, cross_mlp2,
                 weightnet=8, flow_channels=[64, 64], flow_mlp=[64, 64]):
        super(Multiframe_Attention, self).__init__()
        flow_nei = 32
        self.iters = 3
        self.scale = 1.0

        self.bid = BidirectionalLayerFeatCosine(flow_nei, feat_new_ch + feat_ch*2, cross_mlp1)
        self.fe = FlowEmbeddingLayer(flow_nei, cross_mlp1[-1], cross_mlp2)
        flow_channels = [latent_ch, latent_ch]
        self.cross_block = Multi_Frame_Att(
            dim=feat_ch, flow_feats=flow_channels,
            mlp_ratio=4., qkv_bias=True, qk_scale=None,
            drop=0.05, attn_drop=0.05, drop_path=0.04, norm_layer=nn.BatchNorm1d)
        self.downsample = Conv1d(latent_ch, feat_ch)
        self.warping = PointWarping()

    def time_embedding(self, t, embedding_dim):
        time_encoding = torch.zeros(t.shape[0], embedding_dim)

        for i, timestamp in enumerate(t):
            for j in range(0, embedding_dim, 2):
                time_encoding[i, j] = math.sin(timestamp * math.pow(10000, -j / embedding_dim))
                if j + 1 < embedding_dim:
                    time_encoding[i, j + 1] = math.cos(timestamp * math.pow(10000, -(j + 1) / embedding_dim))
        return time_encoding

    def forward(self, pc1, pc2, feat1_new, feat2_new, feat1_0, feat1_1, feat2_0, feat2_1, up_frames, up_feat, t):
        # pc1s[2], pc2s[2], feat1_l3_2, feat2_l3_2, feat1s[2], feat2s[2],up_frame2_lst, up_feat2_lst, t


        c_feat1 = torch.cat([feat1_0, feat1_1, feat1_new], dim=1)
        c_feat2 = torch.cat([feat2_0, feat2_1, feat2_new], dim=1)

        flows = []
        frame_feat_lst = []
        for i, up_frame in enumerate(up_frames):
            # pc2_warp = up_frames
            pc2_warp = self.warping(pc1, pc2, up_frame)
            # pc2_warp = pc1 + up_frame
            feat1_new, feat2_new = self.bid(pc1, pc2_warp, c_feat1, c_feat2, feat1_0, feat2_0)
            fe = self.fe(pc1, pc2_warp, feat1_new, feat2_new, feat1_0, feat2_0).unsqueeze(1)
            frame_feat_lst.append(fe)

        t = torch.tensor(t, dtype=torch.float32).cuda()
        x = torch.cat([feat1_new.unsqueeze(1), *frame_feat_lst[:3], feat2_new.unsqueeze(1)], dim=1)
        time_enc = self.time_embedding(t, embedding_dim=feat1_new.shape[1])  # 假设嵌入维度为128
        time_enc = time_enc.unsqueeze(0)
        time_enc = time_enc.unsqueeze(2).expand(-1, -1, feat1_new.shape[-1], -1)
        time_enc = time_enc.reshape(-1, feat1_new.shape[-1], feat1_new.shape[1]).unsqueeze(0)
        time_enc_expanded = time_enc.repeat(feat1_new.shape[0], 1, 1, 1).permute(0, 1, 3, 2).cuda()
        x = x + time_enc_expanded

        feat_frames, frames, loss_consistence = self.cross_block(x, pc1, pc2, frames=len(t))  #
        B, F, N, C = feat_frames.shape
        feat_frames = self.downsample(feat_frames.reshape(-1, N, C))
        feat_frames = feat_frames.reshape(B, F, feat_frames.shape[1], feat_frames.shape[2])
        return frames, feat1_new, feat2_new, feat_frames, frame_feat_lst, loss_consistence



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
        self.flow = SceneFlowGRUResidual(latent_ch, cross_mlp2[-1] + feat_ch, channels=flow_channels, mlp=flow_mlp)
        self.downsample = Conv1d(latent_ch, 64)
        self.warping = PointWarping()

    def forward(self, pc1, pc2, feat1_new, feat2_new, feat1, feat2, up_frame, up_feat):
        c_feat1 = torch.cat([feat1, feat1_new], dim=1)
        c_feat2 = torch.cat([feat2, feat2_new], dim=1)
        pc2_warp = self.warping(pc1, pc2, up_frame)
        # feat1_new, feat2_new = self.bid(pc1, pc2_warp, c_feat1, c_feat2, feat1, feat2)
        # fe = self.fe(pc1, pc2_warp, feat1_new, feat2_new, feat1, feat2)
        # new_feat1 = torch.cat([feat1, fe], dim=1)
        # feat_frame, frame = self.flow(pc1, up_feat, new_feat1, up_frame)
        fe = None
        frame = None
        feat_frame = None
        return frame, feat1_new, feat2_new, feat_frame, fe


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
        self.gru = GRUMappingNoGCN(neighbors, in_channel=cost_ch, latent_channel=feat_ch, mlp=channels)
        self.fc = nn.Conv1d(channels[-1], 3, 1)

    def forward(self, xyz, feats, cost_volume, flow=None):
        feats_new = self.gru(xyz, xyz, feats, cost_volume)
        new_points = feats_new - feats
        flow_local = self.fc(new_points).clamp(self.clamp[0], self.clamp[1])
        if flow is None:
            flow = flow_local
        else:
            flow = flow_local + flow
        return feats_new, flow


class PointConvEncoder(nn.Module):
    def __init__(self, weightnet=8):
        super(PointConvEncoder, self).__init__()
        feat_nei = 32

        self.level0_lift = Conv1d(3, 32)
        self.level0 = PointConv(feat_nei, 32 + 3, 32, weightnet=weightnet)  # out
        self.level0_1 = Conv1d(32, 64)

        self.level1 = PointConvD(2048, feat_nei, 64 + 3, 64, weightnet=weightnet)
        self.level1_0 = Conv1d(64, 64)  # out
        self.level1_1 = Conv1d(64, 128)

        self.level2 = PointConvD(512, feat_nei, 128 + 3, 128, weightnet=weightnet)
        self.level2_0 = Conv1d(128, 128)  # out
        self.level2_1 = Conv1d(128, 256)

        self.level3 = PointConvD(256, feat_nei, 256 + 3, 256, weightnet=weightnet)
        self.level3_0 = Conv1d(256, 256)  # out
        self.level3_1 = Conv1d(256, 512)

        self.level4 = PointConvD(64, feat_nei, 512 + 3, 256, weightnet=weightnet)  # out

    def forward(self, xyz, color):
        pc_0 = xyz
        feat_l0 = self.level0_lift(color)
        feat_l0 = self.level0(pc_0, feat_l0)
        feat_l0_1 = self.level0_1(feat_l0)

        # l1
        # pc_l1, feat_l1, fps_l1 = self.level1(xyz, feat_l0_1)
        pc_l1, feat_l1 = self.level1(pc_0, feat_l0_1)
        feat_l1 = self.level1_0(feat_l1)
        feat_l1_2 = self.level1_1(feat_l1)

        # l2
        # pc_l2, feat_l2, fps_l2 = self.level2(pc_l1, feat_l1_2)
        pc_l2, feat_l2 = self.level2(pc_l1, feat_l1_2)
        feat_l2 = self.level2_0(feat_l2)
        feat_l2_3 = self.level2_1(feat_l2)

        # l3
        # pc_l3, feat_l3, fps_l3 = self.level3(pc_l2, feat_l2_3)
        pc_l3, feat_l3 = self.level3(pc_l2, feat_l2_3)
        feat_l3 = self.level3_0(feat_l3)
        feat_l3_4 = self.level3_1(feat_l3)

        # l4
        # pc_l4, feat_l4, fps_l4 = self.level4(pc_l3, feat_l3_4)
        pc_l4, feat_l4 = self.level4(pc_l3, feat_l3_4)


        return [xyz, pc_l1, pc_l2, pc_l3, pc_l4], \
               [feat_l0, feat_l1, feat_l2, feat_l3, feat_l4]






class Cross_Frame_Att(nn.Module):
    def __init__(self, dim, num_heads=8, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.PReLU, norm_layer=nn.BatchNorm1d):
        super().__init__()

        self.norm1 = norm_layer(dim)
        self.attn_feats = CrossFrameAttentionInterpretation(
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.trans_block_2 = EasyMlp(in_features=dim, hidden_features=mlp_hidden_dim, out_features=dim,
                                     act_layer=act_layer, drop=drop)
        self.apply(self._init_weights)
        self.mapping_xyz = nn.Linear(dim, 3)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, xs, pc1, pc2):
        B, N, C = pc1.shape
        nwB = xs.shape[0]
        B_appearance = []
        B_frames = []
        B_loss = 0
        for i in range(B):
            x = xs[i, :, :, :]
            x_norm = (self.norm1(x)).permute(0, 2, 1)  # B,C,N
            x_reverse = torch.flip(x_norm, dims=[0])
            x_appearance = self.attn_feats(x_norm, x_reverse, N)  # B,C,N
            x_appearance = self.trans_block_2(x_appearance)
            frames = self.mapping_xyz(x_appearance)
            B_appearance.append(x_appearance)
            B_frames.append(frames)
        frames = torch.stack(B_frames)
        x_appearance = torch.stack(B_appearance)

        # loss = chamfer_loss(pc1.permute(0, 2, 1)+frames[:, 0, :, :], pc1.permute(0, 2, 1))
        loss = torch.tensor(0.0).cuda()
        # print(loss)
        frames = frames[:, 1:, :, :]
        x_appearance = x_appearance[:, 1:, :, :]
        return x_appearance, frames, loss


class Multi_Frame_Att(nn.Module):
    def __init__(self, dim, flow_feats, num_heads=8, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0.,
                 attn_drop=0.,
                 drop_path=0., act_layer=nn.PReLU, norm_layer=nn.BatchNorm1d):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn_feats = InterFrameAttentionInterpretation(
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp_T(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.apply(self._init_weights)
        self.mapping_xyz = nn.Linear(flow_feats[0], 3)
        self.trans_block = Mlp_T(in_features=dim, hidden_features=mlp_hidden_dim, out_features=flow_feats[0],
                               act_layer=act_layer, drop=drop)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        # elif isinstance(m, nn.Conv1d):

    def forward(self, xs, pc1, pc2, frames=5):  # x, pc1, pc2, frames=len(t)
        B, F, N, C = xs.shape
        B_xf = []
        B_frames = []
        for i, x in enumerate(xs):
            x_norm = (self.norm1(x)).permute(0, 2, 1)  # B,C,N
            x_reverse = torch.flip(x_norm, dims=[0])
            x_appearence = self.attn_feats(x_norm, x_reverse, N)  # B,C,N  attention
            x_norm = x_norm + self.drop_path(x_appearence)
            x_back = x_norm
            x_back = self.norm2(x_back.permute(0, 2, 1))
            x_back = self.drop_path(self.mlp(x_back.permute(0, 2, 1), N))
            x = x + x_back.permute(0, 2, 1)
            x_f = self.trans_block(x.permute(0, 2, 1), N)
            frames = self.mapping_xyz(x_f)
            B_xf.append(x_f)
            B_frames.append(frames)
        x_f = torch.stack(B_xf)
        frames = torch.stack(B_frames)
        # loss = chamfer_loss(pc1.permute(0, 2, 1)+frames[:, 0, :, :], pc1.permute(0, 2, 1)) + chamfer_loss(pc1.permute(0, 2, 1)+frames[:, -1, :, :], pc2.permute(0, 2, 1)) # flow based
        loss = torch.tensor(0.0).cuda()
        # print(loss)
        frames = frames[:, 1:-1, :, :]
        x_f = x_f[:, 1:-1, :, :]
        return x_f.permute(0, 1, 3, 2), frames.permute(0, 1, 3, 2), loss / 2





class CrossFrameAttentionInterpretation(nn.Module):
    def __init__(self, dim, num_heads=4, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        # head_dim = dim // num_heads
        head_dim = dim
        self.scale = qk_scale or head_dim ** -0.5
        self.q = nn.Linear(dim, dim * 4, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2 * 4, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x1, x2, N, mask=None):
        B, N, C = x1.shape
        q = self.q(x1).reshape(B, N, self.num_heads, C).permute(0, 2, 1, 3)
        kv = self.kv(x2).reshape(B, -1, 2, self.num_heads, C).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        if mask is not None:
            nW = mask.shape[0]  # mask: nW, N, N
            attn = attn.view(B // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = attn.softmax(dim=-1)
        else:
            attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2)
        x_shape = x.shape
        x = (x[0] + x[1]).permute(1, 0, 2)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class InterFrameAttentionInterpretation(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
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
            attn = attn.view(B // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = attn.softmax(dim=-1)
        else:
            attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x








def index_points_local(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S, [K]]
    Return:
        new_points:, indexed points data, [B, S, [K], C]
    """
    raw_size = idx.size()
    idx = idx.reshape(raw_size[0], -1)
    res = torch.gather(points, 1, idx[..., None].expand(-1, -1, points.size(-1)))
    return res.reshape(*raw_size, -1)




use_bn = False


class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, use_leaky=True, bn=use_bn,
                 bias=True, groups=1):
        super(Conv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        relu = nn.ReLU(inplace=True) if not use_leaky else nn.LeakyReLU(LEAKY_RATE, inplace=True)

        self.composed_module = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias,
                      groups=groups),
            nn.BatchNorm2d(out_channels) if bn else nn.Identity(),
            relu
        )

    def forward(self, x):
        x = self.composed_module(x)
        return x





class SceneFlowGRUResidual(nn.Module):

    def __init__(self, feat_ch, cost_ch, flow_ch=3, channels=[64, 64], mlp=[64, 64], neighbors=9, clamp=[-200, 200],
                 use_leaky=True):
        super(SceneFlowGRUResidual, self).__init__()
        self.clamp = clamp
        self.use_leaky = use_leaky
        self.pointconv_list = nn.ModuleList()
        self.gru = GRUMappingNoGCN(neighbors, in_channel=cost_ch, latent_channel=feat_ch, mlp=channels)
        self.fc = nn.Conv1d(channels[-1], 3, 1)

    def forward(self, xyz, feats, cost_volume, flow=None):
        feats_new = self.gru(xyz, xyz, feats, cost_volume)
        new_points = feats_new - feats
        flow_local = self.fc(new_points).clamp(self.clamp[0], self.clamp[1])
        if flow is None:
            flow = flow_local
        else:
            flow = flow_local + flow
        return feats_new, flow


class MultiFrameEstimatier(nn.Module):
    def __init__(self, scale=1.0, iters=3):
        super(MultiFrameEstimatier, self).__init__()
        flow_nei = 32
        weightnet = 8
        self.scale = scale
        self.iters = iters
        # l0: 8192
        out_channels = [64, 64, 128]
        layers = []
        out_channels = [4, *out_channels]
        for i in range(1, len(out_channels)):
            layers += [nn.Conv2d(out_channels[i - 1], out_channels[i], 1, bias=True),
                       nn.BatchNorm2d(out_channels[i], eps=0.001), nn.ReLU()]
        self.conv = nn.Sequential(*layers)
        self.fusion_gru = SceneFlowGRUResidual(32, 128, 3, [64, 64], [64, 64], neighbors=32, clamp=[-200, 200],
                                               use_leaky=True)

        # self.cross_fusion =
        self.rlevel0 = Conv1d(32, 64)
        self.shape1 = TransformerBlock(64, 64)
        self.level1 = PointConvD(2048, 32, 64 + 3, 64, weightnet=weightnet)
        # self.level1 = PointConvD(2048, 32, 32 + 3, 64, weightnet=weightnet)
        self.pred = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 3))

        self.recurrent0 = RecurrentUnit(iters=iters, feat_ch=32, feat_new_ch=32, latent_ch=64, cross_mlp1=[32, 32],
                                        cross_mlp2=[32, 32], weightnet=weightnet, flow_channels=[64, 64],
                                        flow_mlp=[32, 32])
        # l1: 2048
        c = 32
        self.rf_block0 = FlowRefineNet(32, 32, c=c)
        self.multi_frame_up_1 = Multiframe_Attention(iters=iters, feat_ch=64, feat_new_ch=64, latent_ch=64 + 32 * 4,
                                                     cross_mlp1=[64, 64],
                                                     cross_mlp2=[64, 64], weightnet=weightnet)
        # l2: 512
        self.multi_frame_up_2 = Multiframe_Attention(iters=iters, feat_ch=128, feat_new_ch=128, latent_ch=64 + 64 * 4,
                                                     cross_mlp1=[128, 128],
                                                     cross_mlp2=[128, 128], weightnet=weightnet)
        # l3: 256
        self.cross3 = CrossLayer(flow_nei, 256*2 + 64, [256, 256], [256, 256])  # TODO
        self.cross_block3 = Cross_Frame_Att(
            dim=256, num_heads=4, mlp_ratio=4., qkv_bias=True, qk_scale=None,
            drop=0.05, attn_drop=0.05, drop_path=0., norm_layer=nn.BatchNorm1d)
        # deconv
        self.deconv4_3 = Conv1d(256, 64)
        self.deconv3_2 = Conv1d(256, 128)
        self.deconv2_1 = Conv1d(128, 64)
        self.deconv1_0 = Conv1d(64, 32)
        # warping
        self.warping = PointWarping()
        # upsample
        self.upsample = UpsampleFlow()

        self.ei3 = EI_Crossformer(dim=256)
        self.ei2 = EI_Crossformer(dim=128)
        self.ei1 = EI_Crossformer(dim=64)

    def knn_group(self, points1, points2, k):
        points1 = points1.permute(0, 2, 1).contiguous()
        points2 = points2.permute(0, 2, 1).contiguous()
        knn_idx = knn_point(k, points1, points1)  # B, N1, nsample
        knn_idx_p = knn_point(k, points2, points1)  # B, N1, nsample
        neighbor_xyz = torch.cat((index_points_group(points2, knn_idx), index_points_group(points2, knn_idx_p)), -2)
        points_resi = neighbor_xyz - points1.unsqueeze(2).repeat(1, 1, 2 * k, 1)
        grouped_dist = torch.norm(points_resi, dim=-1, keepdim=True)
        new_features = torch.cat([points_resi, grouped_dist], dim=-1)
        return new_features.permute(0, 3, 1, 2).contiguous(), \
               neighbor_xyz.permute(0, 3, 1, 2).contiguous()

    def fusion(self, points1, points2, k, t):
        N = points1.shape[-1]  # 点数
        B = points1.shape[0]  # batch size
        new_features, neighbor_xyz = self.knn_group(points1, points2, k)
        new_features = self.conv(new_features)  # [B,128,N,32+16]
        new_features = torch.max(new_features, dim=1, keepdim=False)[0]
        weights = F.softmax(new_features, dim=-1)
        weights = weights.unsqueeze(1).repeat(1, 3, 1, 1)
        fused_points = torch.sum(torch.mul(weights, neighbor_xyz), dim=-1, keepdim=False)
        return fused_points

    def forward(self, pc1s, pc2s, feat1_0s, feat1_1s, feat2_0s, feat2_1s, t, train=False):
        inter_num = 3
        B = pc1s[0].shape[0]
        t_f = [0.0, 0.41666666666666663, 0.5, 0.5833333333333333, 1.0]
        t_b = [1.0, 0.5833333333333333, 0.5, 0.41666666666666663, 0.0]

        # pc1s, pc2s, feat1_0s, None, feat2_0s, None, t, train

        feat_fusions = [None, ]
        f1 = self.ei1(feat1_0s[1].permute(0, 2, 1), feat2_0s[1].permute(0, 2, 1)).permute(0, 2, 1)
        feat_fusions.append(f1)

        f2 = self.ei2(feat1_0s[2].permute(0, 2, 1), feat2_0s[2].permute(0, 2, 1)).permute(0, 2, 1)
        feat_fusions.append(f2)

        f3 = self.ei3(feat1_0s[3].permute(0, 2, 1), feat2_0s[3].permute(0, 2, 1)).permute(0, 2, 1)
        feat_fusions.append(f3)



        # l4
        feat1_l4_3 = self.upsample(pc1s[3], pc1s[4], feat1_0s[4])
        feat1_l4_3 = self.deconv4_3(feat1_l4_3)
        feat2_l4_3 = self.upsample(pc2s[3], pc2s[4], feat2_0s[4])
        feat2_l4_3 = self.deconv4_3(feat2_l4_3)

        # l3
        c_feat1_l3 = torch.cat([feat1_0s[3], feat_fusions[3], feat1_l4_3], dim=1)
        c_feat2_l3 = torch.cat([feat2_0s[3], feat_fusions[3], feat2_l4_3], dim=1)
        s_time = time.time()
        feat1_new_l3, feat2_new_l3, cross3 = self.cross3(pc1s[3], pc2s[3], c_feat1_l3, c_feat2_l3, feat1_0s[3], feat2_0s[3])
        # forward:
        x = torch.cat([feat1_new_l3.unsqueeze(1), feat2_new_l3.unsqueeze(1)], 1)
        feats3s_f, frame3s_f, loss_consistence_3 = self.cross_block3(x, pc1s[3], pc2s[3])
        x = torch.cat([feat2_new_l3.unsqueeze(1), feat1_new_l3.unsqueeze(1)], 1)
        feats3s_b, frame3s_b, loss_consistence_3 = self.cross_block3(x, pc2s[3], pc1s[3])
        # print('cross_block3:', time.time()-s_time)
        feat1_l3_2 = self.upsample(pc1s[2], pc1s[3], feat1_new_l3)
        feat1_l3_2 = self.deconv3_2(feat1_l3_2)
        feat2_l3_2 = self.upsample(pc2s[2], pc2s[3], feat2_new_l3)
        feat2_l3_2 = self.deconv3_2(feat2_l3_2)

        # l2
        up_frame2_lst_f = []
        up_feat2_lst_f = []
        frames3_lst_f = []
        up_frame2_lst_b = []
        up_feat2_lst_b = []
        frames3_lst_b = []
        for i in range(inter_num):
            up_frame2_lst_f.append(
                self.upsample(pc1s[2], pc1s[3], self.scale * frame3s_f[:, i, :, :].permute(0, 2, 1)))  # upsample flow3s
            up_feat2_lst_f.append(self.upsample(pc1s[2], pc1s[3], feats3s_f[:, i, :, :].permute(0, 2, 1)))
            frames3_lst_f.append(pc1s[3].permute(0, 2, 1) + frame3s_f[:, i, :, :])
            up_frame2_lst_b.append(
                self.upsample(pc2s[2], pc2s[3], self.scale * frame3s_b[:, i, :, :].permute(0, 2, 1)))  # upsample flow3s
            up_feat2_lst_b.append(self.upsample(pc2s[2], pc2s[3], feats3s_b[:, i, :, :].permute(0, 2, 1)))
            frames3_lst_b.append(pc2s[3].permute(0, 2, 1) + frame3s_b[:, inter_num - i - 1, :, :])

        # s_time = time.time()
        # forward
        frame2s_f, feat1_new_l2_f, feat2_new_l2_f, feats2s_f, cost2, loss_consistence_2 = self.multi_frame_up_2(pc1s[2],
                                                                                                                pc2s[2],
                                                                                                                feat1_l3_2,
                                                                                                                feat2_l3_2,
                                                                                                                feat1_0s[
                                                                                                                    2],
                                                                                                                feat_fusions[
                                                                                                                    2],
                                                                                                                feat2_0s[
                                                                                                                    2],
                                                                                                                feat_fusions[
                                                                                                                    2],
                                                                                                                up_frame2_lst_f,
                                                                                                                up_feat2_lst_f,
                                                                                                                t_f)
        frame2s_b, feat2_new_l2_b, feat1_new_l2_b, feats2s_b, cost2, loss_consistence_2 = self.multi_frame_up_2(pc2s[2],
                                                                                                                pc1s[2],
                                                                                                                feat2_l3_2,
                                                                                                                feat1_l3_2,
                                                                                                                feat2_0s[
                                                                                                                    2],
                                                                                                                feat_fusions[
                                                                                                                    2],
                                                                                                                feat1_0s[
                                                                                                                    2],
                                                                                                                feat_fusions[
                                                                                                                    2],
                                                                                                                up_frame2_lst_b,
                                                                                                                up_feat2_lst_b,
                                                                                                                t_b)
        # print('multi_frame_up_2:', time.time()-s_time)
        frame2s_f = frame2s_f.permute(0, 1, 3, 2)
        feats2s_f = feats2s_f.permute(0, 1, 3, 2)
        frame2s_b = frame2s_b.permute(0, 1, 3, 2)
        feats2s_b = feats2s_b.permute(0, 1, 3, 2)

        # for i in range(1, inter_num):
        #     flow2s[:, i, :, :] = flow2s[:, i-1, :, :] + flow2s[:, i, :, :]
        feat1_l2_1_f = self.upsample(pc1s[1], pc1s[2], feat1_new_l2_f)
        feat1_l2_1_f = self.deconv2_1(feat1_l2_1_f)
        feat2_l2_1_f = self.upsample(pc2s[1], pc2s[2], feat2_new_l2_f)
        feat2_l2_1_f = self.deconv2_1(feat2_l2_1_f)
        feat1_l2_1_b = self.upsample(pc1s[1], pc1s[2], feat1_new_l2_b)
        feat1_l2_1_b = self.deconv2_1(feat1_l2_1_b)
        feat2_l2_1_b = self.upsample(pc2s[1], pc2s[2], feat2_new_l2_b)
        feat2_l2_1_b = self.deconv2_1(feat2_l2_1_b)

        # l1
        up_frame1_lst_f = []
        up_feat1_lst_f = []
        frames2_lst_f = []
        up_frame1_lst_b = []
        up_feat1_lst_b = []
        frames2_lst_b = []
        for i in range(inter_num):
            up_frame1_lst_f.append(
                self.upsample(pc1s[1], pc1s[2], self.scale * frame2s_f[:, i, :, :].permute(0, 2, 1)))  # upsample flow3s
            up_feat1_lst_f.append(self.upsample(pc1s[1], pc1s[2], feats2s_f[:, i, :, :].permute(0, 2, 1)))
            frames2_lst_f.append(pc1s[2].permute(0, 2, 1) + frame2s_f[:, i, :, :])
            up_frame1_lst_b.append(
                self.upsample(pc2s[1], pc2s[2], self.scale * frame2s_b[:, i, :, :].permute(0, 2, 1)))  # upsample flow3s
            up_feat1_lst_b.append(self.upsample(pc2s[1], pc2s[2], feats2s_b[:, i, :, :].permute(0, 2, 1)))
            frames2_lst_b.append(pc2s[2].permute(0, 2, 1) + frame2s_b[:, inter_num - i - 1, :, :])

        # s_time = time.time()
        frame1s_f, feat1_new_l1_f, feat2_new_l1_f, feats1s_f, cost1, loss_consistence_1 = self.multi_frame_up_1(pc1s[1],
                                                                                                                pc2s[1],
                                                                                                                feat1_l2_1_f,
                                                                                                                feat2_l2_1_f,
                                                                                                                feat1_0s[
                                                                                                                    1],
                                                                                                                feat_fusions[
                                                                                                                    1],
                                                                                                                feat2_0s[
                                                                                                                    1],
                                                                                                                feat_fusions[
                                                                                                                    1],
                                                                                                                up_frame1_lst_f,
                                                                                                                up_feat1_lst_f,
                                                                                                                t_f)
        frame1s_b, feat2_new_l1_b, feat1_new_l1_b, feats1s_b, cost1, loss_consistence_1 = self.multi_frame_up_1(pc2s[1],
                                                                                                                pc1s[1],
                                                                                                                feat2_l2_1_b,
                                                                                                                feat1_l2_1_b,
                                                                                                                feat2_0s[
                                                                                                                    1],
                                                                                                                feat_fusions[
                                                                                                                    1],
                                                                                                                feat1_0s[
                                                                                                                    1],
                                                                                                                feat_fusions[
                                                                                                                    1],
                                                                                                                up_frame1_lst_b,
                                                                                                                up_feat1_lst_b,
                                                                                                                t_b)
        # print('multi_frame_up_1:', time.time()-s_time)
        frame1s_f = frame1s_f.permute(0, 1, 3, 2)
        feats1s_f = feats1s_f.permute(0, 1, 3, 2)
        frame1s_b = frame1s_b.permute(0, 1, 3, 2)
        feats1s_b = feats1s_b.permute(0, 1, 3, 2)



        # l0
        up_frame0_lst_f = []
        up_feat0_lst_f = []
        frames1_lst_f = []
        up_frame0_lst_b = []
        up_feat0_lst_b = []
        frames1_lst_b = []
        frame0_lst_f = []
        frame0_lst_b = []
        frame0_lst_f_r = []
        frame0_lst_b_r = []
        out_lst = []
        for i in range(inter_num):
            up_frame0_lst_f.append(
                self.upsample(pc1s[0], pc1s[1], self.scale * frame1s_f[:, i, :, :].permute(0, 2, 1)))  # upsample flow3s
            up_feat0_lst_f.append(self.upsample(pc1s[0], pc1s[1], feats1s_f[:, i, :, :].permute(0, 2, 1)))
            frames1_lst_f.append(pc1s[1].permute(0, 2, 1) + frame1s_f[:, i, :, :])
            up_frame0_lst_b.append(self.upsample(pc2s[0], pc2s[1],
                                                 self.scale * frame1s_b[:, inter_num - i - 1, :, :].permute(0, 2,
                                                                                                            1)))  # upsample flow3s
            up_feat0_lst_b.append(
                self.upsample(pc2s[0], pc2s[1], feats1s_b[:, inter_num - i - 1, :, :].permute(0, 2, 1)))
            frames1_lst_b.append(pc2s[1].permute(0, 2, 1) + frame1s_b[:, inter_num - i - 1, :, :])

        for i in range(inter_num):
            # up_flow0 = up_frame0_lst[i] - pc1s[0]
            warped_pc1t = pc1s[0] + up_frame0_lst_f[i]
            warped_pc2t = pc2s[0] + up_frame0_lst_b[i]

            warped_pc1t_r = pc1s[0] + up_frame0_lst_b[inter_num-1-i]
            warped_pc2t_r = pc2s[0] + up_frame0_lst_f[inter_num-1-i]

            if i <= 1:
                up_flow0 = up_frame0_lst_f[i]
                up_feat0 = up_feat0_lst_f[i]
                flows0 = up_flow0
                warped_feat1t_l0 = feat1_0s[0] + (
                    F.interpolate(flows0.permute(0, 2, 1), size=feat1_0s[0].size(1), mode="area")).permute(0, 2, 1)
                warped_feat1t_l0 = self.rlevel0(warped_feat1t_l0)
                fused_down1, fused_feat1 = self.level1(warped_pc1t, warped_feat1t_l0)
                fea_shape1 = self.shape1(fused_feat1.permute(0, 2, 1), fused_down1.permute(0, 2, 1))  # [B,64,2048]
                up_feat0 = self.upsample(warped_pc1t, fused_down1, fea_shape1)
                refine_out_f = (self.pred(up_feat0.permute(0, 2, 1))).permute(0, 2, 1)
            if i > 1:
                up_flow0 = up_frame0_lst_b[i]
                up_feat0 = up_feat0_lst_b[i]
                flows0 = up_flow0
                warped_feat1t_l0 = feat2_0s[0] + (
                    F.interpolate(flows0.permute(0, 2, 1), size=feat2_0s[0].size(1), mode="area")).permute(0, 2, 1)
                warped_feat1t_l0 = self.rlevel0(warped_feat1t_l0)
                fused_down1, fused_feat1 = self.level1(warped_pc2t, warped_feat1t_l0)
                fea_shape1 = self.shape1(fused_feat1.permute(0, 2, 1), fused_down1.permute(0, 2, 1))
                up_feat0 = self.upsample(warped_pc2t, fused_down1, fea_shape1)
                refine_out_b = (self.pred(up_feat0.permute(0, 2, 1))).permute(0, 2, 1)


            frame0_lst_f.append(warped_pc1t.permute(0, 2, 1))
            frame0_lst_b.append(warped_pc2t.permute(0, 2, 1))

            frame0_lst_f_r.append(warped_pc1t_r.permute(0, 2, 1))
            frame0_lst_b_r.append(warped_pc2t_r.permute(0, 2, 1))
            if i == 0:
                final_out = self.fusion(warped_pc1t, refine_out_f, 32, t)
            if i == 1:
                final_out = self.fusion(warped_pc1t, refine_out_f, 32, t)
            if i == 2:
                final_out = self.fusion(warped_pc2t, refine_out_b, 32, t)
            # # print('fusion:', (time.time()-s_time)*3)
            out_lst.append(final_out.permute(0, 2, 1))


        flows_lst_f = [frame0_lst_f, frame0_lst_f_r, frames1_lst_f, frames2_lst_f, frames3_lst_f]
        flows_lst_b = [frame0_lst_b, frame0_lst_b_r, frames1_lst_b, frames2_lst_b, frames3_lst_b]
        # loss_consistence = 0.8*loss_consistence_1 + 0.4*loss_consistence_2 + 0.2*loss_consistence_3
        return flows_lst_f, flows_lst_b, out_lst


class MoCoPCI(nn.Module):
    def __init__(self):
        super(MoCoPCI, self).__init__()
        self.scale = scale
        self.encoder = PointConvEncoder()
        self.multi_frame_inference = MultiFrameEstimatier()

    def forward(self, xyz1, xyz2, gt, t, train=False):
        iter_num = 3

        pc1s, feat1_0s = self.encoder(xyz1, xyz1)
        pc2s, feat2_0s = self.encoder(xyz2, xyz2)


        flows_lst_f, flows_lst_b, out_lst = self.multi_frame_inference(pc1s, pc2s, feat1_0s, None, feat2_0s, None, t, train)
        _, _, N = pc1s[0].shape
        if train:
            gt_frame = []
            for i in range(iter_num):
                gt1 = downsampling(gt[i], int(N / 4))
                gt2 = downsampling(gt[i], int(N / 16))
                gt3 = downsampling(gt[i], int(N / 32))
                gt_list = [gt[i], gt1, gt2, gt3]
                gt_frame.append(gt_list)
            frames_lst_f = [[], [], []]
            frames_lst_b = [[], [], []]
            for i, (frame_lst_f, frame_lst_b) in enumerate(zip(flows_lst_f, flows_lst_b)):
                frames_lst_f[0].append(frame_lst_f[0])
                frames_lst_f[1].append(frame_lst_f[1])
                frames_lst_f[2].append(frame_lst_f[2])
                frames_lst_b[0].append(frame_lst_b[0])
                frames_lst_b[1].append(frame_lst_b[1])
                frames_lst_b[2].append(frame_lst_b[2])
            return frames_lst_f, frames_lst_b, gt_frame, out_lst
        else:
            return out_lst

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
    def __init__(self, npoint, nsample, in_channel, out_channel, weightnet=16, bn=use_bn, use_leaky=True):
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
        # import ipdb; ipdb.set_trace()
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
        new_points = torch.matmul(input=new_points.permute(0, 1, 3, 2), other=weights.permute(0, 3, 2, 1)).view(B,
                                                                                                                self.npoint,
                                                                                                                -1)
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


class UpsampleFrame(nn.Module):
    def forward(self, xyz, sparse_xyz, sparse_frame):
        # import ipdb; ipdb.set_trace()
        B, C, N = xyz.shape
        _, _, S = sparse_xyz.shape

        xyz = xyz.permute(0, 2, 1)  # B N 3
        sparse_xyz = sparse_xyz.permute(0, 2, 1)  # B S 3
        sparse_flow = sparse_frame.permute(0, 2, 1)  # B S 3
        knn_idx = knn_point(3, sparse_xyz, xyz)
        grouped_xyz_norm = index_points_group(sparse_xyz, knn_idx) - xyz.view(B, N, 1, C)
        dist = torch.norm(grouped_xyz_norm, dim=3).clamp(min=1e-10)
        norm = torch.sum(1.0 / dist, dim=2, keepdim=True)
        weight = (1.0 / dist) / norm

        grouped_flow = index_points_group(sparse_frame, knn_idx)
        dense_flow = torch.sum(weight.view(B, N, 3, 1) * sparse_frame, dim=2).permute(0, 2, 1)
        return dense_flow


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


class Mlp_T(nn.Module):
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

    def forward(self, x, N):
        x = self.fc1(x)
        x = self.dwconv(x, N)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class EasyMlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.ReLU, drop=0.05):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
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

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
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

