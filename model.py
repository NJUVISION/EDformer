import torch.nn as nn
import sparseconvnet as scn
import torch
import torch.nn.functional as F
from pytorch3d.ops import knn_points, knn_gather, ball_query, sample_farthest_points
import math

TIMESTAMP_COLUMN = 0
X_COLUMN = 1
Y_COLUMN = 2
POLARITY_COLUMN = 3


class SparseConv(nn.Module):
    def __init__(self, height, width, in_channel, out_channel, filter_size):
        super(SparseConv, self).__init__()
        self.height = height
        self.width = width
        self.conv = scn.Sequential(scn.InputLayer(dimension=2, spatial_size=torch.LongTensor([height, width]), mode=3),
                                   scn.SubmanifoldConvolution(dimension=2, nIn=in_channel + 2, nOut=out_channel,
                                                              filter_size=filter_size,
                                                              bias=True),
                                   scn.OutputLayer(out_channel))

    def event2image(self, xytp):
        B, N = xytp.shape[:2]
        if not hasattr(self, 'b') or self.b.shape[0] != B * N or self.b.device != xytp.device:
            self.b = torch.zeros([B * N, 1]).long().to(xytp.device)
            for i in range(B):
                self.b[i * N:i * N + N] = i
        yx = xytp[:, :, [Y_COLUMN, X_COLUMN]].view(-1, 2)
        yx[:, 0] = torch.round(yx[:, 0] * self.height)
        yx[:, 1] = torch.round(yx[:, 1] * self.width)
        yxb = torch.cat([yx, self.b], dim=-1).long()
        return yxb

    def forward(self, xytp, features=None):
        pos = xytp[..., POLARITY_COLUMN][..., None]
        neg = 1 - xytp[..., POLARITY_COLUMN][..., None]
        sparse_input = torch.cat([pos, neg], dim=-1).view(-1, 2)
        if features is not None:
            sparse_input = torch.cat([sparse_input, features.view(-1, features.size(-1))], dim=-1)
        sparse_output = self.conv([self.event2image(xytp), sparse_input]).view(xytp.size(0), xytp.size(1), -1)
        return sparse_output


class SpatialEmbedding(nn.Module):
    def __init__(self, height, width, in_channel, out_channel, filter_size=9):
        super(SpatialEmbedding, self).__init__()
        self.embedding = SparseConv(height=height, width=width, in_channel=in_channel,
                                    out_channel=out_channel, filter_size=filter_size)

    def forward(self, xytp):
        xytp = xytp.clone().detach()
        Fsp = self.embedding(xytp, None)
        return Fsp


class TemporalEmbedding(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(TemporalEmbedding, self).__init__()
        self.pos_embed = nn.Linear(in_channel, out_channel)
        self.neg_embed = nn.Linear(in_channel, out_channel)

    def forward(self, xytp):
        xytp = xytp.clone().detach()
        pos_neg = xytp[..., POLARITY_COLUMN][..., None]
        timestamp = xytp[..., TIMESTAMP_COLUMN][..., None]
        Fte = self.pos_embed(timestamp) * pos_neg + self.neg_embed(timestamp) * (1 - pos_neg)
        return Fte


class SpatiotemporalEmbedding(nn.Module):
    def __init__(self, height, width, spatial_embeding_size, temporal_embedding_size, out_channel, norm_layer=None):
        super(SpatiotemporalEmbedding, self).__init__()
        self.spatial_embedding = SpatialEmbedding(
            height=height, width=width, in_channel=0, out_channel=spatial_embeding_size)
        self.temporal_embedding = TemporalEmbedding(in_channel=1, out_channel=temporal_embedding_size)
        self.proj = nn.Linear(spatial_embeding_size + temporal_embedding_size, out_channel, bias=False)
        self.norm = nn.LayerNorm(out_channel)

    def forward(self, xytp):
        Fsp = self.spatial_embedding(xytp)
        Fte = self.temporal_embedding(xytp)
        F = torch.cat([Fsp, Fte], -1)
        F = self.proj(F)
        if self.norm:
            F = self.norm(F)
        return F


class MLP(nn.Module):
    def __init__(self, in_channel, hidden_channel=None, out_channel=None, act_layer=nn.GELU):
        super(MLP, self).__init__()
        out_channel = out_channel or in_channel
        hidden_channel = hidden_channel or in_channel
        self.net = nn.Sequential(
            nn.Linear(in_channel, hidden_channel),
            act_layer(),
            nn.Linear(hidden_channel, out_channel)
        )

    def forward(self, x):
        return self.net(x)


class PositionEncoder(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(PositionEncoder, self).__init__()
        self.linear1 = nn.Linear(in_channel, in_channel)
        self.batch_norm = nn.BatchNorm1d(in_channel)
        self.relu = nn.ReLU(inplace=True)
        self.linear2 = nn.Linear(in_channel, out_channel)

    def forward(self, x):
        x = self.linear1(x)
        b, n, k, f = x.shape
        x = self.batch_norm(x.reshape([b * n, k, f]).transpose(1, 2)).transpose(1, 2)
        x = self.linear2(self.relu(x)).reshape([b, n, k, -1])
        return x


class LXformer(nn.Module):
    def __init__(self, in_channel, out_channel, k_l=16):
        super(LXformer, self).__init__()
        self.k_l = k_l
        self.attn_scale = math.sqrt(out_channel)
        self.local_position_encoding = PositionEncoder(in_channel=4, out_channel=32)
        self.local_transformations = nn.Linear(in_channel, out_channel * 3)
        self.local_norm = nn.LayerNorm(in_channel)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, xytp, features):
        xyt = xytp[:, :, :3].clone().detach()
        idx = knn_points(xyt, xyt, K=self.k_l).idx
        delta = self.local_position_encoding(xytp[:, :, None] - knn_gather(xytp, idx))
        local_transformations = self.local_transformations(features)
        c = local_transformations.shape[-1] // 3
        varphi, psi, alpha = local_transformations[...,
                                                   :c], local_transformations[..., c:2 * c], local_transformations[..., 2 * c:]
        psi, alpha = knn_gather(psi, idx), knn_gather(alpha, idx)
        local_attn = self.softmax(self.local_norm(
            varphi[:, :, None, :] - psi + delta) / self.attn_scale) * (alpha + delta)
        local_attn = torch.sum(local_attn, dim=2)
        return local_attn


class SCformer(nn.Module):
    def __init__(self, in_channel, out_channel, k_sc=16, filter_size=9, height=260, width=346):
        super(SCformer, self).__init__()
        self.k_sc = k_sc
        self.height = height
        self.width = width
        self.attn_scale = math.sqrt(out_channel)
        self.sparse_position_encoding = PositionEncoder(in_channel=2, out_channel=32)
        self.sparse_transformations = SparseConv(height=height, width=width, in_channel=in_channel,
                                                 out_channel=out_channel*3, filter_size=filter_size)
        self.sparse_norm = nn.LayerNorm(in_channel)
        self.softmax = nn.Softmax(dim=2)

    def _get_self_idx(self, k_idx):
        b, n, k = k_idx.shape
        if not hasattr(self, 'idx') or self.idx.shape != k_idx.shape or self.idx.device != k_idx.device:
            self.idx = torch.arange(n, device=k_idx.device)[
                None, :, None].repeat(b, 1, k)
        return self.idx

    def forward(self, xytp, features):
        xyt = xytp[..., :3].clone().detach()
        xyt[..., TIMESTAMP_COLUMN] = 0
        temp = ball_query(xyt, xyt, radius=5/self.height, K=self.k_sc)
        idx = temp.idx
        idx = torch.where(idx == -1, self._get_self_idx(idx), idx)
        xy = xytp[..., [X_COLUMN, Y_COLUMN]]
        delta = self.sparse_position_encoding(xy[:, :, None] - knn_gather(xy, idx))
        sparse_transformations = self.sparse_transformations(xytp, features)
        c = sparse_transformations.shape[-1] // 3
        varphi, psi, alpha = sparse_transformations[...,
                                                    :c], sparse_transformations[..., c:2 * c], sparse_transformations[..., 2 * c:]
        psi, alpha = knn_gather(psi, idx), knn_gather(alpha, idx)
        sparse_attn = self.softmax(self.sparse_norm(
            varphi[:, :, None, :] - psi + delta) / self.attn_scale) * (alpha + delta)
        sparse_attn = torch.sum(sparse_attn, dim=2)
        return sparse_attn


class GXformer(nn.Module):
    def __init__(self, in_channel, out_channel, k_g=16, r=8):
        super(GXformer, self).__init__()
        self.k_g = k_g
        self.r = r
        self.attn_scale = math.sqrt(out_channel)
        self.global_position_encoding = PositionEncoder(in_channel=4, out_channel=32)
        self.global_transformations = nn.Linear(in_channel, out_channel * 3)
        self.global_norm = nn.LayerNorm(in_channel)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, xytp, features):
        xyt = xytp[:, :, :3].clone().detach()
        sample_xyt, sample_idx = sample_farthest_points(xyt, K=xytp.shape[1] // self.r)
        sample_xytp = knn_gather(xytp, sample_idx[:, :, None])[:, :, 0, :]
        pair_idx = knn_points(sample_xyt, xyt, K=self.k_g).idx
        inv_pair_idx = knn_points(xyt, sample_xyt, K=self.k_g).idx
        delta = self.global_position_encoding(xytp[:, :, None] - knn_gather(sample_xytp, inv_pair_idx))
        global_transformations = self.global_transformations(features)
        c = global_transformations.shape[-1] // 3
        varphi, psi, alpha = global_transformations[...,
                                                    :c], global_transformations[..., c:2 * c], global_transformations[..., 2 * c:]
        psi, alpha = knn_gather(psi, pair_idx), knn_gather(alpha, pair_idx)
        psi, alpha = torch.max(psi, dim=2)[0], torch.max(alpha, dim=2)[0]
        psi, alpha = knn_gather(psi, inv_pair_idx), knn_gather(alpha, inv_pair_idx)
        global_attn = self.softmax(self.global_norm(
            varphi[:, :, None, :] - psi + delta) / self.attn_scale) * (alpha + delta)
        global_attn = torch.sum(global_attn, dim=2)
        return global_attn


class TransformerLayers(nn.Module):
    def __init__(self, height, width, in_channel, out_channel, k_l, filter_size, r):
        super(TransformerLayers, self).__init__()
        self.LXformer = LXformer(in_channel=in_channel, out_channel=out_channel, k_l=k_l)
        self.SCformer = SCformer(in_channel=in_channel, out_channel=out_channel, k_sc=k_l, filter_size=filter_size)
        self.GXformer = GXformer(in_channel=in_channel, out_channel=out_channel, k_g=k_l, r=r)
        self.proj = MLP(in_channel=out_channel*3, hidden_channel=out_channel, out_channel=out_channel)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, xytp, features):
        # LXformer
        attn_l = self.LXformer(xytp, features)
        # SCformer
        attn_sc = self.SCformer(xytp, features)
        attn = torch.cat([attn_l, attn_sc], dim=-1)
        # GXformer
        attn_g = self.GXformer(xytp, features)
        attn = torch.cat([attn, attn_g], dim=-1)
        attn = self.proj(attn)
        return attn


class TransformerLayerBlock(nn.Module):
    def __init__(self, height, width, in_channel, out_channel, k_l, filter_size, r):
        super(TransformerLayerBlock, self).__init__()
        self.norm1 = nn.LayerNorm(in_channel)
        self.norm2 = nn.LayerNorm(in_channel)
        self.transformerlayers = TransformerLayers(
            height=height, width=width, in_channel=in_channel, out_channel=out_channel, k_l=k_l, filter_size=filter_size, r=r)
        self.mlp = MLP(in_channel=in_channel, hidden_channel=in_channel*4, out_channel=out_channel)
        self.drop_path = nn.Identity()

    def forward(self, xytp, features):
        shortcut = features
        F_attn = self.transformerlayers(xytp, self.norm1(features))
        F_attn = F_attn + shortcut

        shortcut = F_attn
        F_attn = self.mlp(self.norm2(F_attn))
        F_attn = self.drop_path(F_attn) + shortcut
        return F_attn


class AttentionFeatureFusion(nn.Module):
    def __init__(self, in_channel_Large, in_channel_Small, out_channel, num_heads=1):
        super(AttentionFeatureFusion, self).__init__()
        self.num_heads = num_heads
        self.out_channel = out_channel
        self.proj_1 = nn.Linear(in_channel_Large, out_channel * num_heads, bias=False)
        self.proj_2 = nn.Linear(in_channel_Small, out_channel * num_heads, bias=False)
        self.proj_3 = nn.Linear(out_channel * num_heads * 2, out_channel * num_heads * 2, bias=False)
        self.Tanh = nn.Tanh()
        self.mlp = nn.Linear(out_channel * num_heads * 2, out_channel * num_heads)

    def forward(self, F_L_attn, F_S_attn):
        F_L_attn = self.proj_1(F_L_attn)
        F_S_attn = self.proj_2(F_S_attn)
        F_attn = torch.cat((F_L_attn, F_S_attn), dim=-1)
        F_attn = self.proj_3(F_attn)
        weights = F.softmax(F_attn, dim=-1)
        output = weights * F_attn
        output = self.mlp(output)
        return output


class EDformer(nn.Module):
    def __init__(self, height=260, width=346):
        super(EDformer, self).__init__()
        self.spatial_embeding_size = 32
        self.temporal_embedding_size = 32
        self.spatiotemporal_embedding = SpatiotemporalEmbedding(
            height=height, width=width, spatial_embeding_size=self.spatial_embeding_size, temporal_embedding_size=self.temporal_embedding_size, out_channel=32, norm_layer=True)
        self.transformer_layers_Large = TransformerLayerBlock(
            height=height, width=width, in_channel=32, out_channel=32, k_l=16, filter_size=9, r=8)
        self.transformer_layers_Small = TransformerLayerBlock(
            height=height, width=width, in_channel=32, out_channel=32, k_l=16, filter_size=9, r=16)
        self.attention_fusion = AttentionFeatureFusion(in_channel_Large=32, in_channel_Small=32, out_channel=32)
        self.head = nn.Linear(32, 1)

    def forward(self, xytp):
        B, N, _ = xytp.size()
        # Large scale branch
        F_L = self.spatiotemporal_embedding(xytp)
        F_L_attn = self.transformer_layers_Large(xytp, F_L)
        # Small scale branch
        xytp_S_1 = xytp[:, :N//2, :]
        xytp_S_2 = xytp[:, N//2:, :]
        F_S_1 = self.spatiotemporal_embedding(xytp_S_1)
        F_S_1 = self.transformer_layers_Small(xytp_S_1, F_S_1)
        F_S_2 = self.spatiotemporal_embedding(xytp_S_2)
        F_S_2 = self.transformer_layers_Small(xytp_S_2, F_S_2)
        F_S_attn = torch.cat((F_S_1, F_S_2), dim=1)
        # Attention Feature Fusion
        F = self.attention_fusion(F_L_attn, F_S_attn)
        return self.head(F)


if __name__ == '__main__':
    torch.manual_seed(42)
    device = torch.device("cuda:0")
    events = torch.rand((2, 4096, 4)).to(device)
    edformer = EDformer().to(device)
    res = edformer(events)
    print(res)
