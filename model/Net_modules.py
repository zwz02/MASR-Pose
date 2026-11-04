import torch
import torch.nn as nn
import torch.nn.functional as F

from rotation_utils import Ortho6d2Mat


def index_points(points, idx):
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


class Reconstructor(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.pts_per_kpt = cfg.pts_per_kpt
        self.ndim = cfg.ndim

        self.pos_enc = nn.Sequential(
            nn.Conv1d(3, 64, 1),
            nn.ReLU(),
            nn.Conv1d(64, 128, 1),
            nn.ReLU(),
            nn.Conv1d(128, self.ndim, 1),
        )

        self.mlp = nn.Sequential(
            nn.Conv1d(self.ndim, self.ndim, 1),
            nn.ReLU(),
            nn.Conv1d(self.ndim, self.ndim, 1),
        )

        self.shape_decoder = nn.Sequential(
            nn.Conv1d(2 * self.ndim, 512, 1),
            nn.ReLU(),
            nn.Conv1d(512, 512, 1),
            nn.ReLU(),
            nn.Conv1d(512, 3 * self.pts_per_kpt, 1),
        )

    def forward(self, kpt_3d, kpt_feature):
        """
        Args:
            kpt_3d: (b, 3, kpt_num)
            kpt_feature: (b, c, kpt_num)

        Returns:
            recon_model: (b, 3, pts_per_kpt*kpt_num)
        """
        b = kpt_3d.shape[0]
        kpt_num = kpt_3d.shape[2]
        pos_enc_3d = self.pos_enc(kpt_3d)  # (b, c, kpt_num)
        kpt_feature = self.mlp(kpt_feature)  # (b, c, kpt_num)

        global_feature = torch.mean(pos_enc_3d + kpt_feature, dim=2, keepdim=True)  # (b, c, 1)
        recon_feature = torch.cat([global_feature.repeat(1, 1, kpt_num), kpt_feature],
                                  dim=1)  # (b, 2c, kpt_num)
        # (b, 3*pts_per_kpt, kpt_num)
        recon_delta = self.shape_decoder(recon_feature)
        # (b, pts_per_kpt*kpt_num, 3)
        recon_delta = recon_delta.transpose(1, 2).reshape(b, kpt_num * self.pts_per_kpt, 3).contiguous()
        # (b, pts_per_kpt*kpt_num, 3)
        kpt_3d_interleave = kpt_3d.transpose(1, 2).repeat_interleave(self.pts_per_kpt, dim=1).contiguous()

        recon_model = (recon_delta + kpt_3d_interleave).transpose(1, 2).contiguous()
        return recon_model, recon_delta


class GeometricAwareFeatureAggregator(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.block_num = cfg.block_num
        self.K = cfg.K
        self.d_model = cfg.d_model

        assert self.K.__len__() == self.block_num

        # build GAFA blocks
        self.GAFA_blocks = nn.ModuleList()
        for i in range(self.block_num):
            self.GAFA_blocks.append(GAFA_block(self.K[i], self.d_model))

    def forward(self, kpt_feature, kpt_3d, pts_feature, pts, normals):
        """
        Args:
            kpt_feature: (b, kpt_num, dim)
            kpt_3d: (b, kpt_num, 3)
            pts_feature: (b, n, dim)
            pts: (b, n, 3)

        Returns:
            kpt_feature: (b, kpt_num, dim)
        """
        for i in range(self.block_num):
            kpt_feature = self.GAFA_blocks[i](kpt_feature, kpt_3d, pts_feature, pts, normals)

        return kpt_feature


class GAFA_block(nn.Module):
    def __init__(self, k, d_model):
        super().__init__()
        self.k = k
        self.d_model = d_model

        self.fc_in = nn.Sequential(
            nn.Conv1d(d_model, d_model, 1),
            nn.BatchNorm1d(d_model),
            nn.ReLU(),
            nn.Conv1d(d_model, d_model, 1),
        )

        self.freq_bands = 2.0 ** torch.linspace(0, 2, 3)

        '''self.fc_delta = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )'''
        self.fc_geo_attn = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )

        self.fc_geo_value = nn.Sequential(
            nn.Linear(10, 64),
            nn.ReLU(),
            nn.Linear(64, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )

        self.fc_key_fuse = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )

        # Value fusion.
        self.fc_value_fuse = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )

        self.fc_delta_1 = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )

        self.fc_delta_l = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )

        self.fc_delta_abs = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )

        self.fuse_mlp = nn.Sequential(
            nn.Conv1d(3 * d_model, d_model, 1),
            nn.BatchNorm1d(d_model),
            nn.ReLU(),
            nn.Conv1d(d_model, d_model, 1),
            nn.BatchNorm1d(d_model),
            nn.ReLU(),
        )

        self.out_mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )

        self.relu = nn.ReLU()
        self.tau = 5.0

    def forward(self, kpt_feature, kpt_3d, pts_feature, pts, normals):
        """_summary_

        Args:
            kpt_feature (_type_): (b, kpt_num, 2c)
            kpt_3d (_type_): (b, kpt_num, 3)
            pts_feature (_type_): (b, n, 2c)
            pts (_type_): (b, n, 3)
            normals: (B, 3, N)
        Returns:
            kpt_feature: (b, kpt_num, 2c)
        """
        # KNN neighbors for each keypoint.
        dis_mat = torch.norm(kpt_3d.unsqueeze(2) - pts.unsqueeze(1), dim=3)
        knn_idx = dis_mat.argsort()[:, :, :self.k]
        knn_xyz = index_points(pts, knn_idx)
        knn_feature = index_points(pts_feature, knn_idx)
        knn_normals = index_points(normals.transpose(1, 2), knn_idx)

        # Explicit local geometry: relative offset, radial alignment,
        # and tangential relation.
        SPATIAL_SCALE = 10.0
        delta_xyz = (knn_xyz - kpt_3d.unsqueeze(2)) * SPATIAL_SCALE
        radial_alignment = torch.sum(delta_xyz * knn_normals, dim=-1, keepdim=True)
        tangent_constraint = torch.cross(delta_xyz, knn_normals, dim=-1)
        geo_input = torch.cat(
            [delta_xyz, knn_normals, radial_alignment, tangent_constraint], dim=-1
        )

        # Key stream uses the offset; value stream uses the full descriptor.
        pos_enc_attn = self.fc_geo_attn(delta_xyz)
        pos_enc_value = self.fc_geo_value(geo_input)

        pre = kpt_feature
        kpt_query = self.fc_in(kpt_feature.transpose(1, 2))
        knn_key = self.fc_key_fuse(torch.cat([knn_feature, pos_enc_attn], dim=-1))
        knn_value = self.fc_value_fuse(torch.cat([knn_feature, pos_enc_value], dim=-1))

        sim = F.cosine_similarity(
            kpt_query.transpose(1, 2).unsqueeze(2).repeat(1, 1, self.k, 1),
            knn_key,
            dim=-1
        )
        sim = F.softmax(sim / self.tau, dim=2)
        # (b, kpt_num, 1, k) @ (b, kpt_num, k, 2c) -> (b, kpt_num, 2c)
        kpt_feature = torch.matmul(sim.unsqueeze(2), knn_value).squeeze(2)
        kpt_feature = F.relu(kpt_feature + pre)

        pre = kpt_feature
        pos_enc_abs = self.fc_delta_abs(kpt_3d)
        # Pairwise keypoint offsets.
        dis_mat_l = kpt_3d.unsqueeze(2) - kpt_3d.unsqueeze(1)
        pos_enc_abs_l = self.fc_delta_l(dis_mat_l)
        pos_enc_abs_l = torch.mean(pos_enc_abs_l, dim=2)

        pos_enc_l = pos_enc_abs + pos_enc_abs_l
        kpt_num = kpt_3d.shape[1]
        kpt_global = torch.mean(kpt_feature, dim=1, keepdim=True)
        # [F_kpt, global F_kpt, pairwise position encoding]
        kpt_feature = self.fuse_mlp(torch.cat(
            [kpt_feature.transpose(1, 2), kpt_global.transpose(1, 2).repeat(1, 1, kpt_num), pos_enc_l.transpose(1, 2)],
            dim=1))
        kpt_feature = F.relu(kpt_feature.transpose(1, 2) + pre)
        pre = kpt_feature
        kpt_feature = self.out_mlp(kpt_feature)

        return self.relu(pre + kpt_feature)


class NOCS_Predictor(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.bins_num = cfg.bins_num
        self.cat_num = cfg.cat_num
        bin_lenth = 1 / self.bins_num
        half_bin_lenth = bin_lenth / 2
        self.bins_center = torch.linspace(start=-0.5, end=0.5 - bin_lenth, steps=self.bins_num).view(self.bins_num,
                                                                                                     1).cuda()
        self.bins_center = self.bins_center + half_bin_lenth  # (bins_num, 1)

        self.nocs_mlp = nn.Sequential(
            nn.Conv1d(256, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Conv1d(512, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Conv1d(512, 256, 1),
            nn.BatchNorm1d(256),
        )

        self.self_attn_layer = SelfAttnLayer(cfg.AttnLayer)

        self.atten_mlp = nn.Sequential(
            nn.Conv1d(256, 512, 1),
            nn.ReLU(),
            nn.Conv1d(512, 512, 1),
            nn.ReLU(),
            nn.Conv1d(512, self.cat_num * 3 * self.bins_num, 1),
        )

    def forward(self, kpt_feature, index):
        """_summary_

        Args:
            kpt_feature: (b, kpt_num, dim)
        Return:
            kpt_nocs:           (b, kpt_num, 3)
        """
        b, kpt_num, c = kpt_feature.shape
        kpt_feature = self.nocs_mlp(kpt_feature.transpose(1, 2)).transpose(1, 2)
        kpt_feature = self.self_attn_layer(kpt_feature)
        # (b, self.cat_num*3*bins_num, kpt_num)
        attn = self.atten_mlp(kpt_feature.transpose(1, 2))
        attn = attn.view(b * self.cat_num, 3 * self.bins_num, kpt_num).contiguous()
        attn = torch.index_select(attn, 0, index)
        attn = attn.view(b, 3, self.bins_num, kpt_num).contiguous()
        attn = F.softmax(attn, dim=2)
        attn = attn.permute(0, 3, 1, 2)
        kpt_nocs = torch.matmul(attn, self.bins_center).squeeze(-1)

        return kpt_nocs


class AttnBlock(nn.Module):
    def __init__(self, d_model=256, num_heads=4, dim_ffn=256, dropout=0.0, dropout_attn=None):
        super(AttnBlock, self).__init__()
        if dropout_attn is None:
            dropout_attn = dropout
        self.multihead_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.self_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout, inplace=False)
        self.dropout2 = nn.Dropout(dropout, inplace=False)
        self.dropout3 = nn.Dropout(dropout, inplace=False)
        self.ffn = nn.Sequential(nn.Linear(d_model, dim_ffn),
                                 nn.ReLU(), nn.Dropout(dropout, inplace=False),
                                 nn.Linear(dim_ffn, d_model))

    def with_pos_embed(self, tensor, pos=None):
        return tensor if pos is None else tensor + pos

    def forward(self, kpt_query, input_feature):
        # cross-attn
        kpt_query2 = self.norm1(kpt_query)
        kpt_query2, attn = self.multihead_attn(query=kpt_query2,
                                               key=input_feature,
                                               value=input_feature
                                               )
        kpt_query = kpt_query + self.dropout1(kpt_query2)

        # self-attn
        kpt_query2 = self.norm2(kpt_query)
        kpt_query2, _ = self.self_attn(kpt_query2,
                                       kpt_query2,
                                       value=kpt_query2
                                       )
        kpt_query = kpt_query + self.dropout2(kpt_query2)

        # ffn
        kpt_query2 = self.norm3(kpt_query)
        kpt_query2 = self.ffn(kpt_query2)
        kpt_query = kpt_query + self.dropout3(kpt_query2)

        return kpt_query, attn


class AttnLayer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.block_num = cfg.block_num
        self.d_model = cfg.d_model
        self.num_head = cfg.num_head
        self.dim_ffn = cfg.dim_ffn

        # build attention blocks
        self.attn_blocks = nn.ModuleList()
        for i in range(self.block_num):
            self.attn_blocks.append(
                AttnBlock(d_model=self.d_model, num_heads=self.num_head, dim_ffn=self.dim_ffn, dropout=0.0,
                          dropout_attn=None))

    def forward(self, batch_kpt_query, input_feature):
        """
            update kpt_query to instance-specific queries
        Args:
            batch_kpt_query: b, kpt_num, dim
            input_feature: b, dim, n

        Returns:
            batch_kpt_query: b, kpt_num, dim
            attn:  b, kpt_num, n
        """
        input_feature = input_feature.transpose(1, 2)  # (b, n, 2c)

        # (b, kpt_num, c)  (b, kpt_num, n)
        for i in range(self.block_num):
            batch_kpt_query, attn = self.attn_blocks[i](batch_kpt_query, input_feature)

        return batch_kpt_query, attn


class SelfAttnBlock(nn.Module):
    def __init__(self, d_model=256, num_heads=4, dim_ffn=256, dropout=0.0, dropout_attn=None):
        super().__init__()
        if dropout_attn is None:
            dropout_attn = dropout

        self.self_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout, inplace=False)
        self.dropout2 = nn.Dropout(dropout, inplace=False)
        self.ffn = nn.Sequential(nn.Linear(d_model, dim_ffn),
                                 nn.ReLU(), nn.Dropout(dropout, inplace=False),
                                 nn.Linear(dim_ffn, d_model))

    def with_pos_embed(self, tensor, pos=None):
        return tensor if pos is None else tensor + pos

    def forward(self, kpt_query):
        # self-attn
        kpt_query2 = self.norm1(kpt_query)
        kpt_query2, _ = self.self_attn(kpt_query2,
                                       kpt_query2,
                                       value=kpt_query2
                                       )
        kpt_query = kpt_query + self.dropout1(kpt_query2)

        # ffn
        kpt_query2 = self.norm2(kpt_query)
        kpt_query2 = self.ffn(kpt_query2)
        kpt_query = kpt_query + self.dropout2(kpt_query2)

        return kpt_query


class SelfAttnLayer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.block_num = cfg.block_num
        self.d_model = cfg.d_model
        self.num_head = cfg.num_head
        self.dim_ffn = cfg.dim_ffn

        # build attention blocks
        self.attn_blocks = nn.ModuleList()
        for i in range(self.block_num):
            self.attn_blocks.append(
                SelfAttnBlock(d_model=self.d_model, num_heads=self.num_head, dim_ffn=self.dim_ffn, dropout=0.0,
                              dropout_attn=None))

    def forward(self, batch_kpt_query):
        """
        Args:
            batch_kpt_query: b, kpt_num, dim

        Returns:
            batch_kpt_query: b, kpt_num, dim
        """
        # (b, kpt_num, c)  (b, kpt_num, n)
        for i in range(self.block_num):
            batch_kpt_query = self.attn_blocks[i](batch_kpt_query)

        return batch_kpt_query


class InstanceAdaptiveKeypointDetector(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.kpt_num = cfg.kpt_num
        self.query_dim = cfg.query_dim
        # initialize shared kpt_query for all categories
        self.kpt_query = nn.Parameter(torch.empty(self.kpt_num, self.query_dim))  # (kpt_num, query_dim)
        nn.init.xavier_normal_(self.kpt_query)

        # build attention layer
        self.attn_layer = AttnLayer(cfg.AttnLayer)

    def forward(self, input_feature):
        """_summary_

        Args:
            rgb_local (_type_): (b, c, n)
            pts_local (_type_): (b, c, n)
            cls:                (b, )
        """
        b, c, n = input_feature.shape

        batch_kpt_query = self.kpt_query.unsqueeze(0).repeat(b, 1, 1)
        # (b, kpt_num, 2c)  (b, kpt_num, n)
        batch_kpt_query, attn = self.attn_layer(batch_kpt_query, input_feature)

        # cos similarity <a, b> / |a|*|b|
        norm1 = torch.norm(batch_kpt_query, p=2, dim=2, keepdim=True)
        norm2 = torch.norm(input_feature, p=2, dim=1, keepdim=True)
        heatmap = torch.bmm(batch_kpt_query, input_feature) / (norm1 * norm2 + 1e-7)
        heatmap = F.softmax(heatmap / 0.1, dim=2)

        return batch_kpt_query, heatmap


class PoseSizeEstimator(nn.Module):
    def __init__(self):
        super(PoseSizeEstimator, self).__init__()

        self.pts_mlp1 = nn.Sequential(
            nn.Conv1d(3, 32, 1),
            nn.ReLU(),
            nn.Conv1d(32, 64, 1),
            nn.ReLU(),
        )
        self.pts_mlp2 = nn.Sequential(
            nn.Conv1d(3, 32, 1),
            nn.ReLU(),
            nn.Conv1d(32, 64, 1),
            nn.ReLU(),
        )
        self.pose_mlp1 = nn.Sequential(
            nn.Conv1d(64 + 64 + 256, 512, 1),
            nn.ReLU(),
            nn.Conv1d(512, 256, 1),
            nn.ReLU(),
        )
        self.pose_mlp2 = nn.Sequential(
            nn.Conv1d(512, 512, 1),
            nn.ReLU(),
            nn.Conv1d(512, 512, 1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.rotation_estimator = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 6),
        )
        self.translation_estimator = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 3),
        )
        self.size_estimator = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 3),
        )

    def forward(self, pts1, pts2, pts1_local):
        pts1 = self.pts_mlp1(pts1.transpose(1, 2))
        pts2 = self.pts_mlp2(pts2.transpose(1, 2))
        pose_feat = torch.cat([pts1, pts1_local.transpose(1, 2), pts2], dim=1)  # b, c, n

        pose_feat = self.pose_mlp1(pose_feat)  # b, c, n
        pose_global = torch.mean(pose_feat, 2, keepdim=True)  # b, c, 1
        pose_feat = torch.cat([pose_feat, pose_global.expand_as(pose_feat)], 1)  # b, 2c, n
        pose_feat = self.pose_mlp2(pose_feat).squeeze(2)  # b, c

        r = self.rotation_estimator(pose_feat)
        r = Ortho6d2Mat(r[:, :3].contiguous(), r[:, 3:].contiguous()).view(-1, 3, 3)
        t = self.translation_estimator(pose_feat)
        s = self.size_estimator(pose_feat)
        return r, t, s


class StructuralRelationModule(nn.Module):
    def __init__(self, cfg):
        """
        Structural Relation Transformer (SRT)
        Adds a gated per-head geometric bias to self-attention logits.
        """
        super().__init__()
        self.d_model = cfg.d_model
        self.num_head = cfg.num_head
        self.dropout = 0.1

        assert self.d_model % self.num_head == 0
        self.head_dim = self.d_model // self.num_head

        self.q_proj = nn.Linear(self.d_model, self.d_model)
        self.k_proj = nn.Linear(self.d_model, self.d_model)
        self.v_proj = nn.Linear(self.d_model, self.d_model)

        # Pairwise offset -> one bias per attention head.
        self.geo_encoder = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, self.num_head)
        )

        # Feature -> one gate per query and head.
        self.gate_predictor = nn.Sequential(
            nn.Linear(self.d_model, 64),
            nn.ReLU(),
            nn.Linear(64, self.num_head),
        )

        self.out_proj = nn.Linear(self.d_model, self.d_model)

        self.norm1 = nn.LayerNorm(self.d_model)
        self.norm2 = nn.LayerNorm(self.d_model)

        self.ffn = nn.Sequential(
            nn.Linear(self.d_model, self.d_model * 2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.d_model * 2, self.d_model)
        )

        self.dropout_layer = nn.Dropout(self.dropout)
        self._reset_parameters()

    def _reset_parameters(self):
        # Start from plain self-attention: zero gate contribution and zero output.
        nn.init.constant_(self.gate_predictor[-1].weight, 0.0)
        nn.init.constant_(self.gate_predictor[-1].bias, -4.0)
        nn.init.constant_(self.out_proj.weight, 0.0)
        nn.init.constant_(self.out_proj.bias, 0.0)

    def forward(self, kpt_feature, kpt_3d):
        """
        Args:
            kpt_feature: (b, c, n)
            kpt_3d: (b, n, 3)
        Returns:
            refined_feature: (b, c, n)
        """
        x = kpt_feature.permute(0, 2, 1).contiguous()
        b, n, c = x.shape
        residual = x

        # Pairwise offsets; detached so the bias does not back-propagate to coordinates.
        relative_pos = kpt_3d.detach().unsqueeze(2) - kpt_3d.detach().unsqueeze(1)
        geo_bias = self.geo_encoder(relative_pos).permute(0, 3, 1, 2)

        # Gate per query and head.
        gate = torch.sigmoid(self.gate_predictor(x)).permute(0, 2, 1).unsqueeze(-1)

        query = self.q_proj(x).view(b, n, self.num_head, self.head_dim).transpose(1, 2)
        key = self.k_proj(x).view(b, n, self.num_head, self.head_dim).transpose(1, 2)
        value = self.v_proj(x).view(b, n, self.num_head, self.head_dim).transpose(1, 2)

        scale = self.head_dim ** -0.5
        semantic_logits = torch.matmul(query, key.transpose(-2, -1)) * scale

        # Feature logits + gated geometric bias.
        attn_logits = semantic_logits + gate * geo_bias

        attn_weights = F.softmax(attn_logits, dim=-1)
        attn_weights = self.dropout_layer(attn_weights)

        out = torch.matmul(attn_weights, value)
        out = out.transpose(1, 2).contiguous().view(b, n, c)
        out = self.out_proj(out)

        # Two residual updates with LayerNorm.
        x = self.norm1(out + residual)
        x = self.norm2(x + self.ffn(x))

        return x.permute(0, 2, 1).contiguous()


class SGLR_Module(nn.Module):
    """
    Semantic-Guided Local Rectification (SGLR).
    Splits the geometric channels, modulates one subset with RGB guidance,
    and leaves the remaining channels unmodulated.
    """

    def __init__(self, rgb_dim=128, pts_dim=128, hidden_dim=64):
        super().__init__()

        self.pts_dim = pts_dim
        # Modulate one half; keep the other half unchanged.
        self.modulate_dim = pts_dim // 2
        self.keep_dim = pts_dim - self.modulate_dim

        # RGB feature -> one coefficient per modulated channel.
        self.guidance_mlp = nn.Sequential(
            nn.Conv1d(rgb_dim, hidden_dim, 1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=0.05),
            nn.Conv1d(hidden_dim, self.modulate_dim, 1),
            nn.Sigmoid()
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.guidance_mlp.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.zeros_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, rgb_feat, pts_feat):
        """
        Args:
            rgb_feat: (B, C, N)
            pts_feat: (B, C, N)
        """
        pts_mod, pts_keep = torch.split(pts_feat, [self.modulate_dim, self.keep_dim], dim=1)
        guidance = self.guidance_mlp(rgb_feat)

        # 1 + guidance keeps the modulated amplitude in [1, 2] x the original.
        pts_mod_rectified = pts_mod * (1.0 + guidance)
        pts_final = torch.cat([pts_mod_rectified, pts_keep], dim=1)
        out = torch.cat([pts_final, rgb_feat], dim=1)

        return out