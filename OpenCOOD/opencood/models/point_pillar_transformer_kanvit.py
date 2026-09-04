# KAN-ViT variant of opencood.models.point_pillar_transformer. Identical
# pipeline (PillarVFE -> scatter -> BEV backbone -> V2X-ViT fusion -> heads),
# except the fusion transformer's FFN is KANFeedForward instead of a plain
# MLP (see opencood.models.fuse_modules.v2xvit_kan). Original file untouched.
import torch
import torch.nn as nn

from opencood.models.sub_modules.pillar_vfe import PillarVFE
from opencood.models.sub_modules.point_pillar_scatter import PointPillarScatter
from opencood.models.sub_modules.base_bev_backbone import BaseBEVBackbone
from opencood.models.fuse_modules.fuse_utils import regroup
from opencood.models.sub_modules.downsample_conv import DownsampleConv
from opencood.models.sub_modules.naive_compress import NaiveCompressor
from opencood.models.fuse_modules.v2xvit_kan import V2XTransformerKAN


class PointPillarTransformerKanvit(nn.Module):
    """
    Full KAN-ViT cooperative-perception model: PointPillars per-agent
    feature extraction (PillarVFE -> scatter -> BEV backbone) followed
    by intermediate fusion across agents (V2XTransformerKAN) and two
    1x1-conv detection heads (classification + box regression). This is
    the model.core_method the training pipeline builds when
    `model.core_method: point_pillar_transformer_kanvit` in the hypes
    yaml -- see point_pillar_intermediate_fusion_kanvit_full.yaml (Day 5
    real-training config) or the _smoketest variant (1-sequence sanity
    check). Structurally identical to point_pillar_transformer.py (the
    AttFuse/V2X-ViT-classic model) except fusion_net is
    V2XTransformerKAN instead of V2XTransformer -- see that file and
    v2xvit_kan.py for what actually changes.
    """

    def __init__(self, args):
        super(PointPillarTransformerKanvit, self).__init__()

        self.max_cav = args['max_cav']
        self.pillar_vfe = PillarVFE(args['pillar_vfe'],
                                    num_point_features=4,
                                    voxel_size=args['voxel_size'],
                                    point_cloud_range=args['lidar_range'])
        self.scatter = PointPillarScatter(args['point_pillar_scatter'])
        self.backbone = BaseBEVBackbone(args['base_bev_backbone'], 64)
        self.shrink_flag = False
        if 'shrink_header' in args:
            self.shrink_flag = True
            self.shrink_conv = DownsampleConv(args['shrink_header'])
        self.compression = False

        if args['compression'] > 0:
            self.compression = True
            self.naive_compressor = NaiveCompressor(256, args['compression'])

        self.fusion_net = V2XTransformerKAN(args['transformer'])

        self.cls_head = nn.Conv2d(128 * 2, args['anchor_number'],
                                  kernel_size=1)
        self.reg_head = nn.Conv2d(128 * 2, 7 * args['anchor_number'],
                                  kernel_size=1)

        if args['backbone_fix']:
            self.backbone_fix()

    def backbone_fix(self):
        for p in self.pillar_vfe.parameters():
            p.requires_grad = False
        for p in self.scatter.parameters():
            p.requires_grad = False
        for p in self.backbone.parameters():
            p.requires_grad = False
        if self.compression:
            for p in self.naive_compressor.parameters():
                p.requires_grad = False
        if self.shrink_flag:
            for p in self.shrink_conv.parameters():
                p.requires_grad = False
        for p in self.cls_head.parameters():
            p.requires_grad = False
        for p in self.reg_head.parameters():
            p.requires_grad = False

    def forward(self, data_dict):
        # 1) per-agent feature extraction: raw LiDAR points were already
        # voxelized upstream (SpVoxelPreprocessor, at data-loading time,
        # not here) -- PillarVFE encodes each voxel's points into one
        # pillar feature, scatter places pillars back onto a 2D BEV grid
        # per agent, and the BEV backbone runs ordinary 2D convolutions
        # over that grid to get spatial_features_2d.
        voxel_features = data_dict['processed_lidar']['voxel_features']
        voxel_coords = data_dict['processed_lidar']['voxel_coords']
        voxel_num_points = data_dict['processed_lidar']['voxel_num_points']
        record_len = data_dict['record_len']
        spatial_correction_matrix = data_dict['spatial_correction_matrix']

        prior_encoding = \
            data_dict['prior_encoding'].unsqueeze(-1).unsqueeze(-1)

        batch_dict = {'voxel_features': voxel_features,
                      'voxel_coords': voxel_coords,
                      'voxel_num_points': voxel_num_points,
                      'record_len': record_len}
        batch_dict = self.pillar_vfe(batch_dict)
        batch_dict = self.scatter(batch_dict)
        batch_dict = self.backbone(batch_dict)

        spatial_features_2d = batch_dict['spatial_features_2d']
        if self.shrink_flag:
            spatial_features_2d = self.shrink_conv(spatial_features_2d)
        if self.compression:
            spatial_features_2d = self.naive_compressor(spatial_features_2d)

        # 2) regroup: record_len says how many agents contributed to
        # each sample in the batch (variable, up to max_cav) -- regroup
        # turns the flat (total_agents_in_batch, C, H, W) tensor into a
        # padded (batch, max_cav, C, H, W) tensor + a mask, so every
        # sample has the same shape regardless of its real agent count.
        # prior_encoding carries per-agent metadata (e.g. timestamp delta,
        # used by RTE inside the fusion transformer) broadcast over H, W.
        regroup_feature, mask = regroup(spatial_features_2d,
                                        record_len,
                                        self.max_cav)
        prior_encoding = prior_encoding.repeat(1, 1, 1,
                                               regroup_feature.shape[3],
                                               regroup_feature.shape[4])
        regroup_feature = torch.cat([regroup_feature, prior_encoding], dim=2)

        # 3) intermediate fusion across agents (V2X-ViT with KAN FFN --
        # see v2xvit_kan.py), then two 1x1 conv heads: psm (per-anchor
        # objectness/class score) and rm (per-anchor box regression).
        regroup_feature = regroup_feature.permute(0, 1, 3, 4, 2)
        fused_feature = self.fusion_net(regroup_feature, mask, spatial_correction_matrix)
        fused_feature = fused_feature.permute(0, 3, 1, 2)

        psm = self.cls_head(fused_feature)
        rm = self.reg_head(fused_feature)

        output_dict = {'psm': psm,
                       'rm': rm}

        return output_dict
