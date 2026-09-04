# KAN-ViT variant of opencood.models.fuse_modules.v2xvit_basic.
# Only the FeedForward (FFN/MLP) inside V2XTEncoder's transformer blocks is
# replaced, with KANFeedForward (Chebyshev-polynomial KAN, see
# opencood.models.sub_modules.chebykan_layer). Everything else -- spatial
# transform (STTF), relative temporal encoding (RTE), CAV/pyramid-window
# attention (V2XFusionBlock) -- is reused unmodified from v2xvit_basic.py.
# The original file is left untouched.
from opencood.models.sub_modules.base_transformer import PreNorm
from opencood.models.sub_modules.chebykan_layer import KANFeedForward
from opencood.models.fuse_modules.v2xvit_basic import STTF, RTE, V2XFusionBlock
from opencood.models.sub_modules.torch_transformation_utils import \
    get_roi_and_cav_mask
import torch
import torch.nn as nn


class V2XTEncoderKAN(nn.Module):
    """
    KAN-ViT variant of v2xvit_basic.V2XTEncoder: same encoder loop
    (per layer: V2XFusionBlock attention, then a feed-forward block with
    a residual add), same STTF spatial alignment and RTE time encoding.
    The only change is what the feed-forward block is -- KANFeedForward
    (Chebyshev-KAN, see chebykan_layer.py) instead of the original's
    plain 2-layer MLP. `kan_degree`/`kan_pool_size`, read here from the
    `feed_forward` config block (default degree=4, pool_size=1 i.e. no
    spatial pooling if omitted -- see KANFeedForward's own docstring for
    why the real configs set pool_size>1), are the only new
    hyperparameters this adds over the base encoder's config schema.
    """

    def __init__(self, args):
        super().__init__()

        cav_att_config = args['cav_att_config']
        pwindow_att_config = args['pwindow_att_config']
        feed_config = args['feed_forward']

        num_blocks = args['num_blocks']
        depth = args['depth']
        mlp_dim = feed_config['mlp_dim']
        dropout = feed_config['dropout']
        kan_degree = feed_config.get('kan_degree', 4)
        kan_pool_size = feed_config.get('kan_pool_size', 1)

        self.downsample_rate = args['sttf']['downsample_rate']
        self.discrete_ratio = args['sttf']['voxel_size'][0]
        self.use_roi_mask = args['use_roi_mask']
        self.use_RTE = cav_att_config['use_RTE']
        self.RTE_ratio = cav_att_config['RTE_ratio']
        self.sttf = STTF(args['sttf'])
        # adjust the channel numbers from 256+3 -> 256
        self.prior_feed = nn.Linear(cav_att_config['dim'] + 3,
                                    cav_att_config['dim'])
        self.layers = nn.ModuleList([])
        if self.use_RTE:
            self.rte = RTE(cav_att_config['dim'], self.RTE_ratio)
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                V2XFusionBlock(num_blocks, cav_att_config, pwindow_att_config),
                PreNorm(cav_att_config['dim'],
                        KANFeedForward(cav_att_config['dim'], mlp_dim,
                                      degree=kan_degree, dropout=dropout,
                                      pool_size=kan_pool_size))
            ]))

    def forward(self, x, mask, spatial_correction_matrix):
        prior_encoding = x[..., -3:]
        x = x[..., :-3]
        if self.use_RTE:
            dt = prior_encoding[:, :, 0, 0, 1].to(torch.int)
            x = self.rte(x, dt)
        x = self.sttf(x, mask, spatial_correction_matrix)
        com_mask = mask.unsqueeze(1).unsqueeze(2).unsqueeze(
            3) if not self.use_roi_mask else get_roi_and_cav_mask(x.shape,
                                                                   mask,
                                                                   spatial_correction_matrix,
                                                                   self.discrete_ratio,
                                                                   self.downsample_rate)
        for attn, ff in self.layers:
            x = attn(x, mask=com_mask, prior_encoding=prior_encoding)
            x = ff(x) + x
        return x


class V2XTransformerKAN(nn.Module):
    """
    Thin wrapper matching v2xvit_basic.V2XTransformer's interface: wraps
    V2XTEncoderKAN and keeps only the ego agent's output (index 0 along
    the agent/L dimension) after fusion, since the encoder returns one
    feature map per agent but downstream detection heads only need the
    ego's. This is the class point_pillar_transformer_kanvit.py builds
    as its fusion_net.
    """

    def __init__(self, args):
        super(V2XTransformerKAN, self).__init__()

        encoder_args = args['encoder']
        self.encoder = V2XTEncoderKAN(encoder_args)

    def forward(self, x, mask, spatial_correction_matrix):
        output = self.encoder(x, mask, spatial_correction_matrix)
        output = output[:, 0]
        return output
