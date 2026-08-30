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
    def __init__(self, args):
        super(V2XTransformerKAN, self).__init__()

        encoder_args = args['encoder']
        self.encoder = V2XTEncoderKAN(encoder_args)

    def forward(self, x, mask, spatial_correction_matrix):
        output = self.encoder(x, mask, spatial_correction_matrix)
        output = output[:, 0]
        return output
