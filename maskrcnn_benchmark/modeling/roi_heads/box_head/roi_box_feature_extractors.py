# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
from torch import nn
from torch.nn import functional as F

from maskrcnn_benchmark.modeling import registry
from maskrcnn_benchmark.modeling.backbone import resnet
from maskrcnn_benchmark.modeling.poolers import Pooler
from maskrcnn_benchmark.modeling.make_layers import group_norm
from maskrcnn_benchmark.modeling.make_layers import make_fc


@registry.ROI_BOX_FEATURE_EXTRACTORS.register("ResNet50Conv5ROIFeatureExtractor")
class ResNet50Conv5ROIFeatureExtractor(nn.Module):
    def __init__(self, config, in_channels):
        super(ResNet50Conv5ROIFeatureExtractor, self).__init__()

        resolution = config.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        scales = config.MODEL.ROI_BOX_HEAD.POOLER_SCALES
        sampling_ratio = config.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        pooler = Pooler(
            output_size=(resolution, resolution),
            scales=scales,
            sampling_ratio=sampling_ratio,
        )

        stage = resnet.StageSpec(index=4, block_count=3, return_features=False)
        head = resnet.ResNetHead(
            block_module=config.MODEL.RESNETS.TRANS_FUNC,
            stages=(stage,),
            num_groups=config.MODEL.RESNETS.NUM_GROUPS,
            width_per_group=config.MODEL.RESNETS.WIDTH_PER_GROUP,
            stride_in_1x1=config.MODEL.RESNETS.STRIDE_IN_1X1,
            stride_init=None,
            res2_out_channels=config.MODEL.RESNETS.RES2_OUT_CHANNELS,
            dilation=config.MODEL.RESNETS.RES5_DILATION
        )

        self.pooler = pooler
        self.head = head
        self.out_channels = head.out_channels

    def forward(self, x, proposals):
        x = self.pooler(x, proposals)
        x = self.head(x)
        return x


@registry.ROI_BOX_FEATURE_EXTRACTORS.register("FPN2MLPFeatureExtractor")
class FPN2MLPFeatureExtractor(nn.Module):
    """
    Heads for FPN for classification
    """

    def __init__(self, cfg, in_channels, half_out=False, cat_all_levels=False):
        super(FPN2MLPFeatureExtractor, self).__init__()

        resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        scales = cfg.MODEL.ROI_BOX_HEAD.POOLER_SCALES
        sampling_ratio = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        pooler = Pooler(
            output_size=(resolution, resolution),
            scales=scales,
            sampling_ratio=sampling_ratio,
            in_channels=in_channels,
            cat_all_levels=cat_all_levels,
        )
        input_size = in_channels * resolution ** 2
        representation_size = cfg.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM
        use_gn = cfg.MODEL.ROI_BOX_HEAD.USE_GN
        self.pooler = pooler
        self.fc6 = make_fc(input_size, representation_size, use_gn)

        if half_out:
            out_dim = int(representation_size / 2)
        else:
            out_dim = representation_size

        self.fc7 = make_fc(representation_size, out_dim, use_gn)
        self.resize_channels = input_size
        self.out_channels = out_dim

    def forward(self, x, proposals):
        x = self.pooler(x, proposals)
        x = x.view(x.size(0), -1)

        x = F.relu(self.fc6(x))
        x = F.relu(self.fc7(x))

        return x

    def forward_without_pool(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc6(x))
        x = F.relu(self.fc7(x))
        return x


@registry.ROI_BOX_FEATURE_EXTRACTORS.register("FPNtransFeatureExtractor")
class FPNtransFeatureExtractor(nn.Module):
    """
    Heads for FPN for classification
    """

    def __init__(self, cfg, in_channels, half_out=False, cat_all_levels=False):
        super(FPNtransFeatureExtractor, self).__init__()

        resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        scales = cfg.MODEL.ROI_BOX_HEAD.POOLER_SCALES
        sampling_ratio = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        self.use_tkn_mlp = cfg.MODEL.ROI_RELATION_HEAD.BOX_FEATURE_EXTRACTOR_PARAMS.USE_TKN_MLP
        pooler = Pooler(
            output_size=(resolution, resolution),
            scales=scales,
            sampling_ratio=sampling_ratio,
            in_channels=in_channels,
            cat_all_levels=cat_all_levels,
        )
        input_size = in_channels * resolution ** 2
        representation_size = cfg.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM
        use_gn = cfg.MODEL.ROI_BOX_HEAD.USE_GN
        self.pooler = pooler
        self.fc6 = make_fc(input_size, representation_size, use_gn)

        if half_out:
            out_dim = int(representation_size / 2)
        else:
            out_dim = representation_size

        self.fc7 = make_fc(representation_size, out_dim, use_gn)
        self.resize_channels = input_size
        self.out_channels = out_dim

        self.positional_embedding = nn.init.xavier_normal_(nn.Parameter(torch.randn(1, resolution ** 2, in_channels)))
        encoder_params = dict(cfg.MODEL.ROI_RELATION_HEAD.LG_TRANS.TRANSFORMER_PARAMS)
        encoder_params["d_model"] = in_channels
        encoder_dim = in_channels
        encoder_layer = nn.TransformerEncoderLayer(**encoder_params, batch_first=True)
        self.enc_layers = 1
        self.layer_norm = nn.LayerNorm(encoder_dim)
        self.layer_norm = None
        self.encoder = nn.TransformerEncoder(encoder_layer, self.enc_layers, norm=self.layer_norm)
        self.cls_token = nn.Parameter(torch.randn(1, 1, in_channels))
        self.linear_out = make_fc(encoder_dim, out_dim)

    def forward(self, x, proposals):
        x = self.pooler(x, proposals)

        obj_cnt, C, R, _ = x.size()
        total_patch_cnt = R * R
        # compose object feature
        roi_features = x.permute(0, 2, 3, 1).contiguous().view(obj_cnt, total_patch_cnt, C)
        roi_features = roi_features + self.positional_embedding
        cls_token = self.cls_token.expand(obj_cnt, -1, -1)
        roi_features = torch.cat([cls_token, roi_features], dim=1)
        roi_features = self.encoder(roi_features)
        roi_features = roi_features[:, 0, :]
        roi_features = F.relu(self.linear_out(roi_features))
        if self.use_tkn_mlp:
            x = x.view(x.size(0), -1)
            x = F.relu(self.fc6(x))
            x = F.relu(self.fc7(x))
            roi_features = roi_features + x
        return roi_features

    def forward_without_pool(self, x):
        obj_cnt, C, R, _ = x.size()
        total_patch_cnt = R * R
        # compose object feature
        roi_features = x.permute(0, 2, 3, 1).contiguous().view(obj_cnt, total_patch_cnt, C)
        roi_features = roi_features + self.positional_embedding
        cls_token = self.cls_token.expand(obj_cnt, -1, -1)
        roi_features = torch.cat([cls_token, roi_features], dim=1)
        roi_features = self.encoder(roi_features)
        roi_features = roi_features[:, 0, :]
        roi_features = F.relu(self.linear_out(roi_features))
        if self.use_tkn_mlp:
            x = x.view(x.size(0), -1)
            x = F.relu(self.fc6(x))
            x = F.relu(self.fc7(x))
            roi_features = roi_features + x
        return roi_features


@registry.ROI_BOX_FEATURE_EXTRACTORS.register("FPN2dRelationFeatureExtractor")
class FPN2dRelationFeatureExtractor(nn.Module):
    """
    Heads for FPN for classification
    """

    def __init__(self, cfg, in_channels, half_out=False, cat_all_levels=False):
        super(FPN2dRelationFeatureExtractor, self).__init__()

        resolution = cfg.MODEL.ROI_RELATION_HEAD.BOX_FEATURE_EXTRACTOR_PARAMS.POOLER_RESOLUTION
        scales = cfg.MODEL.ROI_RELATION_HEAD.BOX_FEATURE_EXTRACTOR_PARAMS.POOLER_SCALES
        sampling_ratio = cfg.MODEL.ROI_RELATION_HEAD.BOX_FEATURE_EXTRACTOR_PARAMS.POOLER_SAMPLING_RATIO
        self.pooler = Pooler(
            output_size=(resolution, resolution),
            scales=scales,
            sampling_ratio=sampling_ratio,
            in_channels=in_channels,
            cat_all_levels=cat_all_levels,
        )
        #
        self.out_channels = cfg.MODEL.ROI_RELATION_HEAD.CONTEXT_ROI_DIM
        # if half_out:
        #     self.out_channels = self.out_channels // 2
        # self.out_conv = nn.Conv2d(in_channels, self.out_channels, (1, 1), 1)
        # nn.init.xavier_uniform_(self.out_conv.weight)
        #
        # # nn.init.xavier_uniform_(self.out_conv_d.weight)
        #
        # self.bn = nn.BatchNorm2d(num_features=self.out_channels)
        #
        # input_size = in_channels * resolution ** 2
        # representation_size = cfg.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM
        # use_gn = cfg.MODEL.ROI_BOX_HEAD.USE_GN
        # self.fc6 = make_fc(input_size, representation_size, use_gn)
        #
        # self.fc7 = make_fc(representation_size, self.out_channels, use_gn)
        # self.resize_channels = input_size

    def forward(self, x, proposals, depth_features=None):
        x = self.pooler(x, proposals)
        # x_seq = F.relu(self.bn(self.out_conv(x)))
        # x = x.view(x.size(0), -1)
        # x = F.relu(self.fc6(x))
        # x = F.relu(self.fc7(x))

        return x

    def forward_without_pool(self, x):
        # x_seq = F.relu(self.bn(self.out_conv(x)))
        # x = x.view(x.size(0), -1)
        # x = F.relu(self.fc6(x))
        # x = F.relu(self.fc7(x))
        return x


@registry.ROI_BOX_FEATURE_EXTRACTORS.register("FPNXconv1fcFeatureExtractor")
class FPNXconv1fcFeatureExtractor(nn.Module):
    """
    Heads for FPN for classification
    """

    def __init__(self, cfg, in_channels):
        super(FPNXconv1fcFeatureExtractor, self).__init__()

        resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        scales = cfg.MODEL.ROI_BOX_HEAD.POOLER_SCALES
        sampling_ratio = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        pooler = Pooler(
            output_size=(resolution, resolution),
            scales=scales,
            sampling_ratio=sampling_ratio,
        )
        self.pooler = pooler

        use_gn = cfg.MODEL.ROI_BOX_HEAD.USE_GN
        conv_head_dim = cfg.MODEL.ROI_BOX_HEAD.CONV_HEAD_DIM
        num_stacked_convs = cfg.MODEL.ROI_BOX_HEAD.NUM_STACKED_CONVS
        dilation = cfg.MODEL.ROI_BOX_HEAD.DILATION

        xconvs = []
        for ix in range(num_stacked_convs):
            xconvs.append(
                nn.Conv2d(
                    in_channels,
                    conv_head_dim,
                    kernel_size=3,
                    stride=1,
                    padding=dilation,
                    dilation=dilation,
                    bias=False if use_gn else True
                )
            )
            in_channels = conv_head_dim
            if use_gn:
                xconvs.append(group_norm(in_channels))
            xconvs.append(nn.ReLU(inplace=True))

        self.add_module("xconvs", nn.Sequential(*xconvs))
        for modules in [self.xconvs, ]:
            for l in modules.modules():
                if isinstance(l, nn.Conv2d):
                    torch.nn.init.normal_(l.weight, std=0.01)
                    if not use_gn:
                        torch.nn.init.constant_(l.bias, 0)

        input_size = conv_head_dim * resolution ** 2
        representation_size = cfg.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM
        self.fc6 = make_fc(input_size, representation_size, use_gn=False)
        self.out_channels = representation_size

    def forward(self, x, proposals):
        x = self.pooler(x, proposals)
        x = self.xconvs(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc6(x))
        return x


def make_roi_box_feature_extractor(cfg, in_channels, half_out=False, cat_all_levels=False, for_relation=False):
    feature_extractor = None
    # for relation head
    if for_relation:
        if cfg.MODEL.ROI_RELATION_HEAD.BOX_FEATURE_EXTRACTOR:
            feature_extractor = cfg.MODEL.ROI_RELATION_HEAD.BOX_FEATURE_EXTRACTOR
    #  default
    if feature_extractor is None:
        feature_extractor = cfg.MODEL.ROI_BOX_HEAD.FEATURE_EXTRACTOR

    func = registry.ROI_BOX_FEATURE_EXTRACTORS[feature_extractor]
    return func(cfg, in_channels, half_out, cat_all_levels)
