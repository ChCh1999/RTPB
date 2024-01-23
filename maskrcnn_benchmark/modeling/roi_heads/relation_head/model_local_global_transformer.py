import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from maskrcnn_benchmark.data import get_dataset_statistics
from maskrcnn_benchmark.modeling import registry

from maskrcnn_benchmark.modeling.utils import cat
from .loss import make_roi_relation_loss_evaluator
from .modules.bias_module import build_bias_module
from .utils_motifs import obj_edge_vectors, to_onehot, nms_overlaps, encode_box_info
from .utils_relation import layer_init, nms_per_cls
from torch.nn.utils.rnn import pad_sequence


def make_fc(dim_in, dim_out):
    fc = nn.Linear(dim_in, dim_out)
    nn.init.kaiming_uniform_(fc.weight, a=1)
    nn.init.constant_(fc.bias, 0)
    return fc


class ObjectEmbedding(nn.Module):
    def __init__(self, cfg, num_obj_cls, ebd_dim):
        self.obj_embedding = nn.Embedding(num_obj_cls, ebd_dim)
        self.cfg = cfg

    def forward(self, proposals):
        if self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL:
            obj_labels = cat([proposal.get_field("labels") for proposal in proposals], dim=0)
            obj_cls_embed = self.obj_embedding(obj_labels.long())
        else:
            obj_logits = cat([proposal.get_field("predict_logits") for proposal in proposals], dim=0).detach()
            obj_cls_embed = F.softmax(obj_logits, dim=1) @ self.obj_embedding.weight
        return obj_cls_embed


class RoIFeaturePooler(nn.Module):
    def __init__(self, config, in_dim, out_dim):
        super(RoIFeaturePooler, self).__init__()
        # ROI_RESOLUTION should be same as the size of roi feature. [N,C,R,R]
        roi_feature_resolution = config.MODEL.ROI_RELATION_HEAD.LG_TRANS.ROI_RESOLUTION
        max_length = roi_feature_resolution ** 2

        # token_pooling_dim = config.MODEL.ROI_RELATION_HEAD.LG_TRANS.TOKEN_POOLING_DIM

        hidden_dim = 4096
        self.token_linear_out = self.token_linear_out = nn.Sequential(*[
            make_fc(in_dim * max_length, hidden_dim),
            nn.ReLU(),
            make_fc(hidden_dim, out_dim),
            nn.ReLU()
        ])

    def forward(self, roi_feature):
        obj_cnt = roi_feature.size(0)
        return self.token_linear_out(roi_feature.view(obj_cnt, -1))


class ObjectLocalEncoder(nn.Module):
    def __init__(self, config):
        super(ObjectLocalEncoder, self).__init__()
        self.cfg = config
        in_channel = config.MODEL.ROI_RELATION_HEAD.CONTEXT_ROI_DIM
        out_dim = in_channel
        # ROI_RESOLUTION should be same as the size of roi feature. [N,C,R,R]
        roi_feature_resolution = config.MODEL.ROI_RELATION_HEAD.LG_TRANS.ROI_RESOLUTION
        max_length = roi_feature_resolution ** 2

        self.cls_token = nn.Parameter(torch.randn(1, 1, in_channel))

        self.positional_embedding = nn.init.xavier_normal_(nn.Parameter(torch.randn(1, max_length, in_channel)))
        num_obj_cls = config.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        self.obj_embedding = ObjectEmbedding(self.cfg, num_obj_cls, in_channel)

        encoder_params = config.MODEL.ROI_RELATION_HEAD.LG_TRANS.TRANSFORMER_PARAMS
        encoder_dim = encoder_params.d_model
        encoder_layer = nn.TransformerEncoderLayer(**encoder_params, batch_first=True)
        self.enc_layers = config.MODEL.ROI_RELATION_HEAD.LG_TRANS.OBJ_ENC_LAYERS
        self.layer_norm = nn.LayerNorm(encoder_dim)
        self.layer_norm = None
        if self.enc_layers > 0:
            self.encoder = nn.TransformerEncoder(encoder_layer, self.enc_layers, norm=self.layer_norm)
        self.obj_pooler = nn.MaxPool1d(kernel_size=max_length)

        self.linear_in = None
        self.linear_out = None
        if in_channel != encoder_dim:
            self.linear_in = make_fc(in_channel, encoder_dim)
        if encoder_dim != out_dim:
            self.linear_out = make_fc(encoder_dim, out_dim)

        self.use_token_pooling = config.MODEL.ROI_RELATION_HEAD.LG_TRANS.USE_TOKEN_POOLING
        self.roi_token_pooler = RoIFeaturePooler(config, in_channel, out_dim)

    def forward(self, roi_features, proposals):
        """

        Args:
            roi_features: object patch features [N,C,R,R]
            proposals: list of BoxList, [BoxList]

        Returns:

        """
        obj_cnt, C, R, _ = roi_features.size()
        total_patch_cnt = R * R
        # compose object feature
        roi_features = roi_features.permute(0, 2, 3, 1).contiguous().view(obj_cnt, total_patch_cnt, C)
        roi_features = roi_features_copy = roi_features + self.positional_embedding

        if self.enc_layers == 0:
            return self.roi_token_pooler(roi_features_copy)

        obj_cls_embed = self.obj_embedding(proposals)
        cls_token = self.cls_token.expand(obj_cnt, -1, -1)
        roi_features = torch.cat([cls_token, roi_features, obj_cls_embed.unsqueeze(1)], dim=1)
        # process
        if self.linear_in:
            roi_features = self.linear_in(roi_features)

        roi_features = F.relu(self.encoder(roi_features))

        if self.linear_out:
            roi_features = F.relu(self.linear_out(roi_features))
        # roi_features = roi_features.mean(dim=1)
        roi_features = roi_features[:, 0, :]
        if self.use_token_pooling:
            roi_features = roi_features + self.roi_token_pooler(roi_features_copy)
        return roi_features


class RelationshipLocalEncoder(nn.Module):
    def __init__(self, config):
        super(RelationshipLocalEncoder, self).__init__()
        self.cfg = config

        roi_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_ROI_DIM
        out_dim = config.MODEL.ROI_RELATION_HEAD.LG_TRANS.MODEL_DIM
        # ROI_RESOLUTION should be same as the size of roi feature. [N,C,R,R]
        roi_feature_resolution = config.MODEL.ROI_RELATION_HEAD.LG_TRANS.ROI_RESOLUTION
        max_length = roi_feature_resolution ** 2
        self.include_obj = self.cfg.MODEL.ROI_RELATION_HEAD.LG_TRANS.REL_ENC_INCLUDE_OBJ
        self.cls_token = nn.Parameter(torch.randn(1, 1, roi_dim))
        self.positional_embedding = nn.init.xavier_normal_(nn.Parameter(torch.randn(1, max_length, roi_dim)))
        self.obj_tag_embedding = nn.init.xavier_uniform_(nn.Parameter(torch.randn(1, 2, roi_dim)))

        num_obj_cls = config.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        self.obj_cls_embedding = nn.Embedding(num_obj_cls, roi_dim)
        self.obj_cls_tag_embedding = nn.init.xavier_uniform_(nn.Parameter(torch.randn(1, 2, roi_dim)))

        encoder_params = config.MODEL.ROI_RELATION_HEAD.LG_TRANS.TRANSFORMER_PARAMS
        encoder_dim = encoder_params.d_model
        self.layer_norm = nn.LayerNorm(encoder_dim)
        self.layer_norm = None
        encoder_layer = nn.TransformerEncoderLayer(**encoder_params, batch_first=True)
        self.enc_layers = config.MODEL.ROI_RELATION_HEAD.LG_TRANS.REL_ENC_LAYERS
        if self.enc_layers > 0:
            self.encoder = nn.TransformerEncoder(encoder_layer, self.enc_layers, norm=self.layer_norm)

        self.linear_in = None
        self.linear_out = None
        if roi_dim != encoder_dim:
            self.linear_in = nn.Linear(roi_dim, encoder_dim)
        if encoder_dim != out_dim:
            self.linear_out = nn.Linear(encoder_dim, out_dim)

        self.use_token_pooling = config.MODEL.ROI_RELATION_HEAD.LG_TRANS.USE_TOKEN_POOLING
        if self.use_token_pooling:
            self.roi_token_pooler = RoIFeaturePooler(config, roi_dim, out_dim)

    def forward(self, obj_feature, proposals, union_roi_feature, rel_pair_idxs, num_objs):
        """

        Args:
            obj_feature: total_obj * d_model
            proposals: bbox
            union_roi_feature: total_rel * R * R * in_channel(roi feature)
            rel_pair_idxs: [(n_rel * 2) for BS]
            num_objs: [n_obj for BS]

        Returns:

        """
        rel_cnt, C, R, _ = union_roi_feature.size()
        BS = len(num_objs)
        # ===== compose rel feature =====
        # position embedding
        rel_features = union_roi_feature.permute(0, 2, 3, 1).contiguous().view(rel_cnt, -1, C)

        rel_features = rel_features_copy = rel_features + self.positional_embedding
        if self.enc_layers == 0:
            return self.roi_token_pooler(rel_features_copy)
        # concat obj feature
        num_rels = [r.shape[0] for r in rel_pair_idxs]
        rel_features = list(torch.split(rel_features, num_rels, dim=0))
        obj_feature = torch.split(obj_feature, num_objs, dim=0)
        if self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL:
            obj_labels = cat([proposal.get_field("labels") for proposal in proposals], dim=0)
            obj_cls_feature = self.obj_cls_embedding(obj_labels.long())
        else:
            obj_logits = cat([proposal.get_field("predict_logits") for proposal in proposals], dim=0).detach()
            obj_cls_feature = F.softmax(obj_logits, dim=1) @ self.obj_cls_embedding.weight
        obj_cls_feature = torch.split(obj_cls_feature, num_objs, dim=0)
        for i in range(BS):
            pair_idx = rel_pair_idxs[i]

            cls_token = self.cls_token.expand(len(pair_idx), -1, -1)
            if self.include_obj:
                obj_feature_i = obj_feature[i]
                rel_obj_feature_i = torch.stack((obj_feature_i[pair_idx[:, 0]], obj_feature_i[pair_idx[:, 1]]), dim=1)
                rel_obj_feature_i = rel_obj_feature_i + self.obj_tag_embedding

                obj_cls_feature_i = obj_cls_feature[i]
                rel_obj_cls_feature_i = torch.stack(
                    (obj_cls_feature_i[pair_idx[:, 0]], obj_cls_feature_i[pair_idx[:, 1]]),
                    dim=1)
                rel_obj_cls_feature_i = rel_obj_cls_feature_i + self.obj_cls_tag_embedding

                rel_features[i] = torch.cat([cls_token, rel_features[i], rel_obj_feature_i, rel_obj_cls_feature_i],
                                            dim=1)
            else:
                rel_features[i] = torch.cat([cls_token, rel_features[i]], dim=1)
        rel_features = torch.cat(rel_features, dim=0)
        # process
        if self.linear_in:
            rel_features = self.linear_in(rel_features)

        rel_features = self.encoder(rel_features)
        if self.linear_out:
            rel_features = self.linear_out(rel_features)

        # rel_features = rel_features.mean(dim=1)
        rel_features = rel_features[:, 0, :]
        if self.use_token_pooling:
            rel_features = rel_features + self.roi_token_pooler(rel_features_copy)
        return rel_features


class RelationshipContextDecoder(nn.Module):
    def __init__(self, config):
        super(RelationshipContextDecoder, self).__init__()
        self.cfg = config

        roi_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_ROI_DIM
        model_dim = config.MODEL.ROI_RELATION_HEAD.LG_TRANS.MODEL_DIM
        transformer_params = dict(config.MODEL.ROI_RELATION_HEAD.LG_TRANS.TRANSFORMER_PARAMS)
        transformer_dim = transformer_params["d_model"] = model_dim
        self.obj_linear_in = None
        if roi_dim != model_dim:
            self.obj_linear_in = nn.Linear(roi_dim, model_dim)
        self.obj_embedding = nn.init.xavier_normal_(nn.Parameter(torch.randn(1, model_dim)))
        self.rel_embedding = nn.init.xavier_normal_(nn.Parameter(torch.randn(1, model_dim)))
        decoder_layer = nn.TransformerDecoderLayer(**transformer_params, batch_first=True)
        self.n_layers = config.MODEL.ROI_RELATION_HEAD.LG_TRANS.CTX_DEC_LAYERS
        self.obj_decoder = nn.TransformerDecoder(decoder_layer, self.n_layers, norm=nn.LayerNorm(transformer_dim))
        self.rel_decoder = nn.TransformerDecoder(decoder_layer, self.n_layers, norm=nn.LayerNorm(transformer_dim))
        self.joint_decoding = config.MODEL.ROI_RELATION_HEAD.LG_TRANS.JOINT_DECODING

    def forward(self, rel_features, obj_features, proposals, num_rels, num_obj, **kwargs):
        if self.n_layers == 0:
            return rel_features

        if self.obj_linear_in is not None:
            obj_features = self.obj_linear_in(obj_features)

        rel_features = self.ctx_decoding(rel_features, obj_features, num_rels, num_obj)
        # rel_features = list(torch.split(rel_features, num_rels, dim=0))
        # split_obj_features = list(torch.split(obj_features, num_obj, dim=0))
        # for i in range(BS):
        #     rel_features_i = rel_features[i]
        #     obj_features_i = obj_features[i]
        #     if self.joint_decoding:
        #         # jointly decoding
        #         rel_features_i = rel_features_i + self.rel_embedding
        #         ctx_features = torch.cat((rel_features_i, obj_features_i), dim=0).unsqueeze(0)
        #         rel_features_i = rel_features_i.unsqueeze(0)
        #         rel_features[i] = self.rel_decoder(rel_features_i, ctx_features).squeeze(0)
        #     else:
        #         # separately decoding
        #         rel_features_i = rel_features_i.unsqueeze(0)
        #         obj_features_i = obj_features_i.unsqueeze(0)
        #         rel_features_i = self.obj_decoder(rel_features_i, obj_features_i)
        #         rel_features_i = self.rel_decoder(rel_features_i, rel_features_i)
        #         rel_features[i] = rel_features_i.squeeze(0)
        #
        # rel_features = torch.cat(rel_features, dim=0)
        return rel_features

    def ctx_decoding(self, rel_features, obj_features, num_rels, num_obj):
        BS = len(num_rels)
        split_rel_features = list(torch.split(rel_features, num_rels, dim=0))
        split_obj_features = list(torch.split(obj_features, num_obj, dim=0))
        if self.joint_decoding:
            ctx_features = []
            for i in range(BS):
                rel_features_i = split_rel_features[i]
                obj_features_i = split_obj_features[i]
                ctx_features_i = torch.cat((rel_features_i + self.rel_embedding, obj_features_i + self.obj_embedding),
                                           dim=0)
                ctx_features.append(ctx_features_i)

            ctx_features = pad_sequence(ctx_features, batch_first=True)
            ctx_mask = ctx_features.eq(0).sum(2).eq(ctx_features.size(2))
            ctx_mask[:, 0] = False
            dec_rel_features = pad_sequence(split_rel_features, batch_first=True)
            rel_mask = dec_rel_features.eq(0).sum(2).eq(dec_rel_features.size(2))
            rel_mask[:, 0] = False
            dec_rel_features = self.rel_decoder(dec_rel_features, ctx_features, tgt_key_padding_mask=rel_mask,
                                                memory_key_padding_mask=ctx_mask)
            indices = torch.arange(dec_rel_features.size(1)).unsqueeze(0)
            mask = indices < torch.tensor(num_rels).unsqueeze(1)
            rel_features = dec_rel_features[mask]
        else:
            ctx_rel_features = pad_sequence(split_rel_features, batch_first=True)
            ctx_rel_mask = ctx_rel_features.eq(0).sum(2).eq(ctx_rel_features.size(2))
            ctx_obj_features = pad_sequence(split_obj_features, batch_first=True)
            ctx_obj_mask = ctx_obj_features.eq(0).sum(2).eq(ctx_obj_features.size(2))
            ctx_rel_mask[:, 0] = False
            ctx_obj_mask[:, 0] = False
            dec_rel_features = pad_sequence(split_rel_features, batch_first=True)
            rel_mask = dec_rel_features.eq(0).sum(2).eq(dec_rel_features.size(2))
            rel_mask[:, 0] = False

            dec_rel_features = self.obj_decoder(dec_rel_features, ctx_obj_features, tgt_key_padding_mask=rel_mask,
                                                memory_key_padding_mask=ctx_obj_mask)
            dec_rel_features = self.rel_decoder(dec_rel_features, ctx_rel_features, tgt_key_padding_mask=rel_mask,
                                                memory_key_padding_mask=ctx_rel_mask)
            # dec_rel_features = self.obj_decoder(dec_rel_features, ctx_obj_features)
            # dec_rel_features = self.rel_decoder(dec_rel_features, ctx_rel_features)
            indices = torch.arange(dec_rel_features.size(1)).unsqueeze(0)
            mask = indices < torch.tensor(num_rels).unsqueeze(1)
            rel_features = dec_rel_features[mask]
        return rel_features, obj_features


class RelationshipContextDualEncoder(nn.Module):
    def __init__(self, config):
        super(RelationshipContextDualEncoder, self).__init__()
        self.cfg = config

        roi_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_ROI_DIM
        model_dim = config.MODEL.ROI_RELATION_HEAD.LG_TRANS.MODEL_DIM
        transformer_params = dict(config.MODEL.ROI_RELATION_HEAD.LG_TRANS.TRANSFORMER_PARAMS)
        transformer_dim = transformer_params["d_model"] = model_dim

        self.bbox_embed = nn.Sequential(*[
            nn.Linear(9, 32), nn.ReLU(inplace=True), nn.Dropout(0.1),
            nn.Linear(32, 128), nn.ReLU(inplace=True), nn.Dropout(0.1),
        ])
        self.obj_linear_in = nn.Linear(roi_dim + 128, model_dim)

        self.rel_linear_in = nn.Linear(model_dim * 3, model_dim)

        encoder_layer = nn.TransformerEncoderLayer(**transformer_params, batch_first=True)
        # self.n_layers = config.MODEL.ROI_RELATION_HEAD.LG_TRANS.CTX_DEC_LAYERS
        self.obj_n_layers = config.MODEL.ROI_RELATION_HEAD.LG_TRANS.CTX_OBJ_ENC_LAYERS
        self.rel_n_layers = config.MODEL.ROI_RELATION_HEAD.LG_TRANS.CTX_REL_ENC_LAYERS
        self.obj_encoder = nn.TransformerEncoder(encoder_layer, self.obj_n_layers, norm=nn.LayerNorm(transformer_dim))
        self.rel_encoder = nn.TransformerEncoder(encoder_layer, self.rel_n_layers, norm=nn.LayerNorm(transformer_dim))

    def forward(self, rel_features, obj_features, proposals, num_rels, num_obj, rel_pair_idxs):
        if self.obj_n_layers == 0 and self.rel_n_layers:
            return rel_features
        BS = len(num_rels)
        rel_features = list(torch.split(rel_features, num_rels, dim=0))
        assert proposals[0].mode == 'xyxy'
        obj_pos_embedding = self.bbox_embed(encode_box_info(proposals))
        obj_features = torch.cat([obj_features, obj_pos_embedding], dim=-1)
        obj_features = self.obj_linear_in(obj_features)
        obj_features = list(torch.split(obj_features, num_obj, dim=0))
        for i in range(BS):
            rel_features_i = rel_features[i]
            rel_features_i_copy = rel_features_i.clone()
            obj_features_i = obj_features[i]
            obj_features_i_copy = obj_features_i.clone()
            # obj ctx encoding
            if self.obj_n_layers != 0:
                obj_features_i = obj_features_i.unsqueeze(0)
                obj_features_i = self.obj_encoder(obj_features_i)
                obj_features_i = obj_features_i.squeeze(0)
            pair_idx = rel_pair_idxs[i]
            rel_features_i = torch.cat((rel_features_i, obj_features_i[pair_idx[:, 0]], obj_features_i[pair_idx[:, 1]]),
                                       dim=-1)
            rel_features_i = self.rel_linear_in(rel_features_i)
            # rel ctx encoding
            if self.rel_n_layers != 0:
                rel_features_i = self.rel_encoder(rel_features_i.unsqueeze(0)).squeeze(0)

            rel_features[i] = rel_features_i
            rel_features[i] = rel_features[i] + rel_features_i_copy

            obj_features[i] = obj_features_i + obj_features_i_copy
        rel_features = torch.cat(rel_features, dim=0)
        obj_features = torch.cat(obj_features, dim=0)
        return rel_features, obj_features


@registry.ROI_RELATION_PREDICTOR.register("LocalGlobalTransPredictor")
class LocalGlobalTransPredictor(nn.Module):
    def __init__(self, config, in_channels):
        super(LocalGlobalTransPredictor, self).__init__()
        self.cfg = config
        if self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_BOX:
            self.mode = 'predcls' if self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL else 'sgcls'
        else:
            self.mode = 'sgdet'
        # load parameters
        self.num_obj_cls = config.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        self.num_att_cls = config.MODEL.ROI_ATTRIBUTE_HEAD.NUM_ATTRIBUTES
        self.num_rel_cls = config.MODEL.ROI_RELATION_HEAD.NUM_CLASSES
        self.nms_thresh = self.cfg.TEST.RELATION.LATER_NMS_PREDICTION_THRES
        self.hidden_dim = config.MODEL.ROI_RELATION_HEAD.LG_TRANS.MODEL_DIM
        self.roi_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_ROI_DIM

        self.use_obj_clf_enc = config.MODEL.ROI_RELATION_HEAD.LG_TRANS.OBJ_CLF_ENC
        assert in_channels is not None
        self.use_vision = config.MODEL.ROI_RELATION_HEAD.PREDICT_USE_VISION
        self.use_local_encoding = config.MODEL.ROI_RELATION_HEAD.LG_TRANS.USE_LOCAL_ENC
        self.epsilon = 0.001

        # load class dict
        statistics = get_dataset_statistics(config)
        obj_classes, rel_classes = statistics['obj_classes'], statistics['rel_classes']
        assert self.num_obj_cls == len(obj_classes)
        assert self.num_rel_cls == len(rel_classes)
        # ##################### init model #####################
        if self.use_local_encoding:
            self.obj_enc = ObjectLocalEncoder(config)
            if self.use_obj_clf_enc:
                self.obj_enc_clf = ObjectLocalEncoder(config)
            self.rel_enc = RelationshipLocalEncoder(config)
        else:
            box_extractor = config.MODEL.ROI_RELATION_HEAD.BOX_FEATURE_EXTRACTOR
            if box_extractor == "FPN2MLPFeatureExtractor":
                self.obj_roi_linear_in = nn.Linear(self.roi_dim, self.hidden_dim)
                self.rel_roi_linear_in = nn.Linear(self.roi_dim, self.hidden_dim)

        ctx_layer_name = config.MODEL.ROI_RELATION_HEAD.LG_TRANS.CTX_LAYER
        if ctx_layer_name == "decoder":
            self.ctx_dec = RelationshipContextDecoder(config)
        else:
            self.ctx_dec = RelationshipContextDualEncoder(config)

        self.rel_clf = nn.Linear(self.hidden_dim, self.num_rel_cls)
        self.obj_clf = nn.Linear(self.hidden_dim, self.num_obj_cls)
        # bias module
        self.bias_module = build_bias_module(config, statistics)

        # loss
        self.loss_evaluator = make_roi_relation_loss_evaluator(config)
        self.init_parameters()

    def init_parameters(self):
        def init_weights(m):
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LSTM):
                m.reset_parameters()

        self.apply(init_weights)

    def forward(self, proposals, rel_pair_idxs, rel_labels, roi_features, union_features, logger=None, **kwargs):
        """
        Predicate rel label
        Args:
            proposals: objs
            rel_pair_idxs: object pair index of rel to be predicated
            rel_labels: ground truth rel label
            roi_features: visual feature of objs
            union_features: visual feature of the union boxes of obj pairs
            logger: Logger tool

        Returns:
            obj_dists (list[Tensor]): logits of object label distribution
            rel_dists (list[Tensor])
            rel_pair_idxs (list[Tensor]): (num_rel, 2) index of subject and object
        """

        add_losses = {}
        num_rels = [r.shape[0] for r in rel_pair_idxs]
        num_objs = [len(b) for b in proposals]
        assert len(num_rels) == len(num_objs)
        rel_label_gt = torch.cat(rel_labels, dim=0) if rel_labels is not None else None

        use_gt_label = self.training or self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL
        obj_labels = cat([proposal.get_field("labels") for proposal in proposals], dim=0) if use_gt_label else None
        # encoding and decoding
        if self.use_local_encoding:
            obj_features = self.obj_enc(roi_features, proposals)
            rel_features = self.rel_enc(obj_features, proposals, union_features, rel_pair_idxs, num_objs)
        else:
            obj_features = roi_features
            rel_features = self.rel_roi_linear_in(union_features)

        # decoding
        rel_features, obj_features = self.ctx_dec(rel_features, obj_features, proposals, num_rels, num_objs,
                                                  rel_pair_idxs=rel_pair_idxs)

        obj_dists_raw, obj_preds_raw = self.obj_classification(obj_features, proposals, num_objs, obj_labels)
        obj_dists = obj_dists_raw.split(num_objs, dim=0)
        obj_preds = obj_preds_raw.split(num_objs, dim=0)
        # ### relationship classification
        rel_dists = self.rel_classification(rel_features, union_features)

        # ### use bias module
        pair_preds = []
        for pair_idx, obj_pred in zip(rel_pair_idxs, obj_preds):
            pair_preds.append(torch.stack((obj_pred[pair_idx[:, 0]], obj_pred[pair_idx[:, 1]]), dim=1))
        obj_pair_labels = cat(pair_preds, dim=0)
        bias = self.bias_module(obj_pair_labels=obj_pair_labels, num_rels=num_rels, obj_preds=obj_preds,
                                gt=rel_label_gt, bbox=[proposal.bbox for proposal in proposals],
                                rel_pair_idxs=rel_pair_idxs)
        if bias is not None:
            rel_dists = rel_dists + bias

        # post process and loss function
        rel_dists = rel_dists.split(num_rels, dim=0)
        if self.training:
            loss_relation, loss_refine = self.loss_evaluator(proposals, rel_labels, rel_dists, obj_dists)
            add_losses["loss_rel"] = loss_relation
            if loss_refine.item() != 0:
                add_losses["loss_refine_obj"] = loss_refine
        return obj_dists, rel_dists, add_losses

    def rel_classification(self, rel_features, union_features):
        rel_dists = self.rel_clf(rel_features)
        # remove bias

        # use union box and mask convolution
        if self.use_vision:
            ctx_gate = self.post_rel2ctx(rel_features)
            if self.union_single_not_match:
                visual_rep = ctx_gate * self.up_dim(union_features)
            else:
                visual_rep = ctx_gate * union_features
            rel_dists = rel_dists + self.rel_visual_clf(visual_rep)

        return rel_dists

    def obj_classification(self, obj_feats=None, proposals=None, num_objs=None, obj_labels=None):
        if self.mode == 'predcls':
            assert obj_labels is not None
            obj_preds = obj_labels
            obj_dists = to_onehot(obj_preds, self.num_obj_cls)
        else:
            assert obj_feats is not None
            obj_dists = self.obj_clf(obj_feats)
            use_decoder_nms = self.mode == 'sgdet' and not self.training
            if use_decoder_nms:
                assert proposals is not None and num_objs is not None
                boxes_per_cls = [proposal.get_field('boxes_per_cls') for proposal in proposals]
                obj_preds = nms_per_cls(obj_dists, boxes_per_cls, num_objs, self.nms_thresh)
            else:
                obj_preds = obj_dists[:, 1:].max(1)[1] + 1

        return obj_dists, obj_preds
