DET_CKPT=model/detector_base_x32/model_final.pth
OUT_PATH=/path/to/output/
GLOVE_DIR=/path/to/glove/

python tools/relation_train_net.py --config-file 'configs/e2e_relation_X_101_32_8_FPN_1x_trans_sgdet.yaml' SOLVER.IMS_PER_BATCH 16 TEST.IMS_PER_BATCH 1 DTYPE float32 SOLVER.MAX_ITER 18000 SOLVER.STEPS '(10000,16000)' SOLVER.WARMUP_ITERS 500 GLOVE_DIR $GLOVE_DIR MODEL.PRETRAINED_DETECTOR_CKPT $DET_CKPT OUTPUT_DIR $OUT_PATH SOLVER.PRE_VAL False MODEL.ROI_RELATION_HEAD.PREDICTOR 'DualTransPredictor' MODEL.ROI_RELATION_HEAD.DUAL_TRANS.USE_GRAPH_ENCODE True MODEL.ROI_RELATION_HEAD.DUAL_TRANS.GRAPH_ENCODE_STRATEGY trans MODEL.ROI_RELATION_HEAD.BIAS_MODULE.USE_PENALTY True MODEL.ROI_RELATION_HEAD.BIAS_MODULE.PENALTY_BIAS.PENALTY_TYPE 'margin_loss'