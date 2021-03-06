DET_CKPT=model/detector_base_x32/model_final.pth
OUT_PATH=/path/to/output/
GLOVE_DIR=/path/to/glove/

python tools/relation_train_net.py --config-file 'configs/e2e_relation_X_101_32_8_FPN_1x_trans_sgdet.yaml' SOLVER.IMS_PER_BATCH 16 TEST.IMS_PER_BATCH 1 DTYPE float32 SOLVER.MAX_ITER 18000 SOLVER.STEPS '(10000,16000)' SOLVER.WARMUP_ITERS 500 GLOVE_DIR $GLOVE_DIR MODEL.PRETRAINED_DETECTOR_CKPT /public/data1/users/chenchao278/model/detector_base_x32/model_final.pth SOLVER.PRE_VAL False MODEL.ROI_RELATION_HEAD.PREDICTOR 'MotifPredictor' OUTPUT_DIR /public/data1/users/chenchao278/model/sgdet/motif_base MODEL.ROI_RELATION_HEAD.PREDICT_USE_BIAS True