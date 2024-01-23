# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""
Basic training script for PyTorch
"""

import argparse
import datetime
import os
import time

import torch
import wandb
from omegaconf import OmegaConf

from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.config.utils import DEFAULT_CONFIG_FILE, check_config, set_config, flatten_conf
from maskrcnn_benchmark.data import make_data_loader
from maskrcnn_benchmark.engine.inference import inference
from maskrcnn_benchmark.engine.trainer import reduce_loss_dict
from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.solver import make_lr_scheduler
from maskrcnn_benchmark.solver import make_optimizer
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer
from maskrcnn_benchmark.utils.checkpoint import clip_grad_norm
from maskrcnn_benchmark.utils.collect_env import collect_env_info
from maskrcnn_benchmark.utils.comm import synchronize, get_rank, all_gather, seed_all
# Set up custom environment before nearly anything else is imported
# NOTE: this should be the first import (no not reorder)
from maskrcnn_benchmark.utils.env import setup_environment  # noqa F401 isort:skip
from maskrcnn_benchmark.utils.logger import setup_logger, debug_print, log_result
from maskrcnn_benchmark.utils.metric_logger import MetricLogger
from maskrcnn_benchmark.utils.miscellaneous import mkdir, save_config
from utils import show_params_status
from tqdm import tqdm

# See if we can use apex.DistributedDataParallel instead of the torch default,
# and enable mixed-precision via apex.amp


try:
    from apex import amp
except ImportError:
    raise ImportError('Use APEX for multi-precision via apex.amp')


def train(cfg, local_rank, distributed, logger, debug=False):
    debug_print(logger, 'prepare training')
    model = build_detection_model(cfg)
    debug_print(logger, 'end model construction')

    # modules that should be always set in eval mode
    # their eval() method should be called after model.train() is called
    eval_modules = (model.rpn, model.backbone, model.roi_heads.box,)

    fix_eval_modules(eval_modules)

    # NOTE, we slow down the LR of the layers start with the names in slow_heads
    if cfg.MODEL.ROI_RELATION_HEAD.PREDICTOR == "IMPPredictor":
        slow_heads = ["roi_heads.relation.box_feature_extractor",
                      "roi_heads.relation.union_feature_extractor.feature_extractor", ]
    else:
        slow_heads = []

    # load pretrain layers to new layers
    load_mapping = {"roi_heads.relation.box_feature_extractor": "roi_heads.box.feature_extractor",
                    "roi_heads.relation.union_feature_extractor.feature_extractor": "roi_heads.box.feature_extractor"}

    if cfg.MODEL.ATTRIBUTE_ON:
        load_mapping["roi_heads.relation.att_feature_extractor"] = "roi_heads.attribute.feature_extractor"
        load_mapping[
            "roi_heads.relation.union_feature_extractor.att_feature_extractor"] = "roi_heads.attribute.feature_extractor"

    logger.info(show_params_status(model))

    device = torch.device(cfg.MODEL.DEVICE)
    model.to(device)

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    num_batch = cfg.SOLVER.IMS_PER_BATCH
    optimizer = make_optimizer(cfg, model, logger, slow_heads=slow_heads, slow_ratio=10.0, rl_factor=float(num_batch))
    scheduler = make_lr_scheduler(cfg, optimizer, logger)
    debug_print(logger, 'end optimizer and shcedule')
    # Initialize mixed-precision training
    use_mixed_precision = cfg.DTYPE == "float16"
    amp_opt_level = 'O1' if use_mixed_precision else 'O0'
    model, optimizer = amp.initialize(model, optimizer, opt_level=amp_opt_level)

    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank,
            # this should be removed if we update BatchNorm stats
            broadcast_buffers=False,
            find_unused_parameters=True,
        )
    debug_print(logger, 'end distributed')
    arguments = {}
    arguments["iteration"] = 0

    output_dir = cfg.OUTPUT_DIR

    save_to_disk = get_rank() == 0
    checkpointer = DetectronCheckpointer(
        cfg, model, optimizer, scheduler, output_dir, save_to_disk, custom_scheduler=True
    )
    # if there is certain checkpoint in output_dir, load it, else load pretrained detector
    if checkpointer.has_checkpoint():
        extra_checkpoint_data = checkpointer.load(cfg.MODEL.PRETRAINED_DETECTOR_CKPT,
                                                  update_schedule=cfg.SOLVER.UPDATE_SCHEDULE_DURING_LOAD)
        arguments.update(extra_checkpoint_data)
    else:
        # load_mapping is only used when we init current model from detection model.
        checkpointer.load(cfg.MODEL.PRETRAINED_DETECTOR_CKPT, with_optim=False, load_mapping=load_mapping)
    debug_print(logger, 'end load checkpointer')

    train_data_loader = make_data_loader(
        cfg,
        mode='train',
        is_distributed=distributed,
        start_iter=arguments["iteration"],
    )
    val_data_loaders = make_data_loader(
        cfg,
        mode='val',
        is_distributed=distributed,
    )
    debug_print(logger, 'end dataloader')
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD

    if cfg.SOLVER.PRE_VAL:
        logger.info("Validate before training")
        run_val(cfg, model, val_data_loaders, distributed, logger)

    logger.info("Start training")
    meters = MetricLogger(delimiter="  ")
    max_iter = len(train_data_loader)
    start_iter = arguments["iteration"]
    start_training_time = time.time()
    end = time.time()

    start_training_time = time.time()
    print_first_grad = True

    for iteration, (images, targets, _) in enumerate(train_data_loader, start_iter):
        if any(len(target) < 1 for target in targets):
            logger.error(
                f"Iteration={iteration + 1} || Image Ids used for training {_} || targets Length={[len(target) for target in targets]}")
        data_time = time.time() - end
        iteration = iteration + 1
        arguments["iteration"] = iteration

        model.train()
        fix_eval_modules(eval_modules)

        images = images.to(device)
        targets = [target.to(device) for target in targets]

        loss_dict = model(images, targets, logger=logger)

        losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = reduce_loss_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())
        meters.update(loss=losses_reduced, **loss_dict_reduced)

        optimizer.zero_grad()
        # Note: If mixed precision is not used, this ends up doing nothing
        # Otherwise apply loss scaling for mixed-precision recipe
        with amp.scale_loss(losses, optimizer) as scaled_losses:
            scaled_losses.backward()

        # add clip_grad_norm from MOTIFS, tracking gradient, used for debug
        verbose = (iteration % cfg.SOLVER.PRINT_GRAD_FREQ) == 0 or print_first_grad  # print grad or not
        print_first_grad = False
        # cc turn off output of grad
        # verbose = False

        clip_grad_norm([(n, p) for n, p in model.named_parameters() if p.requires_grad],
                       max_norm=cfg.SOLVER.GRAD_NORM_CLIP, logger=logger, verbose=verbose, clip=True)

        optimizer.step()

        batch_time = time.time() - end
        end = time.time()
        elapsed_time = str(datetime.timedelta(seconds=int(end - start_training_time)))
        meters.update(time=batch_time, data=data_time)

        eta_seconds = meters.time.global_avg * (max_iter - iteration)
        eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

        log_window_size = cfg.LOG_WINDOW_SIZE
        if iteration % log_window_size == 0:
            log_data = {f"train/time/{k}" if "time" in k else f"train/{k}": v.avg for k, v in meters.meters.items()}
            log_data["curr_lr"] = float(optimizer.param_groups[0]["lr"])
            wandb.log(log_data)
        if iteration % 200 == 0 or iteration == max_iter:
            logger.info(
                meters.delimiter.join(
                    [
                        "\nexp: {instance_name}"
                        "\nrunning time: {elapsed_time}",
                        "eta: {eta}",
                        "\niter: {iter}/{max_iter}",
                        "lr: {lr:.6f}",
                        "max mem: {memory:.0f}",
                        "\n{meters}",
                    ]
                ).format(
                    instance_name=cfg.OUTPUT_DIR,
                    elapsed_time=elapsed_time,
                    eta=eta_string,
                    iter=iteration,
                    meters=str(meters),
                    lr=optimizer.param_groups[-1]["lr"],
                    max_iter=max_iter,
                    memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0,
                )
            )
        # print loss detail
        if iteration % 100 == 0:
            if debug:
                print('iter:{};  loss:{}'.format(iteration, str({k: v.item() for k, v in loss_dict_reduced.items()})))

        if iteration % checkpoint_period == 0:
            checkpointer.save("model_{:07d}".format(iteration), **arguments)
        if iteration == max_iter:
            checkpointer.save("model_final", **arguments)

        val_result = None  # used for scheduler updating
        if cfg.SOLVER.TO_VAL and iteration % cfg.SOLVER.VAL_PERIOD == 0:
            logger.info("Start validating")
            # torch.cuda.empty_cache()
            val_result = run_val(cfg, model, val_data_loaders, distributed, logger)
            logger.info("Validation Result: %.4f" % val_result[1])
            log_result("val", val_result[0])
        # scheduler should be called after optimizer.step() in pytorch>=1.1.0
        # https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate
        if cfg.SOLVER.SCHEDULE.TYPE == "WarmupReduceLROnPlateau":
            scheduler.step(val_result, epoch=iteration)
            if scheduler.stage_count >= cfg.SOLVER.SCHEDULE.MAX_DECAY_STEP:
                logger.info("Trigger MAX_DECAY_STEP at iteration {}.".format(iteration))
                checkpointer.save("model_final", **arguments)
                break
        else:
            scheduler.step()
        torch.cuda.empty_cache()

    total_training_time = time.time() - start_training_time
    total_time_str = str(datetime.timedelta(seconds=total_training_time))
    logger.info(
        "Total training time: {} ({:.4f} s / it)".format(
            total_time_str, total_training_time / (max_iter)
        )
    )
    return model


def fix_eval_modules(eval_modules):
    for module in eval_modules:
        for _, param in module.named_parameters():
            param.requires_grad = False
        # DO NOT use module.eval(),
        # otherwise the module will be in the test mode,
        # i.e., all self.training condition is set to False


def run_val(cfg, model, val_data_loaders, distributed, logger):
    if distributed:
        model = model.module
    torch.cuda.empty_cache()
    iou_types = ("bbox",)
    if cfg.MODEL.MASK_ON:
        iou_types = iou_types + ("segm",)
    if cfg.MODEL.KEYPOINT_ON:
        iou_types = iou_types + ("keypoints",)
    if cfg.MODEL.RELATION_ON:
        iou_types = iou_types + ("relations",)
    if cfg.MODEL.ATTRIBUTE_ON:
        iou_types = iou_types + ("attributes",)

    dataset_names = cfg.DATASETS.VAL
    val_result = []
    for dataset_name, val_data_loader in zip(dataset_names, val_data_loaders):
        dataset_result = inference(
            cfg,
            model,
            val_data_loader,
            dataset_name=dataset_name,
            iou_types=iou_types,
            box_only=False if cfg.MODEL.RETINANET_ON else cfg.MODEL.RPN_ONLY,
            device=cfg.MODEL.DEVICE,
            expected_results=cfg.TEST.EXPECTED_RESULTS,
            expected_results_sigma_tol=cfg.TEST.EXPECTED_RESULTS_SIGMA_TOL,
            output_folder=None,
            logger=logger,
        )
        synchronize()
        val_result.append(dataset_result)
    val_values = []
    for each in val_result:
        if isinstance(each, tuple):
            val_values.append(each[0])
    # support for multi gpu distributed testing
    gathered_result = all_gather(torch.tensor(val_values).cpu())
    gathered_result = [t.view(-1) for t in gathered_result]
    gathered_result = torch.cat(gathered_result, dim=-1).view(-1)
    valid_result = gathered_result[gathered_result >= 0]
    val_result_val = float(valid_result.mean())
    del gathered_result, valid_result
    torch.cuda.empty_cache()
    return val_result, val_result_val


def run_test(cfg, model, distributed, logger):
    if distributed:
        model = model.module
    torch.cuda.empty_cache()
    iou_types = ("bbox",)
    if cfg.MODEL.MASK_ON:
        iou_types = iou_types + ("segm",)
    if cfg.MODEL.KEYPOINT_ON:
        iou_types = iou_types + ("keypoints",)
    if cfg.MODEL.RELATION_ON:
        iou_types = iou_types + ("relations",)
    if cfg.MODEL.ATTRIBUTE_ON:
        iou_types = iou_types + ("attributes",)
    output_folders = [None] * len(cfg.DATASETS.TEST)
    dataset_names = cfg.DATASETS.TEST
    if cfg.OUTPUT_DIR:
        for idx, dataset_name in enumerate(dataset_names):
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference", dataset_name)
            mkdir(output_folder)
            output_folders[idx] = output_folder
    data_loaders_val = make_data_loader(cfg, mode='test', is_distributed=distributed)
    test_result = []
    for output_folder, dataset_name, data_loader_val in zip(output_folders, dataset_names, data_loaders_val):
        dataset_result = inference(
            cfg,
            model,
            data_loader_val,
            dataset_name=dataset_name,
            iou_types=iou_types,
            box_only=False if cfg.MODEL.RETINANET_ON else cfg.MODEL.RPN_ONLY,
            device=cfg.MODEL.DEVICE,
            expected_results=cfg.TEST.EXPECTED_RESULTS,
            expected_results_sigma_tol=cfg.TEST.EXPECTED_RESULTS_SIGMA_TOL,
            output_folder=output_folder,
            logger=logger,
        )
        synchronize()
        test_result.append(dataset_result)
    return test_result


def main(debug=False):
    parser = argparse.ArgumentParser(description="PyTorch Relation Detection Training")
    parser.add_argument(
        "-c",
        "--config-file",
        nargs="+",
        default=list(),
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument(
        "--skip-test",
        dest="skip_test",
        help="Do not test the final model",
        action="store_true",
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    parser.add_argument("-s", action="store_true")

    args = parser.parse_args()

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.distributed = num_gpus > 1
    local_rank = int(os.environ['LOCAL_RANK']) if "WORLD_SIZE" in os.environ else 0
    if args.distributed:
        torch.cuda.set_device(local_rank)
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://"
        )
        synchronize()

    default_config = OmegaConf.load(DEFAULT_CONFIG_FILE)
    loaded_cfg = [OmegaConf.load(cfg_file) for cfg_file in args.config_file]
    ext_configs = OmegaConf.from_dotlist(args.opts)
    cfg = OmegaConf.merge(default_config, *loaded_cfg, ext_configs)
    # cfg.freeze()

    # mode
    if cfg.MODEL.ROI_RELATION_HEAD.USE_GT_BOX:
        if cfg.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL:
            mode = "predcls"
        else:
            mode = "sgcls"
    else:
        mode = "sgdet"

    now = datetime.datetime.now()
    time_str = now.strftime("%Y-%m-%d_%H")
    exp_name = f"{cfg.EXPERIMENT_NAME}-{time_str}" \
               + f"{'-debug' if cfg.DEBUG else ''}" \
               + "-rtpb"
    cfg.OUTPUT_DIR = os.path.join(
        cfg.OUTPUT_DIR,
        f"{mode}-{cfg.MODEL.ROI_RELATION_HEAD.PREDICTOR}",
        exp_name
    )
    # init output dir
    output_dir = cfg.OUTPUT_DIR
    if output_dir:
        mkdir(output_dir)

    cfg = check_config(cfg)
    OmegaConf.set_readonly(cfg, True)
    set_config(cfg)

    # set seed
    SEED = cfg.SEED
    seed_all(SEED)

    # init wandb
    wandb.init(
        # set the wandb project where this run will be logged
        project="LGSGG",
        name=f"{mode}-{exp_name}",
        # track hyperparameters and run metadata
        config=flatten_conf(cfg)
    )

    logger = setup_logger("maskrcnn_SGG", output_dir, get_rank())
    logger.info("Using {} GPUs".format(num_gpus))
    logger.info(args)

    if cfg.DEBUG:
        logger.info("Collecting env info (might take some time)")
        logger.info("\n" + collect_env_info())

    logger.info("Running with config:\n{}".format(cfg))
    output_config_path = os.path.join(cfg.OUTPUT_DIR, 'config.yml')
    logger.info("Saving config into: {}".format(output_config_path))
    # save overloaded model config in the output directory
    save_config(cfg, output_config_path)

    model = train(cfg, local_rank, args.distributed, logger, debug=debug)

    if not args.skip_test:
        test_result = run_test(cfg, model, args.distributed, logger)
        log_result("test", test_result)
    wandb.finish()


if __name__ == "__main__":
    import sys

    print('running with system paths :', sys.path)
    main()
