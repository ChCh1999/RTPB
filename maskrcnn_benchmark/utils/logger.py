# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import logging
import os
import sys

import wandb

from maskrcnn_benchmark.utils.comm import get_rank

DEBUG_PRINT_ON = True


def debug_print(logger, info):
    if DEBUG_PRINT_ON:
        logger.info('#' * 20 + ' ' + info + ' ' + '#' * 20)


def setup_logger(name, save_dir, distributed_rank, filename="log.txt"):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    # don't log results for the non-master process
    if distributed_rank > 0:
        return logger
    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    if save_dir:
        fh = logging.FileHandler(os.path.join(save_dir, filename))
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger


def log_result(tag, result):
    if get_rank() <= 0:
        for each_ds_eval in result:
            for each_evalator_res in each_ds_eval[1]:
                wandb.log({f"{tag}/{k}": v for k, v in each_evalator_res.items()})
