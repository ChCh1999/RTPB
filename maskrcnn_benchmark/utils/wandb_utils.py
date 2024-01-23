import wandb
import random


def init(wandb_config):
    wandb.init(
        # set the wandb project where this run will be logged
        project="mySGG",

        # track hyperparameters and run metadata
        config=wandb_config
    )
