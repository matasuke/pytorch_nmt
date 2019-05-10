import argparse

import torch

import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
import model.optim as module_optim
from config_parser import ConfigParser
from trainer import Trainer


def main(config: ConfigParser):
    logger = config.get_logger("train")

    # setup data_loader instances
    data_loader = config.initialize("train_data_loader", module_data)
    valid_data_loader = config.initialize("val_data_loader", module_data)

    # build model architecture, then print to console
    model = config.initialize("arch", module_arch)
    logger.info(model)

    # get function handles of loss and metrics
    loss = getattr(module_loss, config["loss"])
    metrics = [getattr(module_metric, met) for met in config["metrics"]]

    # build optimizer, learning rate scheduler.
    # delete every lines containing lr_scheduler for disabling scheduler
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    # optimizer = config.initialize("optimizer", module_optim, trainable_params)
    optimizer = config.initialize("optimizer", module_optim)
    optimizer.set_parameters(trainable_params)

    trainer = Trainer(
        model,
        loss,
        metrics,
        optimizer,
        config=config,
        data_loader=data_loader,
        valid_data_loader=valid_data_loader,
    )

    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch Template")
    parser.add_argument(
        "-c",
        "--config",
        default=None,
        type=str,
        help="config file path (default: None)",
    )
    parser.add_argument(
        "-r",
        "--resume",
        default=None,
        type=str,
        help="path to latest checkpoint (default: None)",
    )
    parser.add_argument(
        "-d",
        "--device",
        default=None,
        type=str,
        help="indices of GPUs to enable (default: all)",
    )
    args = parser.parse_args()
    config = ConfigParser.parse(args)

    main(config)
