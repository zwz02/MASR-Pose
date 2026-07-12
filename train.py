import os
import sys
import argparse
import logging
import random

import torch
import gorilla

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, 'provider'))
sys.path.append(os.path.join(BASE_DIR, 'model'))
sys.path.append(os.path.join(BASE_DIR, 'model', 'pointnet2'))
sys.path.append(os.path.join(BASE_DIR, 'utils'))

from create_dataloaders import create_dataloaders
from solver import Solver, get_logger
from Net import Net, Loss


def get_parser():
    parser = argparse.ArgumentParser(
        description="Pose Estimation")

    # pretrain
    parser.add_argument("--gpus",
                        type=str,
                        default="1",
                        help="gpu num")

    parser.add_argument("--config",
                        type=str,
                        default="config/REAL/camera_real.yaml",
                        help="path to config file")

    parser.add_argument("--checkpoint_epoch",
                        type=int,
                        default=-1,  # 中断时已完成的epoch数
                        help="checkpoint epoch")

    args_cfg = parser.parse_args()

    return args_cfg


def init():
    args = get_parser()
    exp_name = args.config.split("/")[-1].split(".")[0]
    log_dir = os.path.join("log_SRM_geoablation", exp_name)

    if not os.path.isdir("log_SRM_geoablation"):
        os.makedirs("log_SRM_geoablation")
    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)

    cfg = gorilla.Config.fromfile(args.config)
    cfg.exp_name = exp_name
    cfg.log_dir = log_dir
    cfg.ckpt_dir = os.path.join(log_dir, 'ckpt')

    cfg.checkpoint_epoch = args.checkpoint_epoch

    if not os.path.isdir(cfg.ckpt_dir):
        os.makedirs(cfg.ckpt_dir)

    cfg.gpus = args.gpus
    logger = get_logger(
        level_print=logging.INFO, level_save=logging.WARNING, path_file=log_dir + "/training_logger.log")
    gorilla.utils.set_cuda_visible_devices(gpu_ids=cfg.gpus)

    return logger, cfg


if __name__ == "__main__":

    torch.multiprocessing.set_start_method('spawn', force=True)

    logger, cfg = init()

    logger.warning(
        "************************ Start Logging ************************")
    logger.info(cfg)
    logger.info("using gpu: {}".format(cfg.gpus))

    random.seed(cfg.rd_seed)
    torch.manual_seed(cfg.rd_seed)
    torch.cuda.manual_seed(cfg.rd_seed)
    torch.cuda.manual_seed_all(cfg.rd_seed)

    # model
    logger.info("=> creating model ...")
    model = Net(cfg.pose_net)

    # model
    logger.info("=> creating model ...")
    model = Net(cfg.pose_net)

    if cfg.checkpoint_epoch != -1:
        logger.info("=> loading checkpoint from epoch {} ...".format(cfg.checkpoint_epoch))
        checkpoint = os.path.join(cfg.ckpt_dir, 'epoch_' + str(cfg.checkpoint_epoch) + '.pt')
        checkpoint_file = gorilla.solver.load_checkpoint(model=model, filename=checkpoint)
        start_epoch = 9  # 已完成的epoch+1
        start_iter = 32000  # 已完成的epoch*4000
        del checkpoint_file
    else:
        start_epoch = 1
        start_iter = 0

    model = model.cuda()

    count_parameters = sum(gorilla.parameter_count(model).values())
    logger.warning("#Total parameters : {}".format(count_parameters))
    loss = Loss(cfg.loss).cuda()

    # dataloader
    dataloaders = create_dataloaders(cfg.train_dataset)

    for k in dataloaders.keys():
        dataloaders[k].dataset.reset()

    # solver
    Trainer = Solver(model=model,
                     loss=loss,
                     dataloaders=dataloaders,
                     logger=logger,
                     cfg=cfg,
                     start_epoch=start_epoch,
                     start_iter=start_iter)
    Trainer.solve()

    logger.info('\nFinish!\n')
