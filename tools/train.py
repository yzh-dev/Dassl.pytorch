import argparse
import torch

from dassl.utils import setup_logger, set_random_seed, collect_env_info
from dassl.config import clean_cfg, get_cfg_default
from dassl.engine import build_trainer
import wandb
from share import share_dict


def print_args(args, cfg):
    print("***************")
    print("** Arguments **")
    print("***************")
    optkeys = list(args.__dict__.keys())
    optkeys.sort()
    for key in optkeys:
        print("{}: {}".format(key, args.__dict__[key]))
    print("************")
    print("** Config **")
    print("************")
    print(cfg)


def reset_cfg(cfg, args):
    if args.root:
        cfg.DATASET.ROOT = args.root

    if args.output_dir:
        cfg.OUTPUT_DIR = args.output_dir

    if args.resume:
        cfg.RESUME = args.resume

    if args.seed:
        cfg.SEED = args.seed

    if args.source_domains:
        cfg.DATASET.SOURCE_DOMAINS = args.source_domains

    if args.target_domains:
        cfg.DATASET.TARGET_DOMAINS = args.target_domains

    if args.transforms:
        cfg.INPUT.TRANSFORMS = args.transforms

    if args.trainer:
        cfg.TRAINER.NAME = args.trainer

    if args.backbone:
        cfg.MODEL.BACKBONE.NAME = args.backbone

    if args.head:
        cfg.MODEL.HEAD.NAME = args.head


def extend_cfg(cfg):
    """
    Add new config variables.

    E.g.
        from yacs.config import CfgNode as CN
        cfg.TRAINER.MY_MODEL = CN()
        cfg.TRAINER.MY_MODEL.PARAM_A = 1.
        cfg.TRAINER.MY_MODEL.PARAM_B = 0.5
        cfg.TRAINER.MY_MODEL.PARAM_C = False
    """
    pass


def setup_cfg(args):
    cfg = get_cfg_default()
    extend_cfg(cfg)

    # 1. From the dataset config file
    if args.dataset_config_file:
        cfg.merge_from_file(args.dataset_config_file)

    # 2. From the method config file
    if args.config_file:
        cfg.merge_from_file(args.config_file)

    # 3. From input arguments
    reset_cfg(cfg, args)

    # 4. From optional input arguments
    cfg.merge_from_list(args.opts)

    clean_cfg(cfg, args.trainer)
    cfg.freeze()

    return cfg


def main(args):
    cfg = setup_cfg(args)
    wandb.init(
        project="dassl_OfficeHome",
        name="{}-{}-{}-".format(
            args.trainer,
            args.target_domains,
            cfg.OPTIM.LR
        ),
        config=vars(args)  # namespace to dict
    )
    share_dict['args'] = args
    share_dict['cfg'] = cfg

    if cfg.SEED >= 0:
        print("Setting fixed seed: {}".format(cfg.SEED))
        set_random_seed(cfg.SEED)
    setup_logger(cfg.OUTPUT_DIR)

    if torch.cuda.is_available() and cfg.USE_CUDA:
        torch.backends.cudnn.benchmark = True

    print_args(args, cfg)
    # print("Collecting env info ...")
    # print("** System info **\n{}\n".format(collect_env_info()))
    trainer = build_trainer(cfg)

    if args.eval_only:
        trainer.load_model(args.model_dir, epoch=args.load_epoch)
        trainer.test()
        return

    if not args.no_train:
        trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="D:\ML\Dataset", help="path to dataset")
    parser.add_argument("--output-dir", type=str, default="../output/", help="output directory")
    parser.add_argument("--resume", type=str, default="0",
                        help="checkpoint directory (from which the training resumes)", )
    parser.add_argument("--seed", type=int, default=42, help="only positive value enables a fixed seed")
    parser.add_argument('--pe_type', type=str, choices=['CI', 'CK'], help="Cross-Instance, Cross-Kernel")

    # ------------------------------DA-------------------------
    # DA  for SourceOnly
    # parser.add_argument("--trainer", type=str, default="SourceOnly", help="name of trainer")
    # parser.add_argument("--config-file", type=str, default="../configs/trainers/da/source_only/office31.yaml",
    #                     help="path to config file")
    # parser.add_argument("--dataset-config-file", type=str, default="../configs/datasets/da/office31.yaml",
    #                     help="path to config file for dataset setup", )
    # parser.add_argument("--source-domains", type=str, nargs="+", default=["amazon"], help="source domains for DA/DG")
    # parser.add_argument("--target-domains", type=str, nargs="+", default=["webcam"], help="target domains for DA/DG")

    # DA  for SourceOnly
    # parser.add_argument("--trainer", type=str, default="SourceOnly", help="name of trainer")
    # parser.add_argument("--source-domains", type=str, nargs="+", default=["clipart"], help="source domains for DA/DG")
    # parser.add_argument("--target-domains", type=str, nargs="+", default=["infograph"], help="target domains for DA/DG")
    # parser.add_argument("--config-file", type=str, default="../configs/trainers/da/source_only/mini_domainnet.yaml",
    #                     help="path to config file")
    # parser.add_argument("--dataset-config-file", type=str, default="../configs/datasets/da/domainnet.yaml",
    #                     help="path to config file for dataset setup", )

    # DA  for DANN
    # parser.add_argument("--trainer", type=str, default="DANN", help="name of trainer")
    # parser.add_argument("--source-domains", type=str, nargs="+", default=["clipart", "infograph", "painting"],
    #                     help="source domains for DA/DG")
    # parser.add_argument("--target-domains", type=str, nargs="+", default=["quickdraw"], help="target domains for DA/DG")
    # parser.add_argument("--config-file", type=str, default="../configs/trainers/da/dann/domainnet.yaml",
    #                     help="path to config file")
    # parser.add_argument("--dataset-config-file", type=str, default="../configs/datasets/da/domainnet.yaml",
    #                     help="path to config file for dataset setup", )

    # DA  for ADDA
    # parser.add_argument("--trainer", type=str, default="ADDA", help="name of trainer")
    # parser.add_argument("--source-domains", type=str, nargs="+", default=["clipart", "infograph", "painting"],
    #                     help="source domains for DA/DG")
    # parser.add_argument("--target-domains", type=str, nargs="+", default=["quickdraw"], help="target domains for DA/DG")
    # parser.add_argument("--config-file", type=str, default="../configs/trainers/da/adda/domainnet.yaml",
    #                     help="path to config file")
    # parser.add_argument("--dataset-config-file", type=str, default="../configs/datasets/da/domainnet.yaml",
    #                     help="path to config file for dataset setup", )

    # DA  for SE
    # parser.add_argument("--trainer", type=str, default="SE", help="name of trainer")
    # parser.add_argument("--source-domains", type=str, nargs="+", default=["clipart", "infograph", "painting"],
    #                     help="source domains for DA/DG")
    # parser.add_argument("--target-domains", type=str, nargs="+", default=["quickdraw"], help="target domains for DA/DG")
    # parser.add_argument("--config-file", type=str, default="../configs/trainers/da/se/domainnet.yaml",
    #                     help="path to config file")
    # parser.add_argument("--dataset-config-file", type=str, default="../configs/datasets/da/domainnet.yaml",
    #                     help="path to config file for dataset setup", )

    # DA  for MME
    # parser.add_argument("--trainer", type=str, default="MME", help="name of trainer")
    # parser.add_argument("--source-domains", type=str, nargs="+", default=["clipart", "infograph", "painting"],
    #                     help="source domains for DA/DG")
    # parser.add_argument("--target-domains", type=str, nargs="+", default=["quickdraw"], help="target domains for DA/DG")
    # parser.add_argument("--config-file", type=str, default="../configs/trainers/da/mme/domainnet.yaml",
    #                     help="path to config file")
    # parser.add_argument("--dataset-config-file", type=str, default="../configs/datasets/da/domainnet.yaml",
    #                     help="path to config file for dataset setup", )

    # DA  for CDAC
    # parser.add_argument("--trainer", type=str, default="CDAC", help="name of trainer")
    # parser.add_argument("--source-domains", type=str, nargs="+", default=["clipart","infograph","painting"], help="source domains for DA/DG")
    # parser.add_argument("--target-domains", type=str, nargs="+", default=["quickdraw"], help="target domains for DA/DG")
    # parser.add_argument("--config-file", type=str, default="../configs/trainers/da/cdac/domainnet.yaml",
    #                     help="path to config file")
    # parser.add_argument("--dataset-config-file", type=str, default="../configs/datasets/da/domainnet.yaml",
    #                     help="path to config file for dataset setup", )

    # DA  for M3SDA
    # parser.add_argument("--trainer", type=str, default="M3SDA", help="name of trainer")
    # parser.add_argument("--source-domains", type=str, nargs="+", default=["clipart","infograph","painting"], help="source domains for DA/DG")
    # parser.add_argument("--target-domains", type=str, nargs="+", default=["quickdraw"], help="target domains for DA/DG")
    # parser.add_argument("--config-file", type=str, default="../configs/trainers/da/m3sda/domainnet.yaml",
    #                     help="path to config file")
    # parser.add_argument("--dataset-config-file", type=str, default="../configs/datasets/da/domainnet.yaml",
    #                     help="path to config file for dataset setup", )

    # DA  for DAEL：use both labeled and unlabled data
    # parser.add_argument("--trainer", type=str, default="DAEL", help="name of trainer")
    # parser.add_argument("--config-file", type=str, default="../configs/trainers/da/dael/domainnet.yaml",
    #                     help="path to config file")
    # parser.add_argument("--dataset-config-file", type=str, default="../configs/datasets/da/domainnet.yaml",
    #                     help="path to config file for dataset setup", )
    # parser.add_argument("--source-domains", type=str, nargs="+", default=["clipart","infograph","painting"], help="source domains for DA/DG")
    # parser.add_argument("--target-domains", type=str, nargs="+", default=["quickdraw"], help="target domains for DA/DG")

    # ------------------------------DG-------------------------
    # DG for Vanilla
    # parser.add_argument("--trainer", type=str, default="Vanilla", help="name of trainer")
    # parser.add_argument("--source-domains", type=str, nargs="+", default=["art", "clipart"])
    # parser.add_argument("--target-domains", type=str, nargs="+", default=["product"])
    # parser.add_argument("--config-file", type=str, default="../configs/trainers/dg/vanilla/office_home_dg.yaml")
    # parser.add_argument("--dataset-config-file", type=str, default="../configs/datasets/dg/office_home_dg.yaml")

    # DG for DAEL算法：only use labeled source data.
    # parser.add_argument("--trainer", type=str, default="DAELDG", help="name of trainer")
    # parser.add_argument("--config-file", type=str, default="../configs/trainers/dg/daeldg/office_home_dg.yaml",
    #                     help="path to config file")
    # parser.add_argument("--dataset-config-file", type=str, default="../configs/datasets/dg/office_home_dg.yaml",
    # help="path to config file for dataset setup", )

    # DG for ddaig算法
    # parser.add_argument("--trainer", type=str, default="DAELDG", help="name of trainer")
    # parser.add_argument("--config-file", type=str, default="../configs/trainers/dg/daeldg/office_home_dg.yaml")
    # parser.add_argument("--dataset-config-file", type=str, default="../configs/datasets/dg/office_home_dg.yaml")
    # parser.add_argument("--source-domains", type=str, nargs="+", default=["art", "clipart", "real_world"])
    # parser.add_argument("--target-domains", type=str, nargs="+", default=["product"])

    # DG for DomainMix算法
    parser.add_argument("--trainer", type=str, default="DomainMix", help="name of trainer")
    parser.add_argument("--config-file", type=str, default="../configs/trainers/dg/ddg/OfficeHome.yaml")
    parser.add_argument("--dataset-config-file", type=str, default="../configs/datasets/dg/ddg_OfficeHome_resnet18.yaml")
    parser.add_argument("--source-domains", type=str, nargs="+", default=["art", "clipart", "real_world"])
    parser.add_argument("--target-domains", type=str, nargs="+", default=["product"])


    # DG for CrossGrad算法
    # parser.add_argument("--trainer", type=str, default="CrossGrad", help="name of trainer")
    # parser.add_argument("--config-file", type=str, default="../configs/trainers/dg/vanilla/office_home_dg.yaml")
    # parser.add_argument("--dataset-config-file", type=str, default="../configs/datasets/dg/office_home_dg.yaml")
    # parser.add_argument("--source-domains", type=str, nargs="+", default=["art", "clipart", "real_world"])
    # parser.add_argument("--target-domains", type=str, nargs="+", default=["product"])

    # ------------------------------SSL-------------------------
    # # SSL for EntMin
    # parser.add_argument("--trainer", type=str, default="EntMin", help="name of trainer")
    # parser.add_argument("--config-file", type=str, default="../configs/trainers/ssl/fixmatch/cifar10.yaml",
    #                     help="path to config file")
    # parser.add_argument("--dataset-config-file", type=str, default="../configs/datasets/ssl/cifar10.yaml",
    #                     help="path to config file for dataset setup", )

    # SSL for MeanTeacher
    # parser.add_argument("--trainer", type=str, default="MeanTeacher", help="name of trainer")
    # parser.add_argument("--config-file", type=str, default="../configs/trainers/ssl/fixmatch/cifar10.yaml",
    #                     help="path to config file")
    # parser.add_argument("--dataset-config-file", type=str, default="../configs/datasets/ssl/cifar10.yaml",
    #                     help="path to config file for dataset setup",)

    # SSL for MixMatch
    # parser.add_argument("--trainer", type=str, default="MixMatch", help="name of trainer")
    # parser.add_argument("--config-file", type=str, default="../configs/trainers/ssl/mixmatch/cifar10.yaml",
    #                     help="path to config file")
    # parser.add_argument("--dataset-config-file", type=str, default="../configs/datasets/ssl/cifar10.yaml",
    #                     help="path to config file for dataset setup",)
    # parser.add_argument("--source-domains", type=str, nargs="+", default=["art", "clipart", "real_world"],
    #                     help="SSL任务实际上不需要source-domains和target-domains属性")
    # parser.add_argument("--target-domains", type=str, nargs="+", default=["product"], help="target domains for DA/DG")

    # SSL for FixMatch
    # parser.add_argument("--trainer", type=str, default="FixMatch", help="name of trainer")
    # parser.add_argument("--config-file", type=str, default="../configs/trainers/ssl/fixmatch/cifar10.yaml",
    #                     help="path to config file")
    # parser.add_argument("--dataset-config-file", type=str, default="../configs/datasets/ssl/cifar10.yaml",
    #                     help="path to config file for dataset setup", )
    # parser.add_argument("--source-domains", type=str, nargs="+", default=["art", "clipart", "real_world"],
    #                     help="SSL任务实际上不需要source-domains和target-domains属性")
    # parser.add_argument("--target-domains", type=str, nargs="+", default=["product"], help="target domains for DA/DG")

    parser.add_argument("--backbone", type=str, default="", help="name of CNN backbone")
    parser.add_argument("--transforms", type=str, nargs="+", help="data augmentation methods")
    parser.add_argument("--head", type=str, default="", help="name of head")
    parser.add_argument("--eval-only", action="store_true", help="evaluation only")
    parser.add_argument(
        "--model-dir",
        type=str,
        default="",
        help="load model from this directory for eval-only mode",
    )
    parser.add_argument(
        "--load-epoch",
        type=int,
        help="load model weights at this epoch for evaluation"
    )
    parser.add_argument(
        "--no-train", action="store_true", help="do not call trainer.train()"
    )
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="modify config options using the command-line",
    )
    args = parser.parse_args()
    main(args)
