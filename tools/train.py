import argparse
import torch

from dassl.utils import setup_logger, set_random_seed, collect_env_info
from dassl.config import clean_cfg, get_cfg_default
from dassl.engine import build_trainer


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
    if cfg.SEED >= 0:
        print("Setting fixed seed: {}".format(cfg.SEED))
        set_random_seed(cfg.SEED)
    setup_logger(cfg.OUTPUT_DIR)

    if torch.cuda.is_available() and cfg.USE_CUDA:
        torch.backends.cudnn.benchmark = True

    print_args(args, cfg)
    print("Collecting env info ...")
    print("** System info **\n{}\n".format(collect_env_info()))

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
    parser.add_argument("--resume", type=str, default="",
                        help="checkpoint directory (from which the training resumes)", )
    parser.add_argument("--seed", type=int, default=42, help="only positive value enables a fixed seed")

    # DA  for office31 from amazon to webcam
    # parser.add_argument("--trainer", type=str, default="SourceOnly", help="name of trainer")
    # parser.add_argument("--source-domains", type=str, nargs="+", default=["amazon"], help="source domains for DA/DG")
    # parser.add_argument("--target-domains", type=str, nargs="+", default=["webcam"], help="target domains for DA/DG")
    # parser.add_argument("--config-file", type=str, default="../configs/trainers/da/source_only/office31.yaml",
    #                     help="path to config file")
    # parser.add_argument("--dataset-config-file", type=str, default="../configs/datasets/da/office31.yaml",
    #                     help="path to config file for dataset setup", )

    # DA  for domain_net from clipart to infograph
    # parser.add_argument("--trainer", type=str, default="SourceOnly", help="name of trainer")
    # parser.add_argument("--source-domains", type=str, nargs="+", default=["clipart"], help="source domains for DA/DG")
    # parser.add_argument("--target-domains", type=str, nargs="+", default=["infograph"], help="target domains for DA/DG")
    # parser.add_argument("--config-file", type=str, default="../configs/trainers/da/source_only/mini_domainnet.yaml",
    #                     help="path to config file")
    # parser.add_argument("--dataset-config-file", type=str, default="../configs/datasets/da/domainnet.yaml",
    #                     help="path to config file for dataset setup", )

    # DG for office-home
    # parser.add_argument("--trainer", type=str, default="Vanilla", help="name of trainer")
    # parser.add_argument("--source-domains", type=str, nargs="+", default=["art", "clipart"],
    #                     help="source domains for DA/DG")
    # parser.add_argument("--target-domains", type=str, nargs="+", default=["product"], help="target domains for DA/DG")
    # parser.add_argument("--config-file", type=str, default="../configs/trainers/dg/vanilla/office_home_dg.yaml",
    #                     help="path to config file")
    # parser.add_argument("--dataset-config-file", type=str, default="../configs/datasets/dg/office_home_dg.yaml",
    #                     help="path to config file for dataset setup", )

    # DG for ddaig算法
    parser.add_argument("--trainer", type=str, default="DDAIG", help="name of trainer")
    parser.add_argument("--source-domains", type=str, nargs="+", default=["art", "clipart"],
                        help="source domains for DA/DG")
    parser.add_argument("--target-domains", type=str, nargs="+", default=["product"], help="target domains for DA/DG")
    parser.add_argument("--config-file", type=str, default="../configs/trainers/dg/ddaig/office_home_dg.yaml",
                        help="path to config file")
    parser.add_argument("--dataset-config-file", type=str, default="../configs/datasets/dg/office_home_dg.yaml",
                        help="path to config file for dataset setup", )

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
