import torch

from dassl.utils import check_isfile
from dassl.engine import TRAINER_REGISTRY, TrainerXU


@TRAINER_REGISTRY.register()
class AdaBN(TrainerXU):
    """_17 ICLR adabn Revisiting Batch Normalization For Practical Domain Adaptation.pdf
    https://arxiv.org/abs/1603.04779.
    """

    def __init__(self, cfg):
        super().__init__(cfg)
        self.done_reset_bn_stats = False

    def check_cfg(self, cfg):
        assert check_isfile(
            cfg.MODEL.INIT_WEIGHTS
        ), "The weights of source model must be provided"

    def before_epoch(self):
        if not self.done_reset_bn_stats:
            for m in self.model.modules():
                classname = m.__class__.__name__
                if classname.find("BatchNorm") != -1:
                    m.reset_running_stats()

            self.done_reset_bn_stats = True

    def forward_backward(self, batch_x, batch_u):
        input_u = batch_u["img"].to(self.device)
        # 只利用了无标签样本？未更新模型参数？
        with torch.no_grad():
            self.model(input_u)

        return None
