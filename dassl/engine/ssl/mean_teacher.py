import copy
from torch.nn import functional as F

from dassl.engine import TRAINER_REGISTRY, TrainerXU
from dassl.metrics import compute_accuracy
from dassl.modeling.ops.utils import sigmoid_rampup, ema_model_update


@TRAINER_REGISTRY.register()
class MeanTeacher(TrainerXU):
    """Mean teacher.

    https://arxiv.org/abs/1703.01780.
    """

    def __init__(self, cfg):
        super().__init__(cfg)
        self.weight_u = cfg.TRAINER.MEANTEACHER.WEIGHT_U
        self.ema_alpha = cfg.TRAINER.MEANTEACHER.EMA_ALPHA
        self.rampup = cfg.TRAINER.MEANTEACHER.RAMPUP

        self.teacher = copy.deepcopy(self.model)  # teacher网络设定为冻结
        self.teacher.train()
        for param in self.teacher.parameters():
            param.requires_grad_(False)

    def forward_backward(self, batch_x, batch_u):
        input_x, label_x, input_u = self.parse_batch_train(batch_x, batch_u)
        # 有标签样本损失
        logit_x = self.model(input_x)
        loss_x = F.cross_entropy(logit_x, label_x)

        # 无标签样本
        target_u = F.softmax(self.teacher(input_u), 1)  # teacher net的预测作为gt
        prob_u = F.softmax(self.model(input_u), 1)  # student net的预测
        loss_u = ((prob_u - target_u)**2).sum(1).mean()  # 取L2作为损失

        weight_u = self.weight_u * sigmoid_rampup(self.epoch, self.rampup)
        loss = loss_x + loss_u*weight_u
        self.model_backward_and_update(loss)

        global_step = self.batch_idx + self.epoch * self.num_batches
        ema_alpha = min(1 - 1 / (global_step+1), self.ema_alpha)
        ema_model_update(self.model, self.teacher, ema_alpha)  # 指数平滑更新stu net

        loss_summary = {
            "loss_x": loss_x.item(),
            "acc_x": compute_accuracy(logit_x, label_x)[0].item(),
            "loss_u": loss_u.item(),
        }

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary
