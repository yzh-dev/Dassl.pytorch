import torch
from torch.nn import functional as F

from dassl.engine import TRAINER_REGISTRY, TrainerXU
from dassl.modeling.ops import mixup
from dassl.modeling.ops.utils import (
    sharpen_prob, create_onehot, linear_rampup, shuffle_index
)


@TRAINER_REGISTRY.register()
class MixMatch(TrainerXU):
    """MixMatch: A Holistic Approach to Semi-Supervised Learning.
    # 可参考解读 https://blog.csdn.net/qq_41380292/article/details/119277938?spm=1001.2014.3001.5501
    https://arxiv.org/abs/1905.02249.
    """

    def __init__(self, cfg):
        super().__init__(cfg)
        self.weight_u = cfg.TRAINER.MIXMATCH.WEIGHT_U
        self.temp = cfg.TRAINER.MIXMATCH.TEMP
        self.beta = cfg.TRAINER.MIXMATCH.MIXUP_BETA
        self.rampup = cfg.TRAINER.MIXMATCH.RAMPUP

    def check_cfg(self, cfg):
        assert cfg.DATALOADER.K_TRANSFORMS > 1

    def forward_backward(self, batch_x, batch_u):
        input_x, label_x, input_u = self.parse_batch_train(batch_x, batch_u)
        num_x = input_x.shape[0]

        global_step = self.batch_idx + self.epoch * self.num_batches
        weight_u = self.weight_u * linear_rampup(global_step, self.rampup)

        # Generate pseudo-label for unlabeled data
        with torch.no_grad():
            output_u = 0
            for input_ui in input_u:
                output_ui = F.softmax(self.model(input_ui), 1)   # Batch*Num_classes,沿着feat维度上计算，获取预测prob
                output_u += output_ui  # K种变换下的预测进行累加
            output_u /= len(input_u)  # 计算K种变换下的平均预测
            label_u = sharpen_prob(output_u, self.temp)  # 将预测值进行sharpen
            label_u = [label_u] * len(input_u)
            label_u = torch.cat(label_u, 0)  # 拼接K种变换下预测
            input_u = torch.cat(input_u, 0)  # 拼接K种变换下label

        # Combine and shuffle labeled and unlabeled data
        input_xu = torch.cat([input_x, input_u], 0)
        label_xu = torch.cat([label_x, label_u], 0)
        input_xu, label_xu = shuffle_index(input_xu, label_xu)

        # Mixup: labeled and all data
        input_x, label_x = mixup(
            input_x,
            input_xu[:num_x],
            label_x,
            label_xu[:num_x],
            self.beta,
            preserve_order=True,
        )
        # unlabeled and all data
        input_u, label_u = mixup(
            input_u,
            input_xu[num_x:],
            label_u,
            label_xu[num_x:],
            self.beta,
            preserve_order=True,
        )

        # Compute losses
        output_x = F.softmax(self.model(input_x), 1)
        loss_x = (-label_x * torch.log(output_x + 1e-5)).sum(1).mean()

        output_u = F.softmax(self.model(input_u), 1)
        loss_u = ((label_u - output_u)**2).mean()

        loss = loss_x + loss_u*weight_u
        self.model_backward_and_update(loss)

        loss_summary = {"loss_x": loss_x.item(), "loss_u": loss_u.item()}

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def parse_batch_train(self, batch_x, batch_u):
        input_x = batch_x["img"][0]
        label_x = batch_x["label"]
        label_x = create_onehot(label_x, self.num_classes)
        input_u = batch_u["img"]

        input_x = input_x.to(self.device)
        label_x = label_x.to(self.device)
        input_u = [input_ui.to(self.device) for input_ui in input_u]

        return input_x, label_x, input_u
