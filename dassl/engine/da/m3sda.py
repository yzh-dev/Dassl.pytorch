import torch
import torch.nn as nn
from torch.nn import functional as F

from dassl.optim import build_optimizer, build_lr_scheduler
from dassl.utils import count_num_param
from dassl.engine import TRAINER_REGISTRY, TrainerXU
from dassl.engine.trainer import SimpleNet


class PairClassifiers(nn.Module):

    def __init__(self, fdim, num_classes):
        super().__init__()
        self.c1 = nn.Linear(fdim, num_classes)  # 不同的分类头，初始化状态不同
        self.c2 = nn.Linear(fdim, num_classes)

    def forward(self, x):
        z1 = self.c1(x)
        if not self.training:
            return z1
        z2 = self.c2(x)
        return z1, z2


@TRAINER_REGISTRY.register()
class M3SDA(TrainerXU):
    """Moment Matching for Multi-Source Domain Adaptation.

    https://arxiv.org/abs/1812.01754.
    """

    def __init__(self, cfg):
        super().__init__(cfg)
        n_domain = cfg.DATALOADER.TRAIN_X.N_DOMAIN
        batch_size = cfg.DATALOADER.TRAIN_X.BATCH_SIZE
        if n_domain <= 0:
            n_domain = self.num_source_domains
        self.split_batch = batch_size // n_domain
        self.n_domain = n_domain

        self.n_step_F = cfg.TRAINER.M3SDA.N_STEP_F
        self.lmda = cfg.TRAINER.M3SDA.LMDA

    def check_cfg(self, cfg):
        assert cfg.DATALOADER.TRAIN_X.SAMPLER == "RandomDomainSampler"
        assert not cfg.DATALOADER.TRAIN_U.SAME_AS_X

    def build_model(self):
        cfg = self.cfg

        print("Building F")  # feat extract net
        self.F = SimpleNet(cfg, cfg.MODEL, 0)
        self.F.to(self.device)
        print("# params: {:,}".format(count_num_param(self.F)))
        self.optim_F = build_optimizer(self.F, cfg.OPTIM)
        self.sched_F = build_lr_scheduler(self.optim_F, cfg.OPTIM)
        self.register_model("F", self.F, self.optim_F, self.sched_F)
        fdim = self.F.fdim

        print("Building C")  # 每个域构建一个成对的label clf net
        self.C = nn.ModuleList(
            [
                PairClassifiers(fdim, self.num_classes)
                for _ in range(self.num_source_domains)
            ]
        )
        self.C.to(self.device)
        print("# params: {:,}".format(count_num_param(self.C)))
        self.optim_C = build_optimizer(self.C, cfg.OPTIM)
        self.sched_C = build_lr_scheduler(self.optim_C, cfg.OPTIM)
        self.register_model("C", self.C, self.optim_C, self.sched_C)

    def forward_backward(self, batch_x, batch_u):
        parsed = self.parse_batch_train(batch_x, batch_u)
        input_x, label_x, domain_x, input_u = parsed
        # 按照domain组织数据
        input_x = torch.split(input_x, self.split_batch, 0)
        label_x = torch.split(label_x, self.split_batch, 0)
        domain_x = torch.split(domain_x, self.split_batch, 0)
        domain_x = [d[0].item() for d in domain_x]

        # Step A，同时训练self.F和self.C
        loss_x = 0
        feat_x = []

        for x, y, d in zip(input_x, label_x, domain_x):
            f = self.F(x)
            z1, z2 = self.C[d](f)  # 对应domain的分类器预测结果
            loss_x += F.cross_entropy(z1, y) + F.cross_entropy(z2, y)  # labeled loss
            feat_x.append(f)  # labeled feat
        loss_x /= self.n_domain  # supervised loss

        feat_u = self.F(input_u)  # unlabeled feat
        loss_msda = self.moment_distance(feat_x, feat_u)  # 计算labeled feat与unlabeled feat之间的均值、方差损失

        loss_step_A = loss_x + loss_msda * self.lmda
        self.model_backward_and_update(loss_step_A)

        # Step B，只train分类头self.C（同时利用有标签和无标签样本）
        # The goal is to make the discrepancy of each pair of classifiers as large as possible on the target domain.
        # For example, the outputs of  two classifiers self.C should possess a large discrepancy
        with torch.no_grad():  # 在该模块下，所有计算得出的tensor的requires_grad都自动设置为False
            feat_u = self.F(input_u)

        loss_x, loss_dis = 0, 0

        for x, y, d in zip(input_x, label_x, domain_x):
            with torch.no_grad():
                f = self.F(x)
            z1, z2 = self.C[d](f)
            loss_x += F.cross_entropy(z1, y) + F.cross_entropy(z2, y)  # 实验测试表明，需要在这里加上有标签样本的损失，否则性能下降

            z1, z2 = self.C[d](feat_u)
            p1 = F.softmax(z1, 1)
            p2 = F.softmax(z2, 1)
            #  define the discrepancy of two classifiers as the L1-distance between the outputs of the two classifiers
            loss_dis += self.discrepancy(p1, p2)  # 无标签样本中分类头预测结果一致性损失

        loss_x /= self.n_domain
        loss_dis /= self.n_domain

        loss_step_B = loss_x - loss_dis
        self.model_backward_and_update(loss_step_B, "C")

        # Step C，只train分类头self.F（只利用了无标签样本）
        # Finally, we train self.F to minimize the discrepancy of each classifier pair on the target domain.
        for _ in range(self.n_step_F):
            feat_u = self.F(input_u)

            loss_dis = 0

            for d in domain_x:
                z1, z2 = self.C[d](feat_u)
                p1 = F.softmax(z1, 1)
                p2 = F.softmax(z2, 1)
                loss_dis += self.discrepancy(p1, p2)

            loss_dis /= self.n_domain
            loss_step_C = loss_dis

            self.model_backward_and_update(loss_step_C, "F")

        loss_summary = {
            "loss_step_A": loss_step_A.item(),
            "loss_step_B": loss_step_B.item(),
            "loss_step_C": loss_step_C.item(),
        }

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def moment_distance(self, x, u):
        # x (list): a list of feature matrix.
        # u (torch.Tensor): feature matrix.
        x_mean = [xi.mean(0) for xi in x]
        u_mean = u.mean(0)
        dist1 = self.pairwise_distance(x_mean, u_mean)  # 均值差异

        x_var = [xi.var(0) for xi in x]
        u_var = u.var(0)
        dist2 = self.pairwise_distance(x_var, u_var)  # 方差差异

        return (dist1+dist2) / 2

    def pairwise_distance(self, x, u):
        # x (list): a list of feature vector.
        # u (torch.Tensor): feature vector.
        dist = 0
        count = 0

        for xi in x:
            dist += self.euclidean(xi, u)  # 有标签样本和无标签样本之间距离
            count += 1

        for i in range(len(x) - 1):  # 有标签样本之间距离
            for j in range(i + 1, len(x)):
                dist += self.euclidean(x[i], x[j])
                count += 1

        return dist / count

    def euclidean(self, input1, input2):
        return ((input1 - input2)**2).sum().sqrt()

    def discrepancy(self, y1, y2):
        return (y1 - y2).abs().mean()

    def parse_batch_train(self, batch_x, batch_u):
        input_x = batch_x["img"]
        label_x = batch_x["label"]
        domain_x = batch_x["domain"]
        input_u = batch_u["img"]

        input_x = input_x.to(self.device)
        label_x = label_x.to(self.device)
        input_u = input_u.to(self.device)

        return input_x, label_x, domain_x, input_u

    def model_inference(self, input):
        f = self.F(input)
        p = 0
        for C_i in self.C:
            z = C_i(f)
            p += F.softmax(z, 1)
        p = p / len(self.C)
        return p
