from datetime import datetime

import torch
from torch.nn import functional as F

from dassl.optim import build_optimizer, build_lr_scheduler
from dassl.utils import count_num_param
from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.modeling import build_network
from dassl.engine.trainer import SimpleNet
from PIL import Image
import numpy as np


@TRAINER_REGISTRY.register()
class DDAIG(TrainerX):
    """Deep Domain-Adversarial Image Generation.
    # _20 AAAI Deep Domain-Adversarial Image Generation for Domain Generalisation.pdf
    https://arxiv.org/abs/2003.06054.
    """

    def __init__(self, cfg):
        super().__init__(cfg)
        self.lmda = cfg.TRAINER.DDAIG.LMDA
        self.clamp = cfg.TRAINER.DDAIG.CLAMP
        self.clamp_min = cfg.TRAINER.DDAIG.CLAMP_MIN
        self.clamp_max = cfg.TRAINER.DDAIG.CLAMP_MAX
        self.warmup = cfg.TRAINER.DDAIG.WARMUP
        self.alpha = cfg.TRAINER.DDAIG.ALPHA

    def build_model(self):
        cfg = self.cfg
        # 类别分类器，不与D共享backbone
        print("Building F")
        self.F = SimpleNet(cfg, cfg.MODEL, self.num_classes)  # 分别构建backbone
        self.F.to(self.device)
        print("# params: {:,}".format(count_num_param(self.F)))
        self.optim_F = build_optimizer(self.F, cfg.OPTIM)
        self.sched_F = build_lr_scheduler(self.optim_F, cfg.OPTIM)
        self.register_model("F", self.F, self.optim_F, self.sched_F)
        # 域分类器
        print("Building D")
        self.D = SimpleNet(cfg, cfg.MODEL, self.num_source_domains)
        self.D.to(self.device)
        print("# params: {:,}".format(count_num_param(self.D)))
        self.optim_D = build_optimizer(self.D, cfg.OPTIM)
        self.sched_D = build_lr_scheduler(self.optim_D, cfg.OPTIM)
        self.register_model("D", self.D, self.optim_D, self.sched_D)

        print("Building G")  # 生成器
        self.G = build_network(cfg.TRAINER.DDAIG.G_ARCH, verbose=cfg.VERBOSE)
        self.G.to(self.device)
        print("# params: {:,}".format(count_num_param(self.G)))
        self.optim_G = build_optimizer(self.G, cfg.OPTIM)
        self.sched_G = build_lr_scheduler(self.optim_G, cfg.OPTIM)
        self.register_model("G", self.G, self.optim_G, self.sched_G)

    def saveImg(self, origin_imgs, generated_imgs):
        # 由于代码中的中间结果是带有梯度的要进行detach()操作
        mean = [0.485, 0.456, 0.406]  # dataLoader中设置的mean参数
        std = [0.229, 0.224, 0.225]  # dataLoader中设置的std参数
        origin_imgs_out = origin_imgs.clone()
        generated_imgs_out = generated_imgs.clone()

        for i in range(len(mean)):  # 反标准化
            origin_imgs_out[:, i, :, :] = origin_imgs[:, i, :, :] * std[i] + mean[i]
            generated_imgs_out[:, i, :, :] = generated_imgs[:, i, :, :] * std[i] + mean[i]

        origin_imgs_out = origin_imgs_out.permute(0, 2, 3, 1)  # BHWC
        generated_imgs_out = generated_imgs_out.permute(0, 2, 3, 1)  # BHWC
        origin_imgs_out = origin_imgs_out.cpu().detach().numpy()  # to numpy array
        generated_imgs_out = generated_imgs_out.cpu().detach().numpy()  # to numpy array
        for i in range(origin_imgs_out.shape[0]):
            origin_out = origin_imgs_out[i] * 255  # 得到batch中其中一步的图片,从[0,1]转为[0,255]
            origin_image = Image.fromarray(np.uint8(origin_out)).convert('RGB')
            # image.show()
            # 通过时间命名存储结果
            timestamp = datetime.now().strftime("%H-%M-%S") + "-{}".format(i)
            savepath = timestamp + '_origin.jpg'
            root = "D:/GeneratedImg/"
            origin_image.save(root + savepath)

            generated_out = generated_imgs_out[i] * 255  # 得到batch中其中一步的图片,从[0,1]转为[0,255]
            generated_image = Image.fromarray(np.uint8(generated_out)).convert('RGB')
            # image.show()
            # 通过时间命名存储结果
            timestamp = datetime.now().strftime("%H-%M-%S") + "-{}".format(i)
            savepath = timestamp + '_generated.jpg'
            root = "D:/GeneratedImg/"
            generated_image.save(root + savepath)


    def forward_backward(self, batch):
        input, label, domain = self.parse_batch_train(batch)
        # correctly classified by the label classifier while fooling the domain classifier.
        #############
        # Update G，特征变换器G（反复使用3*3卷积层，保持形状不变），负责对输入input进行特征增强
        #############
        input_p = self.G(input, lmda=self.lmda)
        # self.saveImg(input, input_p)

        if self.clamp:  # 将输入张量的每个元素的范围限制到指定的区间
            input_p = torch.clamp(input_p, min=self.clamp_min, max=self.clamp_max)
        # Minimize label loss
        label_loss = F.cross_entropy(self.F(input_p), label)  # recognised by label clf
        # Maximize domain loss，分不清域时，loss变大
        domain_loss = F.cross_entropy(self.D(input_p), domain)  # fool the domain clf
        # print("label_loss: {:,}, domain_loss: {:,}, ".format(label_loss, domain_loss))
        loss_g = label_loss - domain_loss
        self.model_backward_and_update(loss_g, "G")  # 训练特征增强网络G

        # Perturb data with new G
        with torch.no_grad():
            input_p = self.G(input, lmda=self.lmda)  # 计算特征增强后的input_p
            if self.clamp:
                input_p = torch.clamp(input_p, min=self.clamp_min, max=self.clamp_max)

        #############
        # Update F，类别分类器，tips: do not share backbone with D net
        #############
        loss_f = F.cross_entropy(self.F(input), label)  # train F net (can extract and clf feat)
        if (self.epoch + 1) > self.warmup:
            loss_fp = F.cross_entropy(self.F(input_p), label)
            loss_f = (1.0 - self.alpha) * loss_f + self.alpha * loss_fp  # 综合原始图片input和特征增强input_p的损失，作为分类损失
        self.model_backward_and_update(loss_f, "F")  # use x and x_g to train D

        #############
        # Update D，域分类器，# Minimize domain loss，分得清各个域，对抗性训练，目的是给G生成不同风格的新图片
        #############
        loss_d = F.cross_entropy(self.D(input), domain)  # use origin x to train D net(can extract domain related feat and the clf)
        self.model_backward_and_update(loss_d, "D")

        loss_summary = {
            "loss_g_label": label_loss.item(),
            "loss_g_domain": domain_loss.item(),
            # "loss_g": loss_g.item(),
            "loss_f": loss_f.item(),
            "loss_d": loss_d.item(),
        }

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def model_inference(self, input):
        return self.F(input)
