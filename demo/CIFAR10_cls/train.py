#!/usr/bin/env python
# _*_ coding: utf-8 _*_
# @Time : 2021/6/5 下午4:24
# @Author : PH
# @Version：V 0.1
# @File : train.py
# @desc :
import time
import torch
import torch.nn as nn
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from torchvision.models.resnet import resnet101
from torchvision.transforms import Compose, ToTensor, RandomHorizontalFlip, RandomRotation
from torch.utils.tensorboard import SummaryWriter
from template import TemplateModel
import argparse


def get_arg():
    parser = argparse.ArgumentParser(description="train model")
    # train config
    parser.add_argument("--epochs", type=int, default=75, help="训练的轮次")
    parser.add_argument("--batch_size", type=int, default=200, help="batch_size")
    parser.add_argument("--init_lr", type=float, default=1e-3, help="初始学习率")
    arg = parser.parse_args()
    return arg


class ModelT(TemplateModel):

    def __init__(self, model_list, optimizer_list, loss_fn, train_loader, test_loader):
        super(ModelT, self).__init__()
        # 必须设定
        # 模型架构
        # 将模型和优化器以list保存，方便对整个模型的多个部分设定对应的优化器
        self.model_list = model_list  # 模型的list
        self.optimizer_list = optimizer_list  # 优化器的list
        self.criterion = loss_fn

        # 数据集
        self.train_loader = train_loader
        self.eval_loader = test_loader

        # 下面的可以不设定
        # 训练时print的间隔
        self.log_per_step = 25  # 推荐按数据集大小设定
        self.lr_scheduler_type = "annealing"  # 默认None，即不会使用lr_scheduler
        self.warm_up_epoch = 5
        self.warm_up_mode = 'exponential'

    def loss_per_batch(self, batch):
        """
        计算数据集的一个batch的loss，这个部分是可能要按需求修改的部分
        Pytorch 中的loss函数中一般规定x和y都是float，而有些loss函数规定y要为long（比如经常用到的CrossEntropyLoss）
        如果官网：https://pytorch.org/docs/stable/nn.html#loss-functions 对y的数据类型有要求请做相应的修改
        这里除了CrossEntropyLoss将y的数据类型设为long外， 其他都默认x和y的数据类型为float
        """
        x, y = batch
        x = x.to(self.device, dtype=torch.float)

        # 标签y的数据类型
        y_dtype = torch.float
        if isinstance(self.criterion, torch.nn.CrossEntropyLoss):
            y_dtype = torch.long

        # 保证标签y至少是个列向量，即shape "B, 1"
        if y.dim == 1:
            y = y.unsqueeze(dim=1).to(self.device, dtype=y_dtype)
        else:
            y = y.to(self.device, dtype=y_dtype)

        # 若模型的输入不是一个tensor，按需求改
        pred = x
        for model in self.model_list:
            pred = model(pred)
        loss = self.criterion(pred, y)
        loss_dict = {'tol_loss': loss}
        return loss_dict

    def eval_scores_per_batch(self, batch):
        x, y = batch
        x = x.to(self.device)
        y = y.to(self.device)

        pred = x
        for model in self.model_list:
            pred = model(pred)
        scores_pre_batch = self.metric(pred, y)
        return scores_pre_batch

    def metric(self, pred, y):
        """
        不同任务的性能指标太难统一了，这里只是实现了多分类任务求准确率的方法。其他任务请按需求继承
        这个类的时候再重载这个metric函数，注意返回数据类型为字典,且一定要有self.key_metric这个
        指标，因为self.key_metric用于保存训练过程中的最优模型.这个模板使用分批计算metric再求全
        部批次的平均值的策略得到整体的metric。不会将全部的预测和ground truth保存在preds和ys中
        然后在cpu上进行预测。因为如果测试集或验证集太大（>50000）可能CPU内存装不下，训练会报错.但
        是有的metric可能不能使用分批得到的metric求平均来表示整体的metric,按需求改吧
        Args:
            pred: torch.tensor
                测试集或验证集的一个批次的预测
            y: torch.tensor
                测试集或验证集的一个批次的ground truth

        Returns:
            scores：dict
                各种性能指标的字典，务必要有scores[self.key_metric]

        """
        # 初始化self.key_metric
        self.key_metric = "acc"

        scores = {}
        correct = (torch.argmax(pred, dim=1) == y).type(torch.float).sum().item()
        scores[self.key_metric] = correct / self.eval_loader.batch_size
        return scores

    def inference(self, x):
        x = x.to(self.device)
        for model in self.model_list:
            x = model(x)
        return x


def get_loader(batch_size, transforms):
    train_set = CIFAR10(root="/home/ph/Desktop/Tutorials/Pytorch/data",
                        train=True, transform=transforms, download=True
                        )
    test_set = CIFAR10(root="/home/ph/Desktop/Tutorials/Pytorch/data",
                       train=False, transform=transforms, download=True
                       )
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=6, pin_memory=True
                              )
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True,
                             num_workers=6, pin_memory=True
                             )
    return train_loader, test_loader


def main(continue_model=None, clean_log=False, write_params=False):
    # 1.读取配置信息：arg
    arg = get_arg()
    epochs = arg.epochs
    batch_size = arg.batch_size
    init_lr = arg.init_lr
    # 2.设置训练集和验证集和测试集
    transforms = Compose([RandomRotation(degrees=60),
                          RandomHorizontalFlip(p=0.5),
                          ToTensor()]
                         )
    train_loader, test_loader = get_loader(batch_size, transforms)
    # 3.设置model，loss_fun, optimizer,lr_schedule...
    model = resnet101(pretrained=False)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=init_lr)
    loss_fn = nn.CrossEntropyLoss()
    # 4.使用模板
    model_t = ModelT([model], [optimizer], loss_fn, train_loader, test_loader)
    model_t.check_init(clean_log=clean_log, arg=arg)  # 一定要check_init()
    if continue_model is not None:  # 用于恢复训练
        model_t.load_state(continue_model)
    else:
        model_t.print_all_member(print_model=True)
        model_t.get_model_info(fake_inp=torch.randn(1, 3, 64, 64))
    start = time.time()
    # 5.一个epoch的训练逻辑
    for _ in range(epochs):
        model_t.train_loop(write_params=False)
        model_t.eval_loop(save_per_epochs=5)
        model_t.epoch += 1
    print(f"DONE!")
    model_t.print_best_metrics()
    model_t.print_final_lr()
    print(f"Total Time:{time.time() - start}")


if __name__ == '__main__':
    main(clean_log=True, write_params=False)
