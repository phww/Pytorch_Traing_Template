import time
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary
import os
import os.path as osp
import sys
import torch
from torch.optim.lr_scheduler import _LRScheduler


class WarmUpLR(_LRScheduler):

    def __init__(self, optimizer, total_epoch, init_lr=1e-6, mode='linear', last_epoch=-1, verbose=False):

        self.total_epoch = total_epoch
        self.init_lr = init_lr
        self.mode = mode

        super(WarmUpLR, self).__init__(optimizer, last_epoch, verbose)
        for base_lr in self.base_lrs:
            assert self.init_lr < base_lr

    def get_lr(self):
        if self.mode == 'constant':
            if self._step_count > self.total_epoch:
                return self.base_lrs
            return [self.init_lr] * len(self.base_lrs)
        if self.mode == 'linear':
            if self._step_count == 1:
                self.linear_coef = [(base_lr - self.init_lr) / (self.total_epoch + 1e-8) for base_lr in self.base_lrs]
            return [self.init_lr + coef * self.last_epoch for coef in self.linear_coef]
        if self.mode == 'exponential':
            if self._step_count == 1:
                self.exp_coef = [(base_lr / self.init_lr) ** (1 / self.total_epoch) for base_lr in self.base_lrs]
            return [self.init_lr * (coef ** self.last_epoch) for coef in self.exp_coef]


class Logger(object):
    """将stdout重定位到log文件和控制台
        即print时会将信息打印在控制台的
        同时也将信息保存在log文件中
    """

    def __init__(self, filename):
        self.console = sys.stdout
        self.log = open(filename, "a")

    def write(self, message):
        self.console.write(message)
        self.log.write(message)

    # 清空log文件中的内容
    def clean(self):
        self.log.truncate(0)

    def flush(self):
        pass


class TemplateModel:

    def __init__(self):
        # 必须设定
        # 模型架构
        # 将模型和优化器以list保存，方便对整个模型的多个部分设定对应的优化器
        self.model_list = None  # 模型的list
        self.optimizer_list = None  # 优化器的list
        self.criterion = None
        # 数据集
        self.train_loader = None
        self.eval_loader = None
        # check_point 目录
        self.ckpt_dir = "./check_point/" + time.strftime("%Y-%m-%d::%H:%M:%S")
        # 运行设备
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # 下面的可以不设定
        # 推荐设置
        self.lr_scheduler_list = None
        self.lr_scheduler_type = None  # None "metric" "loss" "annealing" "step"
        self.warm_up_epoch = 0
        self.warm_up_mode = 'linear'
        self.warm_up_scheduler_list = None
        # tensorboard
        self.writer = SummaryWriter(log_dir=os.path.join(self.ckpt_dir, 'runs'))

        # 按需求设定
        # 训练时print的间隔
        self.log_per_step = 5  # 推荐按数据集大小设定

        # 训练状态
        self.global_step = 0
        self.global_step_eval = 0
        self.epoch = 1
        self.best_metric = {}
        self.key_metric = None  # 记得在self.metric中设定

    def check_init(self, log_name="log.txt", clean_log=False, arg=None, use_tb=True):
        # 检测摸板的初始状态，可以在这加上很多在训练之前的操作
        assert isinstance(self.model_list, list)
        assert isinstance(self.optimizer_list, list)
        assert self.criterion
        assert self.train_loader
        assert self.eval_loader
        assert self.device
        assert self.ckpt_dir
        assert self.log_per_step
        assert self.lr_scheduler_type in [None, "metric", "loss", "annealing", "step"]
        # 建立check point 目录
        if not osp.exists(self.ckpt_dir):
            os.makedirs(self.ckpt_dir)
        # 默认的warm up，也可以自定义self.warm_up_scheduler_list
        if self.warm_up_epoch > 0 and self.warm_up_scheduler_list is None:
            self.warm_up_scheduler_list = [WarmUpLR(optimizer,
                                                    total_epoch=self.warm_up_epoch,
                                                    last_epoch=-1,
                                                    mode=self.warm_up_mode,
                                                    verbose=True
                                                    )
                                           for optimizer in self.optimizer_list]

        # 默认的lr_scheduler，也可以自定义self.lr_scheduler_list
        if self.lr_scheduler_list is None and self.lr_scheduler_type is not None:
            # 如果以测试集的metric为学习率改变的依据，选择mode="max";以loss为依据，选择mode="min"
            if self.lr_scheduler_type in ['metric', 'loss']:
                mode = None
                if self.lr_scheduler_type == "metric":
                    mode = "max"
                elif self.lr_scheduler_type == "loss":
                    mode = "min"
                self.lr_scheduler_list = [torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                                                     mode=mode,
                                                                                     factor=0.1,
                                                                                     patience=5,
                                                                                     cooldown=5,
                                                                                     min_lr=1e-9,
                                                                                     verbose=True
                                                                                     )
                                          for optimizer in self.optimizer_list]
            elif self.lr_scheduler_type == 'step':
                self.lr_scheduler_list = [torch.optim.lr_scheduler.StepLR(optimizer,
                                                                          step_size=15,
                                                                          gamma=0.1,
                                                                          last_epoch=-1,
                                                                          verbose=True
                                                                          )
                                          for optimizer in self.optimizer_list]
            elif self.lr_scheduler_type == 'annealing':
                self.lr_scheduler_list = [torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                                                     T_max=15,
                                                                                     eta_min=0,
                                                                                     last_epoch=-1,
                                                                                     verbose=True
                                                                                     )
                                          for optimizer in self.optimizer_list]

        # 设置log,log保存目录为"self.ckpt_dir/log_name"
        logger = Logger(os.path.join(self.ckpt_dir, log_name))
        if clean_log:
            logger.clean()
        sys.stdout = logger
        print(time.strftime("%Y-%m-%d::%H-%M-%S"))
        # 如果有，打印arg
        if arg is not None:
            print(15 * "=", "args", 15 * "=")
            arg_dict = arg.__dict__
            for key in arg_dict.keys():
                print(f"{key}:{arg_dict[key]}")
        # 不使用TensorBoard，比如debug时
        if not use_tb:
            self.writer = None
        # 清空cuda中的cache
        torch.cuda.empty_cache()
        # 模型默认放置于GPU上
        for model in self.model_list:
            model.to(self.device)

    def load_state(self, fname, optim=True, lr_list=None):
        # 读取保存的模型到模板之中。如果要继续训练的模型optim=True；使用最佳模型做推断optim=False
        state = torch.load(fname)
        for idx, model in enumerate(self.model_list):
            if isinstance(model, torch.nn.DataParallel):  # 多卡训练
                model.module.load_state_dict(state[f'model{idx}'])
            else:  # 非多卡训练
                model.load_state_dict(state[f'model{idx}'])
            # 恢复一些状态参数
            if optim and f'optimizer_list{idx}' in state:
                self.optimizer_list[idx].load_state_dict(state[f'optimizer_list{idx}'])
                # 改变先前模型的优化器中保存的学习率
                if lr_list is not None:
                    self.change_lr(lr_list)

        self.global_step = state['global_step']
        self.global_step_eval = state["global_step_eval"]
        self.epoch = state['epoch']
        self.best_metric = state['best_metric']
        self.key_metric = state['key_metric']
        print('load model state from {}'.format(fname))

    def save_state(self, fname, optim=True):
        # 保存模型，其中最佳模型不用保存优化器中的参数。
        # 而训练过程中保存的其他模型需要保存优化器中的梯度以便继续训练
        state = {}
        for idx, model in enumerate(self.model_list):
            if isinstance(model, torch.nn.DataParallel):
                state[f'model{idx}'] = model.module.state_dict()
            else:
                state[f'model{idx}'] = model.state_dict()
            # 训练过程中的模型除了保存模型的参数外，还要保存当前训练的状态：optim中的参数
            if optim:
                state[f'optimizer_list{idx}'] = self.optimizer_list[idx].state_dict()
        state['global_step'] = self.global_step
        state['global_step_eval'] = self.global_step_eval
        state['epoch'] = self.epoch
        state['best_metric'] = self.best_metric
        state['key_metric'] = self.key_metric
        torch.save(state, fname)
        print('save model state at {}'.format(fname))

    def save_model(self, fname):
        """
        直接保存整个模型，不是模板！模型复杂时，生成文件会很大
        """
        all_models = {}
        for idx, model in enumerate(self.model_list):
            if isinstance(model, torch.nn.DataParallel):
                all_models[f'model{idx}'] = model.module
            else:
                all_models[f'model{idx}'] = model
        torch.save(all_models, fname)
        print('save model at {}'.format(fname))

    def train_loop(self, write_params=False, clip_grad=None):
        """训练一个epoch，一般来说不用改"""
        print("*" * 15, f"epoch:{self.epoch}", "*" * 15)
        for model in self.model_list:
            model.train()

        running_loss_dict = None  # other loss only for tensorboard
        all_avg_loss_dict = None
        cnt_loss = 0
        for step, batch in enumerate(self.train_loader):
            self.global_step += 1
            batch_loss_dict = self.loss_per_batch(batch)
            # total loss used for backward
            batch_tol_loss = batch_loss_dict['tol_loss']

            # 多个优化器需要按逆序更新每一个优化器
            for optimizer in reversed(self.optimizer_list):
                optimizer.zero_grad()

            batch_tol_loss.backward()

            # 截断太大的梯度
            if clip_grad is not None:
                for model in self.model_list:
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad, norm_type=2)

            for optimizer in reversed(self.optimizer_list):
                optimizer.step()

            # 累计running loss
            if running_loss_dict is None:
                running_loss_dict = batch_loss_dict
            else:
                for loss, value in batch_loss_dict.items():
                    running_loss_dict[loss] += value

            # 记录损失除了训练刚开始时是用此时的loss外，其他都是用一批loss的平均loss
            # 但是为了tensorboard的曲线好看，不记录第一个样本的loss
            if self.global_step == 1:
                # if self.writer is not None:
                #     self.writer.add_scalar('train_loss', batch_loss.item(), self.global_step):
                print(f"loss:{batch_tol_loss.item() : .5f}\t"
                      f"cur:[{step * self.train_loader.batch_size}]\[{len(self.train_loader.dataset)}]"
                      )

            # 每处理self.log_per_step批数据就打印和记录这批数据的的平均loss
            elif (step + 1) % self.log_per_step == 0:
                avg_loss_dict = {}
                for loss, value in running_loss_dict.items():
                    avg_loss_dict[loss] = value / (self.log_per_step * len(batch))
                for avg_loss, value in avg_loss_dict.items():
                    print(f"{avg_loss}:{value : .5f}\t"
                          f"cur:[{(step + 1) * self.train_loader.batch_size}]\[{len(self.train_loader.dataset)}]"
                          )
                # 累计avg_loss
                if all_avg_loss_dict is None:
                    all_avg_loss_dict = avg_loss_dict
                else:
                    for loss, value in avg_loss_dict.items():
                        all_avg_loss_dict[loss] += value

                cnt_loss += 1
                for loss, value in running_loss_dict.items():
                    running_loss_dict[loss] = 0.0  # running loss 归零.用于累计下N批次数据的loss

                # Tensorboard记录self.log_per_step批数据的的平均loss
                if self.writer is not None:
                    # 平均loss
                    if len(avg_loss_dict) == 1:
                        self.writer.add_scalar('train_loss', avg_loss_dict['tol_loss'], self.global_step)
                    else:
                        self.writer.add_scalars('train_loss', avg_loss_dict, self.global_step)

                    # write_params=True在Tensorboard中记录模型的参数和梯度的分布情况，但是也费时间。默认关闭
                    if write_params:
                        for model in self.model_list:
                            for tag, value in model.named_parameters():
                                tag = tag.replace('.', '/')
                                self.writer.add_histogram('weights/' + tag, value.data.cpu().numpy())
                                if value.grad is not None:  # 在FineTurn时有些参数被冻结了，没有梯度。也就不用记录了
                                    self.writer.add_histogram('grads/' + tag, value.grad.data.cpu().numpy())

        # 训练一个epoch后，记录训练集全部样本的平均loss
        avg_batch_loss_dict = {}
        for loss, value in all_avg_loss_dict.items():
            avg_batch_loss_dict[loss] = value / cnt_loss
        for loss, value in avg_batch_loss_dict.items():
            print(f"epoch:{self.epoch}\tavg_epoch|{loss}:{value:.5f}")

        # Tensorboard记录一个epoch的平均loss
        if self.writer is not None:
            if len(avg_batch_loss_dict) == 1:
                self.writer.add_scalar("avg_epoch_train_loss",
                                       avg_batch_loss_dict['tol_loss'],
                                       self.epoch
                                       )
            else:
                self.writer.add_scalars("avg_epoch_train_loss", avg_batch_loss_dict, self.epoch)
            # 用于观测模型是否过拟合/欠拟合
            self.writer.add_scalars("avg_epoch_train&eval|tol_loss",
                                    {"train": avg_batch_loss_dict["tol_loss"]},
                                    self.epoch
                                    )

        # step模式更新lr,在每个train_loop后更新
        # annealing模式 更新lr. 根据迭代step更新
        if self.lr_scheduler_type == 'annealing':
            self.update_lr_scheduler()
        if self.lr_scheduler_type == 'step':
            self.update_lr_scheduler()

        # self.epoch += 1

    def eval_loop(self, save_per_epochs=1):
        """一个epoch的评估。一般不用改"""
        print("-" * 15, "Evaluation", "-" * 15)
        # 在整个测试集上做评估，使用分批次的metric的平均值表示训练集整体的metric
        with torch.no_grad():
            # 先计算，当前模型在验证集上时的loss
            # 累计running loss
            running_loss_dict = None
            all_avg_loss_dict = None
            # 保存各种metric的得分
            scores = {}
            cnt = 0
            cnt_loss = 0
            for step, batch in enumerate(self.eval_loader):
                cnt += 1  # 记录迭代了多少batch
                self.global_step_eval += 1
                # 计算评估时多个批次的平均loss
                batch_loss_dict = self.loss_per_batch(batch)
                # 累计running loss
                if running_loss_dict is None:
                    running_loss_dict = batch_loss_dict
                else:
                    for loss, value in batch_loss_dict.items():
                        running_loss_dict[loss] += value

                # 每self.log_per_step个step打印信息
                if (step + 1) % self.log_per_step == 0:
                    avg_loss_dict = {}
                    for loss, value in running_loss_dict.items():
                        avg_loss_dict[loss] = value / (self.log_per_step * len(batch))
                    for avg_loss, value in avg_loss_dict.items():
                        print(f"{avg_loss}:{value : .5f}\t"
                              f"cur:[{(step + 1) * self.eval_loader.batch_size}]\[{len(self.eval_loader.dataset)}]"
                              )

                    if all_avg_loss_dict is None:
                        all_avg_loss_dict = avg_loss_dict
                    else:
                        for loss, value in avg_loss_dict.items():
                            all_avg_loss_dict[loss] += value

                    cnt_loss += 1
                    for loss, value in running_loss_dict.items():
                        running_loss_dict[loss] = 0.0  # running loss 归零.用于累计下N批次数据的loss

                    if self.writer is not None:
                        # 平均loss
                        if len(avg_loss_dict) == 1:
                            self.writer.add_scalar('eval_loss', avg_loss_dict['tol_loss'], self.global_step_eval)
                        else:
                            self.writer.add_scalars('eval_loss', avg_loss_dict, self.global_step_eval)

                # 分批计算metric，并累计
                if step == 0:
                    scores = self.eval_scores_per_batch(batch)  # 每个epoch初始化scores
                # 累加所有批次的metric。这里有个问题：
                # 准确率可以使用分批的准确率之和除以分批数量得到，并且与用全部数据集计算准确率是等价的
                # 但是有的metric使用一部分批次的计算出来的结果可能与使用全部数据集计算出来的结果不同
                else:
                    batch_scores = self.eval_scores_per_batch(batch)
                    for key in scores.keys():
                        scores[key] += batch_scores[key]
                if self.global_step_eval == 1:
                    self.best_metric = scores  # 第一次eval时，初始化self.best_metric

            # 整个验证集上的平均running_loss
            avg_batch_loss_dict = {}
            for loss, value in all_avg_loss_dict.items():
                avg_batch_loss_dict[loss] = value / cnt_loss
            # for loss, value in avg_batch_loss_dict.items():
            #     print(f"epoch:{self.epoch}\tavg_epoch_{loss}:{value:.5f}")

            # 整个验证集上的平均metric
            for key in scores.keys():
                scores[key] /= cnt
            # 打印信息:一个epoch上的平均loss和关键指标
            for avg_loss, value in avg_batch_loss_dict.items():
                print(f"epoch:{self.epoch}\tavg_epoch_eval|{avg_loss}:{value:.5f}")
            for avg_metric, value in scores.items():
                print(f'epoch:{self.epoch}\tavg_metric|{avg_metric}:{value:.5f}')

            # 根据scores[self.key_metric]来判定是否保存最佳模型.
            # self.key_metric需要在metric函数中初始化，分类任务常用self.key_metric = "acc"
            for key in scores.keys():
                # 更新所有metric的最佳结果到self.best_metric字典中
                if scores[key] >= self.best_metric[key]:
                    self.best_metric[key] = scores[key]

                    # 保存最佳模型
                    if key == self.key_metric:
                        self.save_state(osp.join(self.ckpt_dir, f'best.pth'), False)

            # 每save_per_epochs次评估就保存当前模型，这种模型一般用于继续训练
            if self.epoch % save_per_epochs == 0:
                self.save_state(osp.join(self.ckpt_dir, f'epoch{self.epoch}.pth'), True)

            # Tensorboard
            if self.writer is not None:
                # 记录每个epoch的metric
                # for key in scores.keys():
                self.writer.add_scalars(f"avg_epoch_eval_metric", scores, self.epoch)

                # 记录一个epoch中验证集中全部样本的平均loss
                if len(avg_batch_loss_dict) == 1:
                    self.writer.add_scalar("avg_epoch_eval_loss",
                                           avg_batch_loss_dict['tol_loss'],
                                           self.epoch
                                           )
                else:
                    self.writer.add_scalars("avg_epoch_eval_loss", avg_batch_loss_dict, self.epoch)
                self.writer.add_scalars("avg_epoch_train&eval|tol_loss",
                                        {"eval": avg_batch_loss_dict['tol_loss']},
                                        self.epoch
                                        )

            # 根据验证集的情况更新lr.在每个eval_loop后更新
            if self.lr_scheduler_type == 'metric':
                self.update_lr_scheduler(key_metric=scores[self.key_metric])
            elif self.lr_scheduler_type == 'loss':
                self.update_lr_scheduler(avg_batch_loss=avg_batch_loss_dict['tol_loss'])
            # self.epoch += 1
        return scores[self.key_metric]

    def update_lr_scheduler(self, **kwargs):
        """
        更新lr_scheduler.使用lr_scheduler.step()的时机主要分3种
        1. 在train_loop后使用，此时lr_scheduler根据训练的epoch来更新lr.适用于"annealing", "step"type
        2. 在eval_loop后使用，此时lr_scheduler根据评估的指标来更新lr.适用于"metric", "loss"type
        3. 在train_loop中的loos_per_batch后使用，此时lr_scheduler根据训练的iter来更新lr.
        warm up机制使用epoch作为lr更新的milestone
        """
        assert self.lr_scheduler_list is not None
        if self.warm_up_epoch > 0 and self.epoch <= self.warm_up_epoch:
            print(f'Warm UP:\tcur epoch[{self.epoch}]/[{self.warm_up_epoch}]')
            for warm_up_scheduler in self.warm_up_scheduler_list:
                warm_up_scheduler.step()

        else:
            if self.lr_scheduler_type in ["annealing", "step"]:
                if self.lr_scheduler_type == 'annealing':
                    for lr_scheduler in self.lr_scheduler_list:
                        lr_scheduler.step()
                if self.lr_scheduler_type == 'step':
                    for lr_scheduler in self.lr_scheduler_list:
                        lr_scheduler.step()
            if self.lr_scheduler_type in ["metric", "loss"]:
                for i, lr_scheduler in enumerate(self.lr_scheduler_list):
                    if self.lr_scheduler_type == "metric":
                        lr_scheduler.step(kwargs['key_metric'])
                    elif self.lr_scheduler_type == "loss":
                        lr_scheduler.step(kwargs['avg_batch_loss'])
        # tensorboard记录学习率
        if self.writer is not None:
            for i, lr_scheduler in enumerate(self.lr_scheduler_list):
                self.writer.add_scalar(f"lr_scheduler{i}",
                                       self.optimizer_list[i].param_groups[0]["lr"],
                                       self.epoch
                                       )

    def loss_per_batch(self, **kwargs):
        """
        每个batch，计算损失的逻辑。按需求修改。
        保证输出为loss字典，其中必须包含loss['tol_loss']
        """
        raise NotImplementedError

    def eval_scores_per_batch(self, **kwargs):
        """
        每个batch，计算评估分数的逻辑，配合metric使用。按需求修改
        保证输出为scores字典，其中必须包含scores[self.key_metric]
        """
        raise NotImplementedError

    def metric(self, **kwargs):
        """
        计算性能指标，配合eval_scores_per_batch使用。
        在这里面需要设置self.key_metric
        """
        raise NotImplementedError

    def inference(self, **kwargs):
        """不需要使用就pass"""
        raise NotImplementedError

    def get_model_info(self, fake_inp):
        # 输出模型信息和在Tensorboard中绘制模型计算图
        print(15 * "=", "model info", 15 * "=")
        if self.writer is not None:
            for model in self.model_list:
                self.writer.add_graph(model, fake_inp.to(self.device))
                # summary对transformer有BUG
                print(summary(model, batch_size=32, input_size=fake_inp.shape[1:], device=self.device))

    def num_parameters(self):
        num = 0
        for model in self.model_list:
            num += sum([p.data.nelement() for p in model.parameters()])
        return num

    def print_best_metrics(self):
        print(f"Bast Metric")
        for key in self.best_metric.keys():
            print(f"{key}:\t{self.best_metric[key]}")

    def print_final_lr(self):
        print("Final Learning Rate:")
        for i, optimizer in enumerate(self.optimizer_list):
            print(f"final_lr_{i}:{optimizer.param_groups[0]['lr']}")

    def print_all_member(self, print_model=False):
        print(15 * "=", "template config", 15 * "=")
        # 不重要，不需要打印的信息
        except_member = ["best_metric", "key_metric", "train_loader", "test_loader", "writer"]
        # 模型信息太长了，选择打印
        if not print_model:
            except_member.append("model_list")
        for name, value in vars(self).items():
            if name not in except_member:
                print(f"{name}:{value}")

    def change_lr(self, lr_list):
        """
        改变优化器中记录的学习率，主要用于使用已有的模型继续训练时。
        指定新的学习率，而不是已有模型的优化器中保存的学习率。
        """
        for i, optimizer in enumerate(self.optimizer_list):
            optimizer.param_groups[0]['lr'] = lr_list[i]
