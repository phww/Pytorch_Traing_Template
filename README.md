# 个人使用的Pytorch训练模板

[魔改自这个github](https://github.com/KinglittleQ/Pytorch-Template)

[我的github](https://github.com/phww/Pytorch_Traing_Template)

## 更新

### 2021年11月18日17:07:33

1. 学习率更新策略

   - lr_scheduler增加StepLR，CosineAnnealingLR。分别对应lr_scheduler_type=['step', 'annealing']

   - 增加Warm UP机制。Warm Up为从初始比较小的lr，线性增长、指数增长、保持恒定直到结束warm up恢复正常lr。对应warm_up_type=['linear','exponential','constant]。

     - 线性warm up + cos annealing ![image-20211118172503279](https://pic-1305686174.cos.ap-nanjing.myqcloud.com/image-20211118172503279.png)
     - 指数warm up + cos annealing![image-20211118172635783](https://pic-1305686174.cos.ap-nanjing.myqcloud.com/image-20211118172635783.png)

     - 恒定warm up + step![image-20211118172733514](https://pic-1305686174.cos.ap-nanjing.myqcloud.com/image-20211118172733514.png)

   - ```python
         
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
     ```

     ```python
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
                     self.linear_coef = [(base_lr - self.init_lr) / (self.total_epoch + 1e-8) 
                                         for base_lr in self.base_lrs]
                 return [self.init_lr + coef * self.last_epoch for coef in self.linear_coef]
             if self.mode == 'exponential':
                 if self._step_count == 1:
                     self.exp_coef = [(base_lr / self.init_lr) ** (1 / self.total_epoch) 
                                      for base_lr in self.base_lrs]
                 return [self.init_lr * (coef ** self.last_epoch) for coef in self.exp_coef]
     ```

     

2. self.loss_per_batch和self.eval_scores_per_batch， 返回loss字典和scores字典。同时tensorboard中会记录字典中的所有scalar。

3. 增加直接保存整个模型的接口seve_model。

4. 调整Tensorboard历史数据文件默认保存在对应的check point目录下

4. 增加使用demo：CIFAR10 数据集分类。



### 2021年06月16日20:04:40

1. lr_scheduler.ReduceLROnPlateau可以基于**最大化验证集的‘metric’或者最小化验证集的‘loss’**来调整学习率

2. 打印各种信息的函数

   ```python
   def print_best_metrics(self):
       """打印模型的最优metric"""
       for key in self.best_metric.keys():
           print(f"{key}:\t{self.best_metric[key]}")
   
   def print_final_lr(self):
       """打印学习率"""
       for i, optimizer in enumerate(self.optimizer_list):
           print(f"final_lr_{i}:{optimizer.param_groups[0]['lr']}")
   
   def print_all_member(self, print_model=False):
       """模板的信息"""
       print(15 * "=", "template config", 15 * "=")
       # 不重要，不需要打印的信息
       except_member = ["best_metric", "key_metric", "train_loader", "test_loader", "writer"]
       # 模型信息太长了，选择打印
       if not print_model:
           except_member.append("model_list")
       for name, value in vars(self).items():
           if name not in except_member:
               print(f"{name}:{value}")
   ```

   ![image-20210616222826108](https://pic-1305686174.cos.ap-nanjing.myqcloud.com/image-20210616222826108.png)

3. check_init时如果有设置**argparser**,将打印参数信息到控制台和log文件

4. 调整log文件保存在check_point目录下

5. 训练时，tensorboard不在记录第一个样本的训练loss

6. 发现并修正一个bug：保存的最佳metric是每个epoch中最后一个batch的metric，不是每个epoch的平均metric

7. tensorboard中使用一个坐标轴记录每个epoch中训练集所有样本的平均loss和验证集中所有样本的平均loss。方便观察模型是否过拟合或欠拟合

   ![avg_epoch_loss](https://pic-1305686174.cos.ap-nanjing.myqcloud.com/avg_epoch_loss.svg)

---



### 2021年06月06日20:49:26

1. 将stdout重定位到控制台和log.txt文件。这样print的信息会在控制台出现的同时，也会保存在log.txt文件中。

```python
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
```

2. 增加lr_scheduler，基于以下代码实现

```python
self.lr_scheduler_list = [torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                                     mode="max",
                                                                     factor=0.1,
                                                                     patience=3,
                                                                     cooldown=1,
                                                                     min_lr=1e-7,
                                                                     verbose=True)
                          for optimizer in self.optimizer_list]
```

3. 一个demo见[train.py]()。同时log.txt也记录了一次训练，且Tensorboard也有全部功能展示，在命令行使用tensorboard --logdir runs 查看。 

4. 评估时，也会计算loss

   ![image-20210606210506497](https://pic-1305686174.cos.ap-nanjing.myqcloud.com/image-20210606210506497.png)

---



### 2021年06月01日21:39:03

**1.使用self.model_list 代替单独的 self.model，使用self.optimizer_list代替单独的self.optimizer**

- 如果模型和优化器还是只有一个使用self.model_list = [mdoel], self.optimizer_list = [optimizer]和以前没有任何区别
- 使用list保存整个模型的多个部分，有利于为整个模型设置多个优化器。比如一个CNN+RNN网络，可以为CNN和RNN网络分别设置对应的优化器

**2.用self.best_metric字典记录所有metric的最优值**

---



## 使用方法

**见train.py!!!**

1. 下载Pytorch-training-template文件夹下的文件，推荐将文件夹改名为utils。然后将文件夹复制到模型工程目录下

2. 在训练模型代码中继承模板类并重载\__init__()函数，如下：

   ```python
   from utils.template import TemplateModel 
   from torch.utils.tensorboard import SummaryWriter
   # 数据集
   train_loader = ...
   test_loader = ...
   # 模型信息
   model_list = [model1, model2...]
   optimizer_list= [optimizer1, optimizer2...]
   loss_fn = ...
   class Trainer(TemplateModel):
       def __init__(self):
           super(Trainer, self).__init__()
        	# 必须设定
           # 模型架构
           # 将模型和优化器以list保存，方便对整个模型的多个部分设定对应的优化器
           self.model_list = None  # 模型的list
           self.optimizer_list = None  # 优化器的list
           self.criterion = None
           # 数据集
           self.train_loader = None
           self.test_loader = None
   
           # 下面的可以不设定
           # tensorboard
           self.writer = SummaryWriter() # 推荐设定
           # 训练状态
           self.global_step = 0
           self.epoch = 0
           self.best_metric = {}
           self.key_metric = None
           # 运行设备
           self.device = "cuda" if torch.cuda.is_available() else "cpu"
           # check_point 目录
           self.ckpt_dir = "./check_point"
           # 训练时print的间隔
           self.log_per_step = 5  # 推荐按数据集大小设定
   ```

3. 生成Trainer的一个实例trainer，主要使用train_loop()和eval()两个成员函数，同时注意是否要继续训练某个模型。如下：

   ```python
   epochs = 10
   def train(continue_training=False, continue_model=None):
       trainer = Trainer()
       trainer.check_init()
       trainer.get_model_info(fake_inp=torch.randn(1, 1, 32, 32))
       # 如果继续训练某个模型
       if continue_training:
           trainer.load_state(continue_model)
       # 否则直接重新训练
       for epoch in range(epochs):
           trainer.train_loop()
           trainer.eval(save_per_epochs=5)
   ```

4.训练完成后模型工程根目录下会有runs文件夹，里面是Tensorboard的文件。在模型工程根目录下的命令行中输入**tensorboard --logdir runs**查看模型训练曲线等信息，如下所示：

![image-20210527213159772](https://pic-1305686174.cos.ap-nanjing.myqcloud.com/image-20210527213159772.png)

![image-20210527214955077](https://pic-1305686174.cos.ap-nanjing.myqcloud.com/image-20210527214955077.png)

![image-20210527215024350](https://pic-1305686174.cos.ap-nanjing.myqcloud.com/image-20210527215024350.png)



## 注意事项

模板代码逻辑见代码注释，这里说几个容易出bug的点

### 单独取出模型

保存的.pth文件里面不只是有模型参数，因此如果要**单独用训练好的模型进行推断**，有两个方法。如下:

``` python
# 1.用直接读取后，用模板中的inference(self, x)方法
model_T = torch.load("best.pth")
pred = model_T.inference(inp)
# 2.读取模型参数后正常使用
state = torch.load("best.pth")
for idx in range(len(model_list)):
	model_list[idx].load_state_dict(state[f'model{idx}'])
pred = inp
for model in model_list:
	pred = model(pred)
```



### 继续训练

训练过程中每次使用eval()评估当前epoch下的模型时，会将准确率最高的模型保存在check_point目录下的best.pth文件中。同时每隔几个epoch（比如1就代表每次评估模型时都保存当前的模型状态）调用eval()时，会保存当前模型的状态在check_point目录下的epoch+数字.pth文件中。

best.pth文件不会保存当前模型在优化器中的参数（因为很占空间），只有epoch+数字.pth文件才会保存优化器中的参数。所以**要继续训练务必使用epoch+数字.pth文件**。

如图保存了优化器参数的模型文件会很大：

![image-20210527214640805](https://pic-1305686174.cos.ap-nanjing.myqcloud.com/image-20210527214640805.png)





### 计算训练集中一个批次的loss

train_loss_per_batch(self, batch)函数负责计算训练集中一个批次的loss，这个部分是可能要按需求修改的部分。因为Pytorch 中的loss函数中一般规定x和y都是float，而有些loss函数规定y要为long（比如经常用到的CrossEntropyLoss）如果[官网](https://pytorch.org/docs/stable/nn.html#loss-functions) 对y的数据类型有要求请做相应的修改。本模板除了CrossEntropyLoss将y的数据类型设为long外其他都默认x和y的数据类型为float



### 计算测试集的性能指标metric

本模板使用eval_scores_per_batch(self, batch)配合metric(self, pred, y)函数计算测试集中一批数据的metric，因为不同任务的性能指标太难统一了，这里只是实现了**多分类任务求准确率**的方法。其他任务请按需求继承这个类的时候再重载metric()函数，注意metric()函数返回数据类型为字典,且一定设定self.key_metric这个指标，如self.key_metric=“acc”。因为scores[self.key_metric]用于保存训练过程中的最优模型。这个模板使用**分批计算**metric再求全部批次的平均值的策略得到整体的metric。不会将全部的预测和ground truth保存在preds和ys中然后在cpu上进行预测。因为如果测试集或验证集太大（>50000）可能CPU内存装不下，训练会报错.**但是有的metric可能不能使用分批得到的metric求平均来表示整体的metric**,按需求改吧



### 模型summary的BUG

get_model_info()函数内使用了torchsummary这个包中的summary()方法。貌似含有Transformer结构的模型该方法会报错。但是不会影响writer.add_graph()方法在Tensorboard中绘制带Transformer结构的模型的计算图。**注意绘制了计算图的Tensorboard log文件会比较大**

![image-20210527215825803](https://pic-1305686174.cos.ap-nanjing.myqcloud.com/image-20210527215825803.png)

