""" BigGAN: The Authorized Unofficial PyTorch release
    Code by A. Brock and A. Andonian
    This code is an unofficial reimplementation of
    "Large-Scale GAN Training for High Fidelity Natural Image Synthesis,"
    by A. Brock, J. Donahue, and K. Simonyan (arXiv 1809.11096).

    Let's go.
"""

import os
import functools
import math
import numpy as np
from tqdm import tqdm, trange


import torch
import paddle
import torch.nn as nn
# torch.nn.DataParallel 多个GPU加速


## =========================以下无使用=========================
from torch.nn import init
import torch.optim as optim
import torch.nn.functional as F
from torch.nn import Parameter as P
import torchvision
## ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

# Import my stuff
import inception_utils
## local: inception_utils.prepare_inception_metrics

##  TODO  包括采样，里面工具比较多 
import utils
##  这个比较容易，损失函数
import losses

## training conditional image model
import train_fns

## TODO 同步的批归一化方法
from sync_batchnorm import patch_replication_callback

# The main training file. Config is a dictionary specifying the configuration
# of this training run.
def run(config):

  # Update the config dict as necessary
  # This is for convenience, to add settings derived from the user-specified
  # configuration into the config-dict (e.g. inferring the number of classes
  # and size of the images from the dataset, passing in a pytorch object
  # for the activation specified as a string)
  ## *** 新增 resolution 使用 I128_hdf5 数据集, 这里也许需要使用 C10数据集
  config['resolution'] = utils.imsize_dict[config['dataset']]
  ## *** 新增 nclass_dict 加载 I128_hdf5 的类别, 这里也许需要使用 C10的类别 10类
  config['n_classes'] = utils.nclass_dict[config['dataset']]
  ## 加载 GD的 激活函数, 都用Relu, 这里的Relu是小写，不知道是否要改大写R
  config['G_activation'] = utils.activation_dict[config['G_nl']]
  config['D_activation'] = utils.activation_dict[config['D_nl']]

  ## 从头训练吧，么有历史的参数，不用改，默认的就是
  # By default, skip init if resuming training.
  if config['resume']:
    print('Skipping initialization for training resumption...')
    config['skip_init'] = True
  
  ## 日志加载，也不用改应该
  config = utils.update_config_roots(config)
  device = 'cuda'
  
  # Seed RNG
  ## 设置初始随机数种子，都为0，*** 需要修改为paddle的设置
  utils.seed_rng(config['seed'])

  # Prepare root folders if necessary
  ## 设置日志根目录，这个应该也不用改
  utils.prepare_root(config)

  # Setup cudnn.benchmark for free speed
  ## *** 需要改一下paddle的设置
  torch.backends.cudnn.benchmark = True

  # Import the model--this line allows us to dynamically select different files.
  ## *** !!! 这个方法很酷哦，直接导入BigGan的model，要看一下BigGAN里面的网络结构配置
  model = __import__(config['model'])
  ## 不用改，把一系列配置作为名字放到了实验名称中
  experiment_name = (config['experiment_name'] if config['experiment_name']
                       else utils.name_from_config(config))
  print('Experiment name is %s' % experiment_name)

  # Next, build the model
  ## *** 导入参数，需要修改两个方法 
  G = model.Generator(**config).to(device)
  D = model.Discriminator(**config).to(device)
  
   # If using EMA, prepare it
  ## *** 默认不开，可以先不改EMA部分
  if config['ema']:
    print('Preparing EMA for G with decay of {}'.format(config['ema_decay']))
    G_ema = model.Generator(**{**config, 'skip_init':True, 
                               'no_optim': True}).to(device)
    ema = utils.ema(G, G_ema, config['ema_decay'], config['ema_start'])
  else:
    G_ema, ema = None, None
  
  # FP16?
  ## C10比较小，G和D这部分也可以暂时不改，使用默认精度
  if config['G_fp16']:
    print('Casting G to float16...')
    G = G.half()
    if config['ema']:
      G_ema = G_ema.half()
  if config['D_fp16']:
    print('Casting D to fp16...')
    D = D.half()
    # Consider automatically reducing SN_eps?
  ## 把设置完结构G和D打包放入结构模型G_D中
  GD = model.G_D(G, D)
  ## *** 这两个print也许可以删掉，没必要。可能源于继承的nn.Module的一些打印属性
  print(G)
  print(D)
  ## *** 这个parameters也是继承torch的属性
  print('Number of params in G: {} D: {}'.format(
    *[sum([p.data.nelement() for p in net.parameters()]) for net in [G,D]]))
  # Prepare state dict, which holds things like epoch # and itr #
  ## 初始化统计参数记录表 不用变动
  state_dict = {'itr': 0, 'epoch': 0, 'save_num': 0, 'save_best_num': 0,
                'best_IS': 0, 'best_FID': 999999, 'config': config}

  # If loading from a pre-trained model, load weights
  ## 暂时不用预训练，所以这一块不用更改
  if config['resume']:
    print('Loading weights...')
    utils.load_weights(G, D, state_dict,
                       config['weights_root'], experiment_name, 
                       config['load_weights'] if config['load_weights'] else None,
                       G_ema if config['ema'] else None)

  # If parallel, parallelize the GD module
  ## 暂时不用管，GD 默认不并行
  if config['parallel']:
    GD = nn.DataParallel(GD)
    if config['cross_replica']:
      patch_replication_callback(GD)

  ## 日志中心，应该也可以不用管，如果需要就是把IS和FID的结果看看能不能抽出来
  # Prepare loggers for stats; metrics holds test metrics,
  # lmetrics holds any desired training metrics.
  test_metrics_fname = '%s/%s_log.jsonl' % (config['logs_root'],
                                            experiment_name)
  train_metrics_fname = '%s/%s' % (config['logs_root'], experiment_name)
  print('Inception Metrics will be saved to {}'.format(test_metrics_fname))
  test_log = utils.MetricsLogger(test_metrics_fname, 
                                 reinitialize=(not config['resume']))
  print('Training Metrics will be saved to {}'.format(train_metrics_fname))
  train_log = utils.MyLogger(train_metrics_fname, 
                             reinitialize=(not config['resume']),
                             logstyle=config['logstyle'])

  ## 这个才是重要的，这个是用来做结果统计的。
  # Write metadata
  utils.write_metadata(config['logs_root'], experiment_name, config, state_dict)

  ## *** D的数据加载，加载的过程中,get_data_loaders用到了torchvision的transforms方法
  # Prepare data; the Discriminator's batch size is all that needs to be passed
  # to the dataloader, as G doesn't require dataloading.
  # Note that at every loader iteration we pass in enough data to complete
  # a full D iteration (regardless of number of D steps and accumulations)
  D_batch_size = (config['batch_size'] * config['num_D_steps']
                  * config['num_D_accumulations'])
  loaders = utils.get_data_loaders(**{**config, 'batch_size': D_batch_size,
                                      'start_itr': state_dict['itr']})

  ## 准备评价指标，FID和IS的计算流程，可以使用np版本计算，也不用改
  # Prepare inception metrics: FID and IS
  get_inception_metrics = inception_utils.prepare_inception_metrics(config['dataset'], config['parallel'], config['no_fid'])

  ## 准备噪声和随机采样的标签组
  # Prepare noise and randomly sampled label arrays
  # Allow for different batch sizes in G
  G_batch_size = max(config['G_batch_size'], config['batch_size'])
  ## *** 有一部分torch的numpy用法，需要更改一下，获得噪声和标签
  z_, y_ = utils.prepare_z_y(G_batch_size, G.dim_z, config['n_classes'],
                             device=device, fp16=config['G_fp16'])
  # Prepare a fixed z & y to see individual sample evolution throghout training
  fixed_z, fixed_y = utils.prepare_z_y(G_batch_size, G.dim_z,
                                       config['n_classes'], device=device,
                                       fp16=config['G_fp16'])  
  fixed_z.sample_()
  fixed_y.sample_()
  # Loaders are loaded, prepare the training function
  if config['which_train_fn'] == 'GAN':
    train = train_fns.GAN_training_function(G, D, GD, z_, y_, 
                                            ema, state_dict, config)
  # Else, assume debugging and use the dummy train fn
  else:
    train = train_fns.dummy_training_function()
  # Prepare Sample function for use with inception metrics
  ## 把函数utils.sample中部分入参事先占掉，定义为新的函数sample
  sample = functools.partial(utils.sample,
                              G=(G_ema if config['ema'] and config['use_ema']
                                 else G),
                              z_=z_, y_=y_, config=config)

  print('Beginning training at epoch %d...' % state_dict['epoch'])
  # Train for specified number of epochs, although we mostly track G iterations.
  for epoch in range(state_dict['epoch'], config['num_epochs']):    
    # Which progressbar to use? TQDM or my own?
    if config['pbar'] == 'mine':
      pbar = utils.progress(loaders[0],displaytype='s1k' if config['use_multiepoch_sampler'] else 'eta')
    else:
      pbar = tqdm(loaders[0])
    for i, (x, y) in enumerate(pbar):
      # Increment the iteration counter
      state_dict['itr'] += 1
      # Make sure G and D are in training mode, just in case they got set to eval
      # For D, which typically doesn't have BN, this shouldn't matter much.
      G.train()
      D.train()
      if config['ema']:
        G_ema.train()
      if config['D_fp16']:
        x, y = x.to(device).half(), y.to(device)
      else:
        x, y = x.to(device), y.to(device)
      metrics = train(x, y)
      train_log.log(itr=int(state_dict['itr']), **metrics)
      
      # Every sv_log_interval, log singular values
      if (config['sv_log_interval'] > 0) and (not (state_dict['itr'] % config['sv_log_interval'])):
        train_log.log(itr=int(state_dict['itr']), 
                      **{**utils.get_SVs(G, 'G'), **utils.get_SVs(D, 'D')})

      # If using my progbar, print metrics.
      if config['pbar'] == 'mine':
          print(', '.join(['itr: %d' % state_dict['itr']] 
                           + ['%s : %+4.3f' % (key, metrics[key])
                           for key in metrics]), end=' ')

      # Save weights and copies as configured at specified interval
      if not (state_dict['itr'] % config['save_every']):
        if config['G_eval_mode']:
          print('Switchin G to eval mode...')
          G.eval()
          if config['ema']:
            G_ema.eval()
        train_fns.save_and_sample(G, D, G_ema, z_, y_, fixed_z, fixed_y, 
                                  state_dict, config, experiment_name)

      # Test every specified interval
      if not (state_dict['itr'] % config['test_every']):
        if config['G_eval_mode']:
          print('Switchin G to eval mode...')
          G.eval()
        train_fns.test(G, D, G_ema, z_, y_, state_dict, config, sample,
                       get_inception_metrics, experiment_name, test_log)
    # Increment epoch counter at end of epoch
    state_dict['epoch'] += 1


def main():
  # parse command line and run
  # 外部运行主程序入口时候可以输入参数，控制计算模式
  parser = utils.prepare_parser()
  ## vars() 函数返回对象object的属性和属性值的字典对象。
  ## 将parser字典化
  ## 数据集dataset 默认为 I128_hdf5
  ## 是否采用数据增强augment 默认为 0
  ## num_workers 加速数据读取的加载，默认为8， 应小于HDF5 TODO 待查为啥要小于某个数据格式
  ## no_pin_memory 作者也不确定，暂时不管，默认为FALSE TODO
  ## shuffle 数据还是需要shuffle，万一数据有聚集BATCH的情况，模型就疯了，默认为True
  ## load_in_mem 作者也不确定，估计是用来加速数据加载速度的 TODO
  ## use_multiepoch_sampler 使用多回合采样方法，默认为True，TODO multi-epoch sampler 这个东西一定要研究一下，采样相当重要，这个不知道是啥
  ## model 默认使用 BigGAN
  ## G_param 生成模型使用图像归一方法，默认使用谱归一化，还可以选择SVD或者None，TODO 利用代码加深一下SN和SVD的区别
  ## D_param 判别模型使用图像归一方法，默认使用谱归一化，还可以选择SVD或者None，TODO 利用代码加深一下SN和SVD的区别
  ## G_ch 生成模型的信道 默认64 TODO 这个是啥？
  ## D_ch 辨别模型的信道 默认64 TODO 这个是啥？
  ## G_depth 每个阶段G的resblocks的数量 TODO 盲猜和ResNet结构有关系
  ## D_depth 每个阶段D的resblocks的数量
  ## TODO D_thin和D_wide是干啥的？默认都是FALSE
  ## G_shared TODO Use shared embeddings in G 这个需要细看一下是怎么操作的
  ## shared_dim TODO 'G''s shared embedding dimensionality; if 0, will be equal to dim_z.
  ## dim_z 噪声的维度，默认为128
  ## z_var 噪声的标准差，premiere为1
  ## hier 这个是使用多层噪声，TODO 需要看一下这个是怎么实现，以及怎么个意思
  ## cross_replica TODO 这个是啥？把G模型的batchnorm再复制一遍吗
  ## 使用默认的归一化方法 mybn，看一下为啥
  ## G_nl&D_nl的激活函数
  ## G_attn和D_attn是否使用attention机制
  ## norm_style 使用归一化方法，CNN还是使用BN比较好，四者之间的区别可以百度
  ## seed 随机数种子，默认为0，控制初始化参数和数据读取的
  ## G_init 生成模型初始化方法 TODO !!! 要知晓一下ortho是什么类型的初始化方法
  ## D_init 判别模型初始化方法 TODO !!! 要知晓一下ortho是什么类型的初始化方法
  ## skip_init 跳过初始化 TODO 为什么跳过初始化，就是ideal的testing？
  ## G_lr 生成模型的学习率
  ## D_lr 辨别模型的学习率
  ## G_B1 & D_B1 beta1 % G_B2 & D_B2 beta2 TODO 属于GAN中的什么类型的参数
  ## batch_size 批大小 64
  ## G_batch_size TODO 为何还要写G的batch size？？
  ## num_G_accumulations TODO 把G的梯度相加又是为了什么？？
  ## num_D_steps TODO 每一步G需要跑几次D，D需要多训练训练，如果G跑太前面就太逼近原图了？
  ## num_D_accumulations TODO 把D的梯度相加又是为了什么？？
  ## split_D 运行D两次，而不是连接输入 TODO ？
  ## num_epochs @@ 训练的轮次数量
  ## parallel 默认是FALSE 训练时是否使用多GPUs一起
  ## G_fp16 & D_fp16 & G_mixed_precision & D_mixed_precision 对于精度的使用，一个是加快计算，一个减小模型大小.有的时候会带一点准确率的损失，所以mix表示预测的时候用更高的精度
  ## accumulate_stats 默认FALSE TODO 积累统计数据
  ## num_standing_accumulations TODO 这个和楼上那个什么关系？!!!

  #@@ 接下来是运行模式
  ## G_eval_mode TODO 每次采样或者测试的时候，都评估一下G？
  ## save_every 每2000迭代储存一下模型
  ## num_save_copies TODO 储几个备份？默认2个
  ## num_best_copies 存几个最好的版本，默认2个
  ## which_best 依据IS或者FID作为评价指标去储存模型
  ## no_fid 是否只计算IS，不计算FID。默认都计算
  ## test_every 没多少论测试一下，默认5000
  ## num_inception_images TODO 用于计算初始度量的样本数量50000
  ## hashname 是否使用HasName而不是配置中的类别，默认不使用
  ## base_root 默认储存所有权重,采样，数据，日志
  ## data_root TODO !!!数据默认存在哪里
  ## weights_root，logs_root，samples_root 本地存在哪儿
  ## pbar 是否使用进度条，默认使用mine
  ## name_suffix 命名添加后缀
  ## experiment_name 自定义存储实验名称
  ## config_from_name 是否使用hash实验名

  #@@ EMA 期望最大化网络配置参数
  ## ema 是否保存G的ema参数
  ## ema_decay EMA的衰减率
  ## use_ema 是否在G中使用评估
  ## ema_start 什么时候去更新EMA权重

  #@@ SV stuff 奇异值的迭代性质
  ## adam_eps adam的epsilon value，TODO 看一下它在干嘛
  ## BN_eps & SN_eps 批归一化和谱归一化的参数
  ## num_G_SVs G的奇异值追踪数量，默认为1
  ## num_D_SVs D的奇异值追踪数量，默认为1
  ## num_G_SV_itrs G的迭代次数，默认为1
  ## num_D_SV_itrs D的迭代次数，默认为1

  #@@ Ortho reg stuff 这个又是什么
  ## 控制Ortho的迭代性质

  ## which_train_fn 默认是GAN
  ## load_weights 加载哪类参数，是copy的或者历史最好的etc.
  ## resume TODO Resume training是从历史记录开始训练，可以用于预训练

  ## Log stuff 日志系统
  ## logstyle 日志类型
  ## log_G_spectra 记录G的头3个奇异值
  ## log_D_spectra 记录D的头3个奇异值
  ## sv_log_interval 每个多少论记录一次

  #@@ Arguments for sample.py; not presently used in train.py
  ## add_sample_parser 增加一些采样的参数
  config = vars(parser.parse_args())
  print(config)
  ## 将运行字典传入GAN模型主程序
  run(config)

if __name__ == '__main__':
  main()