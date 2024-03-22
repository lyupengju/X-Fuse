from tqdm import tqdm
import os
from tensorboardX import SummaryWriter
import argparse
import time
import random
import numpy as np
from monai.losses import DiceCELoss
from monai.utils import set_determinism
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
from monai_datamodule import organ_data_module
from monai.transforms import AsDiscrete, SaveImage
from monai.metrics import HausdorffDistanceMetric,ConfusionMatrixMetric,DiceMetric
import logging
from full_trainer import trainer
import torch.optim.lr_scheduler as lr_scheduler



os.environ['CUDA_VISIBLE_DEVICES'] = "3"
import warnings
warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser()

parser.add_argument('--data_path', type=str, default='projects/tumors/liver_tumor_dataset/', help='data directory')
parser.add_argument('--dataset', type=str, default='liver_tumor_aug', help='dataset') #'BTCV', coronary, flare2023
parser.add_argument('--exp', type=str,  default='tumor_result', help='model save path')
parser.add_argument('--max_iteration', type=int,  default=20000, help='maximum iteration number to train')
parser.add_argument('--batch_size', type=int, default=1, help='batch_size per gpu') #总数=batch*gpu数
parser.add_argument('--base_lr', type=float,  default=5e-4, help='starting learning rate') #若出现Nan or Inf found in input tensor,则 是因为学习率太大
parser.add_argument('--deterministic', type=int,  default=1, help='whether use deterministic training')
parser.add_argument('--seed', type=int,  default=0, help='random seed')
parser.add_argument('--gpu', type=list,  default=[0], help='GPU to use')
parser.add_argument('--num_classes', type=int,  default=3, help='number of classes including background')

args = parser.parse_args()
max_iterations = args.max_iteration
num_classes = args.num_classes
train_data_path = args.data_path
snapshot_path =  args.exp + "/"  #存储log 模型参数
gpus = args.gpu
batch_size = args.batch_size
base_lr = args.base_lr 

logging.disable(logging.WARNING)
writer = SummaryWriter(snapshot_path+'/log')
if args.deterministic:  #设置种子.固定随机数,及相关算法,以便复现
    cudnn.benchmark = True
    cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    set_determinism(seed=args.seed)

torch.cuda.empty_cache()
if not os.path.exists(snapshot_path):
    os.makedirs(snapshot_path)

'''choosing suitable architecture'''

from smt import  SMT
net = SMT(img_size=96, in_chans=1, num_classes=3, 
embed_dims=[30*2, 60*2, 120*2, 240*2], ca_num_heads=[3, 3, 3, -1], sa_num_heads=[-1, -1, 8, 16], mlp_ratios=[2, 2, 2, 2], 
qkv_bias=True, depths=[2, 4, 6, 2], ca_attentions=[1, 1, 1, 0], head_conv=3, expand_ratio=2).cuda()

'''validation onehot'''
post_label = AsDiscrete(to_onehot=num_classes)
post_pred = AsDiscrete(argmax=True, to_onehot=num_classes)
Dice_coefficient = DiceMetric(include_background=False,reduction='mean',get_not_nans=False)#求各类dice 的平均值
hd_95 = HausdorffDistanceMetric(include_background=False,percentile=95,reduction='mean',get_not_nans=False)
F1_score = ConfusionMatrixMetric(include_background=False,metric_name='f1_score',compute_sample =False,reduction='mean',get_not_nans=False)

# '''for monai '''
train_ds,train_loader,val_ds,val_loader = organ_data_module( task_dir  = train_data_path, dataset = args.dataset,batch_size=batch_size,cache=False,train_mode=True)
max_epoch = max_iterations//len(train_loader)+1
print("max_epoch_num: {} ".format(max_epoch))
print("{} iterations per epoch".format(len(train_loader)))  #trainloder 长度，一个epoch 多少个iteration, 一个epoch里面的batch num


'''trainer 参数设置 '''
DiceCEloss_early_stage = DiceCELoss(softmax=True,include_background = True, to_onehot_y = True,lambda_dice=1.0, lambda_ce=1.0) #对 dice 做softmax 和 label onehot, 去除背景只针对dice
optimizer = optim.AdamW(net.parameters(), lr=base_lr,weight_decay=1e-5)
scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=int(max_epoch), T_mult=1,eta_min=1e-4)#第五个epoch回到初始lr, 周期逐次multiply增加，最小lr
scaler = torch.cuda.amp.GradScaler() #混合精度加速训练

'''start trainning'''
global_step = 0
best_dice = 0
corresponding_HD = 200
corresponding_f1 = 0
best_dice_global_step = 0
epoch_num = 0
alpha = 1.0
DiceCEloss = DiceCEloss_early_stage #前期让ce weight 大
#while global_step < max_iterations:
while epoch_num <= max_epoch:

# start_time= time.time()
    global_step, best_dice, best_dice_global_step,corresponding_HD,corresponding_f1 = trainer(epoch_num,global_step,alpha,train_loader,val_loader, net,DiceCEloss,scaler,optimizer,max_iterations,max_epoch,writer,
                snapshot_path,post_label,post_pred,Dice_coefficient,hd_95,F1_score,best_dice, best_dice_global_step,corresponding_HD,corresponding_f1)
# print('train_time_per_epoch', time.time()-start_time)
    writer.add_scalar('lr', optimizer.state_dict()['param_groups'][0]['lr'], global_step)
    scheduler.step() # change lr every epoch
    epoch_num += 1

print(f"train completed, best_DSC: {best_dice:.2f} at step: {best_dice_global_step} its corresponding HD  is:{corresponding_HD}")
writer.close()