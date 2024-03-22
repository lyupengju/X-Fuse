import os
import argparse
import glob
import torch
from torch.utils.data import random_split
import matplotlib.pyplot as plt
import re
from pathlib import Path
import numpy as np
from monai import transforms
## Online Tumor Generation
from TumorGenerated import TumorGenerated
from monai.utils import set_determinism
from monai.data import (
    ThreadDataLoader,
    DataLoader,
    CacheDataset,
    Dataset,
    PersistentDataset,
    load_decathlon_datalist,
    decollate_batch,
    set_track_meta
    
)

import warnings
warnings.filterwarnings('ignore')   
set_determinism(seed=50)

def organ_data_module(task_dir,dataset = '', batch_size=1,cache=False,train_mode=False,test_mode = False):
 
    data_root = task_dir
    data_dicts =[]
    train_images = sorted(glob.glob(os.path.join(data_root, 'imagesTr', '*.nii.gz')))
    train_labels = sorted(glob.glob(os.path.join(data_root, 'labelsTr', '*.nii.gz')))
    for image_name, label_name in zip(train_images, train_labels):
        assert re.findall("\d+",Path(image_name).stem) == re.findall("\d+",Path(label_name).stem)
        data_dicts.append({'image': image_name, 'label': label_name})
    num_subjects = len(data_dicts)
    print(num_subjects)
    train_files, val_files = data_dicts[:-20], data_dicts[-20:]
    #print(train_files)
    print(len(val_files))


    if train_mode:
        sample_patch = (96,96,96)
        num_samples= 1
        # resize_fine = (192,192,192)

        if dataset == 'liver_tumor_aug':
     
            train_transforms = transforms.Compose(
            [
            transforms.LoadImaged(keys=["image", "label"]),
            transforms.AddChanneld(keys=["image", "label"]),
            TumorGenerated(keys=["image", "label"], prob=0.9), # here we use online 
           
            transforms.Spacingd(keys=["image", "label"], pixdim=(1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
       
            transforms.ScaleIntensityRangePercentilesd(keys=["image"],lower= 1, upper= 99, b_min=0, b_max=1, clip=True, channel_wise=True), #channel_wise=True for MRI
            transforms.CropForegroundd(keys=["image", "label"], source_key="image"),
            transforms.NormalizeIntensityd(keys=['image'], nonzero=True,channel_wise=True),
            transforms.SpatialPadd(keys=["image", "label"], mode=["minimum", "constant"], spatial_size=[96, 96, 96]),
            transforms.RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=(96, 96, 96),
                pos=2,
                neg=1,
                num_samples=num_samples,
                # image_key="image",
                # image_threshold=0,
            ),
            transforms.RandFlipd(keys=["image", "label"], prob=0.2, spatial_axis=0),
            transforms.RandFlipd(keys=["image", "label"], prob=0.2, spatial_axis=1),
            transforms.RandFlipd(keys=["image", "label"], prob=0.2, spatial_axis=2),
            transforms.RandRotate90d(keys=["image", "label"], prob=0.2, max_k=3),
            transforms.RandScaleIntensityd(keys="image", factors=0.1, prob=0.15),
            transforms.RandShiftIntensityd(keys="image", offsets=0.1, prob=0.15),
            transforms.ToTensord(keys=["image", "label"]),
            transforms.RandAffined( keys=['image', 'label'],mode=('bilinear', 'nearest'),prob=0.1, 
            spatial_size=(96, 96, 96),rotate_range=(0, 0, np.pi / 15),scale_range=(0.1, 0.1, 0.1)),
            ]
            )

            val_transforms = transforms.Compose(
            [
            transforms.LoadImaged(keys=["image", "label"]),
#             transforms.ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
#             transforms.NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            transforms.AddChanneld(keys=["image", "label"]),
            # transforms.Orientationd(keys=["image", "label"], axcodes="RAS"),
            transforms.Spacingd(keys=["image", "label"], pixdim=(1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
            transforms.ScaleIntensityRangePercentilesd(keys=["image"],lower= 1, upper= 99, b_min=0, b_max=1, clip=True, channel_wise=True), #channel_wise=True for MRI
            # transforms.ScaleIntensityRanged(keys=["image"], a_min=-21, a_max=189, b_min=0.0, b_max=1.0, clip=True),
            transforms.CropForegroundd(keys=["image", "label"], source_key="image"),
            transforms.NormalizeIntensityd(keys=['image'], nonzero=True,channel_wise=True),
            transforms.SpatialPadd(keys=["image", "label"], mode=["minimum", "constant"], spatial_size=[96, 96, 96]),
            transforms.ToTensord(keys=["image", "label"]),
            ]
            )



        #先把数据cache起来，减少训练时间, 但是容易爆显存
        if cache:
            # cache_dir_train = "/home/plyu/Documents/projects/flare2023_competition/code/stage2_patch/train_cache/"
            # cache_dir_val = "/home/plyu/Documents/projects/flare2023_competition/code/stage2_patch/val_cache/"
            train_ds = CacheDataset(
                data=train_files,
                transform=train_transforms,
                cache_num=len(train_files),
                cache_rate=1.0,
                num_workers=12,
                copy_cache=False
            )
            # train_ds = PersistentDataset(data=train_files, transform=train_transforms, cache_dir= cache_dir_train)

            # disable multi-workers because `ThreadDataLoader` works with multi-threads
            train_loader = DataLoader(train_ds, num_workers=0, batch_size=batch_size, shuffle=True)
            val_ds = CacheDataset(
                data=val_files, transform=val_transforms, cache_num=len(val_files), cache_rate=1.0, num_workers=4,copy_cache=False
            )
            # val_ds = PersistentDataset(data=val_files, transform=val_transforms, cache_dir=cache_dir_val)

            val_loader = DataLoader(val_ds, num_workers=0, batch_size=1,shuffle=False,pin_memory=True)
        else:
            train_ds = Dataset(
                data=train_files,
                transform=train_transforms,   
            )
            train_loader =    DataLoader(train_ds, num_workers=8, batch_size=batch_size, shuffle=True,pin_memory=True) #num_worker必须为0
            val_ds = Dataset(
                data=val_files, transform=val_transforms
            )
            val_loader = DataLoader(val_ds, num_workers=0, batch_size=1,pin_memory=True) #因shape 不一样，batchsize暂定为1

            # for i in  val_ds:
            #      print(i["image"].meta["filename_or_obj"])
        return train_ds,train_loader,val_ds,val_loader 
        #return train_ds,val_ds 


  
