import torch
import os
from tqdm import tqdm
import sys
from monai.inferers import sliding_window_inference,SliceInferer
import argparse
from monai.transforms import AsDiscrete, SaveImage
from monai.data import decollate_batch 
from datetime import datetime
import numpy as np
from medpy.metric.binary import __surface_distances,dc
from monai.metrics import HausdorffDistanceMetric,SurfaceDistanceMetric,DiceMetric

def validationer(val_loader,net,global_step,max_iterations,post_label,post_pred,Dice_coefficient,HD_95,F1_score):
    net.eval()
    patch_size =  (96,96,96)
    sw_batch_size = 4
    with torch.no_grad():
        epoch_iterator_val = tqdm(val_loader, desc="validating", dynamic_ncols=True)
        for step, batch in enumerate(epoch_iterator_val):
            #数据cache 在gpu上, 推理占很多内存，cpu  run 较慢
            #val_images, val_labels = (batch["image"][tio.DATA].cuda(), batch["label"][tio.DATA].cuda())
            val_images, val_labels = batch["image"].cuda(), batch["label"].cuda()

            # one_metrics=torch.ones_like(val_labels)
            # val_labels=torch.where(val_labels>13,one_metrics,val_labels)

            
            # print(val_images.shape)
            # print(val_labels.shape)
            with torch.cuda.amp.autocast():
                    # val_outputs = net(val_images)                    
                    val_outputs = sliding_window_inference(val_images,patch_size,sw_batch_size,net,0.5)   
                 
            '''预测和标签都 onehot 处理'''


            val_labels_list = decollate_batch(val_labels)
            val_labels_convert = [
                post_label(val_label_tensor) for val_label_tensor in val_labels_list
            ]
            # print(torch.unique(val_output_convert[0]))
            # print(torch.unique(val_labels_convert[0]))
            val_outputs_list = decollate_batch(val_outputs)
            val_output_convert = [
                post_pred(val_pred_tensor) for val_pred_tensor in val_outputs_list
            ]
            # print(val_output_convert)
            # print(val_output_convert)
          


            Dice_coefficient(val_output_convert, val_labels_convert)#计算dice

            HD_95(val_output_convert, val_labels_convert)
            F1_score(val_output_convert, val_labels_convert)
            
       
            # print(len(val_output_convert))
            # print(val_output_convert[0].shape)
            # print(val_labels_convert[0].shape)
    
        # aggregate the final mean dice result 整个val_loader
        DSC= Dice_coefficient.aggregate().item()#tolist()
        # print(DSC)
        # average_dice = round((DSC[0]+DSC[1])/2,2)
        # DSC = round(DSC, 2) #保留两位小数
        HD95= HD_95.aggregate().item()#.tolist()
        # average_hd = round((HD95[0]+HD95[1])/2,2)
        # print(HD95)
        # HD95 = round(HD95, 2)
        F1 = F1_score.aggregate()[0].item()#tolist(
        # average_f1 = round((F1[0]+F1[1])/2,2)

        # F1 = round(F1, 2)


        
        # reset the status for next validation round
        Dice_coefficient.reset()
        HD_95.reset()
        F1_score.reset()
        # epoch_iterator_val.set_description("validation (%d / %d Steps) (dice_liver=%.2f), (dice_tumor=%.2f),(HD95_liver=%.2f), (HD95_tumor=%.2f)" % (global_step, max_iterations,DSC[0],DSC[1],HD95[0],HD95[1]))
        # print("validation (%d / %d Steps) (dice_liver=%.2f), (dice_tumor=%.2f),(HD95_liver=%.2f), (HD95_tumor=%.2f)" % (global_step, max_iterations,DSC[0],DSC[1],HD95[0],HD95[1]))

        # del val_outputs
        # return DSC,HD95,ASD,val_images, val_labels,val_outputs
        return DSC,HD95,F1 #average_hd,average_f1 
    




