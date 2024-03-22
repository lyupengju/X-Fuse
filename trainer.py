
import os
from tqdm import tqdm
import pandas as pd
import torch
import torch.nn.functional as F
from full_validationer import validationer

from collections import OrderedDict


def trainer(epoch_num, global_step,alpha,train_loader,val_loader, net, DiceCEloss,scaler,optimizer,max_iterations,max_epochs,writer,
                    snapshot_path,post_label,post_pred,Dice_coefficient,HD_95,F1_score,best_dice, best_dice_global_step,corresponding_HD,corresponding_f1):

    
        
    net.train()
    step = 0
    train_loss_per_epoch=0
    
    '''start training'''
    epoch_iterator_train = tqdm(train_loader, desc="Training", dynamic_ncols=True)
    for i_batch, sampled_batch in enumerate(epoch_iterator_train):
        global_step += 1
        step += 1
        
        
        volume_batch, label_batch = sampled_batch['image'].cuda(), sampled_batch['label'].cuda()
        
        with torch.cuda.amp.autocast():
            outputs = net(volume_batch)
            
            DiceCE_per_step = DiceCEloss(outputs,label_batch)
              
            total_loss_per_step = alpha*(DiceCE_per_step) 
            writer.add_scalar('loss/DiceCE_per_step',DiceCE_per_step.item(),global_step)

        scaler.scale(total_loss_per_step).backward()
        train_loss_per_epoch += total_loss_per_step.item()
        scaler.unscale_(optimizer)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        epoch_iterator_train.set_description( "Training (%d / %d Steps) (loss=%2.5f)" % (global_step, max_iterations, total_loss_per_step))
    train_loss_per_epoch /= step
    writer.add_scalar('loss/train_loss_per_epoch', train_loss_per_epoch, epoch_num)# save to tensorboard 保存每个epoch的loss
    torch.cuda.empty_cache()

    if  epoch_num % ( 15 if epoch_num<round(max_epochs/1.5) else 5) == 0 or global_step == max_iterations:

    
        DSC,HD95,F1 = validationer(val_loader,net,global_step,max_iterations,
                                    post_label,post_pred,Dice_coefficient,HD_95,F1_score)
        

        writer.add_scalar('metrics/DSC', DSC,global_step)
        writer.add_scalar('metrics/HD95', HD95,global_step)

        

        if DSC > best_dice or ((DSC == best_dice) and  (HD95 <= corresponding_HD)) :
                best_dice = DSC
                best_dice_global_step = global_step
                corresponding_HD =HD95
                corresponding_f1 = F1 
                save_best_mode_path = os.path.join(snapshot_path, "best_metric_model_step_"+ str(global_step)  + '.pth')
                torch.save(net.state_dict(), save_best_mode_path)
                
                print("saved new metric model")

        print("current step: {}, current mean dice: {}, mean_HD: {}, best mean dice: {:.2f} at step {} ,its HD: {:.2f},f1 score:{:.2f}"
                        .format(global_step, DSC,HD95, best_dice, best_dice_global_step,corresponding_HD,corresponding_f1))

    torch.cuda.empty_cache()
   
    return global_step, best_dice, best_dice_global_step,corresponding_HD,corresponding_f1