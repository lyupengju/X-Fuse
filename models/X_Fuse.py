
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from timm.models.layers import DropPath, to_2tuple, trunc_normal_,to_3tuple
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg
import math
from monai.networks.blocks import PatchEmbed, UnetOutBlock, UnetrBasicBlock, UnetrUpBlock
# from ptflops import get_model_complexity_info
# from thop import profile
from typing import Optional, Sequence, Tuple, Type, Union
import os
from Global_filter import GFnet,UnetrUpBlock_trilinear
from SAM import SMT



class Xfuse(nn.Module):
    def __init__(self,in_channels = 1,out_channels = 2, norm_name: Union[Tuple, str] = "instance",):
        super().__init__()
        feature_size = 60
        spatial_dims = 3
        self.Spectral_encoder = GFnet(img_size=96, in_chans=1, num_classes=2,embed_dims=[30*2, 60*2, 120*2, 240*2], mlp_ratios=[2, 2, 2, 2],depths=[2, 2, 2, 2], head_conv=3, expand_ratio=2)
        self.Spatial_encoder = SMT(img_size=96, in_chans=1, num_classes=2,embed_dims=[30*2, 60*2, 120*2, 240*2], ca_num_heads=[3, 3, 3, 3], sa_num_heads=[-1, -1, -1, 4], mlp_ratios=[2, 2, 2, 2], 
        qkv_bias=True, depths=[2, 2, 2, 2], ca_attentions=[1, 1, 1, 1], head_conv=3, expand_ratio=2)
        
        
        '''GFnet'''
        self.encoder2_GF = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size,
            out_channels=feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )

        self.encoder3_GF = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=2 * feature_size,
            out_channels=2 * feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )

        self.encoder4_GF = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=4 * feature_size,
            out_channels=4 * feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )

        self.encoder5_GF = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=8 * feature_size,
            out_channels=8 * feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )

        
        self.decoder4_GF = UnetrUpBlock_trilinear(
                    spatial_dims=spatial_dims,
                    in_channels=feature_size * 8,
                    out_channels=feature_size * 4,
                    kernel_size=3,
                    upsample_kernel_size=2,
                    norm_name=norm_name,
                    res_block=True,
                )

        self.decoder3_GF = UnetrUpBlock_trilinear(
                spatial_dims=spatial_dims,
                in_channels=feature_size * 4,
                out_channels=feature_size * 2,
                kernel_size=3,
                upsample_kernel_size=2,
                norm_name=norm_name,
                res_block=True,
            )
        self.decoder2_GF = UnetrUpBlock_trilinear(
                spatial_dims=spatial_dims,
                in_channels=feature_size * 2,
                out_channels=feature_size,
                kernel_size=3,
                upsample_kernel_size=2,
                norm_name=norm_name,
                res_block=True,
            )
        self.decoder1_GF = UnetrUpBlock_trilinear(
                spatial_dims=spatial_dims,
                in_channels=feature_size,
                out_channels=feature_size,
                kernel_size=3,
                upsample_kernel_size=2,
                norm_name=norm_name,
                res_block=True,
            )

        self.outup_GF = nn.ConvTranspose3d(
                    feature_size , int(feature_size/2), kernel_size=2, stride=2,)

        self.out_GF = UnetOutBlock(spatial_dims=spatial_dims, in_channels=int(feature_size/2), out_channels=out_channels)
      
        #bottleneck fusion
        self.encoder_bottenneck = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=16 * feature_size,
            out_channels=8 * feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )

        '''SAM'''
        self.encoder2_SAM = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size,
            out_channels=feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )

        self.encoder3_SAM = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=2 * feature_size,
            out_channels=2 * feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )

        self.encoder4_SAM = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=4 * feature_size,
            out_channels=4 * feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )

        self.encoder5_SAM = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=8 * feature_size,
            out_channels=8 * feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )

        
        self.decoder4_SAM = UnetrUpBlock(
                    spatial_dims=spatial_dims,
                    in_channels=feature_size * 8,
                    out_channels=feature_size * 4,
                    kernel_size=3,
                    upsample_kernel_size=2,
                    norm_name=norm_name,
                    res_block=True,
                )

        self.decoder3_SAM = UnetrUpBlock(
                spatial_dims=spatial_dims,
                in_channels=feature_size * 4,
                out_channels=feature_size * 2,
                kernel_size=3,
                upsample_kernel_size=2,
                norm_name=norm_name,
                res_block=True,
            )
        self.decoder2_SAM = UnetrUpBlock(
                spatial_dims=spatial_dims,
                in_channels=feature_size * 2,
                out_channels=feature_size,
                kernel_size=3,
                upsample_kernel_size=2,
                norm_name=norm_name,
                res_block=True,
            )
        self.decoder1_SAM = UnetrUpBlock(
                spatial_dims=spatial_dims,
                in_channels=feature_size,
                out_channels=feature_size,
                kernel_size=3,
                upsample_kernel_size=2,
                norm_name=norm_name,
                res_block=True,
            )

        self.outup_SAM = nn.ConvTranspose3d(
                    feature_size , int(feature_size/2), kernel_size=2, stride=2,)

        self.out_SAM = UnetOutBlock(spatial_dims=spatial_dims, in_channels=int(feature_size/2), out_channels=out_channels)
      
    def forward(self, x):
        outs_Spectral = self.Spectral_encoder(x)
        outs_Spatial = self.Spatial_encoder(x)
        # print('en')
        bottleneck = self.encoder_bottenneck(torch.cat((outs_Spectral[3], outs_Spatial[3]), dim=1))
        # print(bottleneck.shape)

        #'spatial decoder'
        enc1 = self.encoder2_SAM(outs_Spectral[0])
        enc2 = self.encoder3_SAM(outs_Spectral[1])
        enc3 = self.encoder4_SAM(outs_Spectral[2])
        enc4 = self.encoder5_SAM(bottleneck)
        dec2 = self.decoder4_SAM(enc4, enc3)
        dec1 = self.decoder3_SAM(dec2, enc2)
        dec0 = self.decoder2_SAM(dec1, enc1)
        output_SAW = self.out_SAM(self.outup_SAM(dec0))

        'spectral decoder'
        enc1 = self.encoder2_GF(outs_Spatial[0])
        enc2 = self.encoder3_GF(outs_Spatial[1])
        enc3 = self.encoder4_GF(outs_Spatial[2])
        enc4 = self.encoder5_GF(bottleneck)
        dec2 = self.decoder4_GF(enc4, enc3)
        dec1 = self.decoder3_GF(dec2, enc2)
        dec0 = self.decoder2_GF(dec1, enc1)
        output_GF = self.out_GF(self.outup_GF(dec0))

        out = (output_GF+output_SAW)/2
        # out = output_SAW
        # print('out')
        return out

if __name__ == '__main__':
    from thop import profile
    import time
    from monai.inferers import sliding_window_inference

    # with torch.no_grad():

    #     net = Xfuse(in_channels = 1,out_channels = 2)
    #     model = net.cuda()
    #     input = torch.rand(1, 1, 96,96, 96).cuda()
    #     output = model(input)
    #     print(output.shape)

    #     flops, params = profile(model,inputs=(input,))
    #     print('FLOPs = ' + str(flops/1000**3) + 'G')
    #     print('Params = ' + str(params/1000**2) + 'M')
    start_time  = time.time()

    with torch.no_grad():
        input = torch.rand(1, 1, 96,96, 96)

        input = torch.rand(1, 1, 512,512, 100).cuda()
        model = Xfuse(in_channels = 1,out_channels = 2).cuda()

        with torch.cuda.amp.autocast():

            test_data = sliding_window_inference(
                input,(96,96,96), 1, model, overlap=0.5,sw_device='cuda',device='cpu')
        
    t2 = time.time()
    print("total推理时间(s)：{}".format(t2 - start_time))        