import torch
from monai.networks.blocks.dynunet_block import UnetBasicBlock, UnetResBlock, get_conv_layer

import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from timm.models.layers import DropPath, to_2tuple, trunc_normal_,to_3tuple
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg
# from monai.networks.blocks import MLPBlock as Mlp
from monai.networks.blocks import PatchEmbed, UnetOutBlock, UnetrBasicBlock, UnetrUpBlock
# from ptflops import get_model_complexity_info
# from thop import profile
from typing import Optional, Sequence, Tuple, Type, Union
import os
from monai.utils import ensure_tuple_rep, look_up_option, optional_import
rearrange, _ = optional_import("einops", name="rearrange")
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)


    def forward(self, x, H, W ,D):
        x = self.fc1(x)
        x = self.act(x + self.dwconv(x, H, W, D))
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class GlobalFilter(nn.Module):
    def __init__(self, dim, size = 48):
        super().__init__()
        
        size = to_2tuple(size)
        self.size = size[0]
        self.filter_size = size[1] // 2 + 1
        self.complex_weight = nn.Parameter(torch.randn(self.size, self.size, self.filter_size, dim, 2) * 0.02)
        # print()
    def forward(self, x, H, W, D):
        B, N, C = x.shape
        x = x.reshape(B, H, W, D, C)#.permute(4, 0, 5, 1, 2, 3)

        # B, H, W, D, C = x.shape
        x = torch.fft.rfftn(x, dim=(1, 2, 3), norm='ortho')
        weight = torch.view_as_complex(self.complex_weight)
        # y=torch.view_as_real(x.contiguous()).reshape(B,self.size,self.size,self.filter_size,-1)

        # print('x', x.shape)
        # print('w',weight.shape)
        x = x * weight
        x = torch.fft.irfftn(x, s=(H, W, D), dim=(1, 2, 3), norm='ortho')
        x = x.reshape(B, N, C)

        return x
    
class Block(nn.Module):

    def __init__(self, dim,mlp_ratio=2,act = nn.GELU(), drop_path=0., layer_scale_init_value=1e-6,size = 48):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.fft = GlobalFilter(size = size, dim = dim)
        self.norm2 = nn.LayerNorm(dim)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        # self.mlp_channels = Mlp(hidden_size=dim, mlp_dim=dim*mlp_ratio)
        self.mlp_channels = Mlp(in_features=dim, hidden_features=dim*mlp_ratio)

        

    def forward(self, x, H, W, D):
        
        x = x + self.drop_path(self.fft(self.norm1(x), H, W, D))
        x = x + self.drop_path(self.mlp_channels(self.norm2(x), H, W, D))
       
        return x


class OverlapPatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=96, patch_size=3, stride=2, in_chans=3, embed_dim=768):
        super().__init__()
        patch_size = to_3tuple(patch_size)
        img_size = to_3tuple(img_size)

        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2,patch_size[2] // 2))
        self.norm = nn.LayerNorm(embed_dim)

        # self.apply(self._init_weights)


    def forward(self, x):
        x = self.proj(x)
        _, _, H, W, D = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)

        return x, H, W ,D


class Head(nn.Module):
    def __init__(self, in_chans,head_conv, dim):
        super(Head, self).__init__()
        # stem = [nn.Conv3d( in_chans, dim, head_conv, 2, padding=3 if head_conv==7 else 1, bias=False), nn.InstanceNorm3d(dim), nn.ReLU(True)]
        # stem.append(nn.Conv3d(dim, dim, kernel_size=2, stride=2))
        stem = [nn.Conv3d(in_chans, int(dim/2), head_conv, 2,  padding=1, bias=False), nn.InstanceNorm3d(int(dim/2)), nn.ReLU(True)]
        stem.append(nn.Conv3d(int(dim/2), dim, kernel_size=3, stride=1,padding=1,))
        self.conv = nn.Sequential(*stem)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        x = self.conv(x)
        _, _, H, W, D = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, H, W,D


class GFnet(nn.Module):
    def __init__(self, img_size=96, in_chans=1, num_classes=15, embed_dims=[64, 128, 256, 512],size = [48, 24, 12, 6],
                  mlp_ratios=[8, 6, 4, 2], 
                  norm_layer=partial(nn.LayerNorm, eps=1e-6),
                 norm_name: Union[Tuple, str] = "instance", depths=[2, 2, 8, 1],  num_stages=4, head_conv=3, expand_ratio=2, **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths
        self.num_stages = num_stages


        for i in range(num_stages):
            if i ==0:
                patch_embed = Head(in_chans,head_conv, embed_dims[i])#
            else:
                patch_embed = OverlapPatchEmbed(img_size=img_size if i == 0 else img_size // (2 ** (i + 1)),
                                            patch_size=3,
                                            stride=2,
                                            in_chans=embed_dims[i - 1],
                                            embed_dim=embed_dims[i])

            block = nn.ModuleList([Block(
                dim=embed_dims[i], mlp_ratio=mlp_ratios[i],size = size[i])
                for j in range(depths[i])])
            
            norm = norm_layer(embed_dims[i])

            setattr(self, f"patch_embed{i + 1}", patch_embed)
            setattr(self, f"block{i + 1}", block)
            setattr(self, f"norm{i + 1}", norm)

       
        # feature_size = 60
        # #heavy decoder
        # # norm_name = "instance",
        # spatial_dims = 3

        # self.encoder2 = UnetrBasicBlock(
        #     spatial_dims=spatial_dims,
        #     in_channels=feature_size,
        #     out_channels=feature_size,
        #     kernel_size=3,
        #     stride=1,
        #     norm_name=norm_name,
        #     res_block=True,
        # )

        # self.encoder3 = UnetrBasicBlock(
        #     spatial_dims=spatial_dims,
        #     in_channels=2 * feature_size,
        #     out_channels=2 * feature_size,
        #     kernel_size=3,
        #     stride=1,
        #     norm_name=norm_name,
        #     res_block=True,
        # )

        # self.encoder4 = UnetrBasicBlock(
        #     spatial_dims=spatial_dims,
        #     in_channels=4 * feature_size,
        #     out_channels=4 * feature_size,
        #     kernel_size=3,
        #     stride=1,
        #     norm_name=norm_name,
        #     res_block=True,
        # )

        # self.encoder5 = UnetrBasicBlock(
        #     spatial_dims=spatial_dims,
        #     in_channels=8 * feature_size,
        #     out_channels=8 * feature_size,
        #     kernel_size=3,
        #     stride=1,
        #     norm_name=norm_name,
        #     res_block=True,
        # )

        # self.decoder4 = UnetrUpBlock(
        #             spatial_dims=spatial_dims,
        #             in_channels=feature_size * 8,
        #             out_channels=feature_size * 4,
        #             kernel_size=3,
        #             upsample_kernel_size=2,
        #             norm_name=norm_name,
        #             res_block=True,
        #         )

        # self.decoder3 = UnetrUpBlock(
        #         spatial_dims=spatial_dims,
        #         in_channels=feature_size * 4,
        #         out_channels=feature_size * 2,
        #         kernel_size=3,
        #         upsample_kernel_size=2,
        #         norm_name=norm_name,
        #         res_block=True,
        #     )
        # self.decoder2 = UnetrUpBlock(
        #         spatial_dims=spatial_dims,
        #         in_channels=feature_size * 2,
        #         out_channels=feature_size,
        #         kernel_size=3,
        #         upsample_kernel_size=2,
        #         norm_name=norm_name,
        #         res_block=True,
        #     )
        # self.decoder1 = UnetrUpBlock(
        #         spatial_dims=spatial_dims,
        #         in_channels=feature_size,
        #         out_channels=feature_size,
        #         kernel_size=3,
        #         upsample_kernel_size=2,
        #         norm_name=norm_name,
        #         res_block=True,
        #     )
        # self.outup = nn.ConvTranspose3d(
        #             feature_size , int(feature_size/2), kernel_size=2, stride=2,)

        # self.out = UnetOutBlock(spatial_dims=spatial_dims, in_channels=int(feature_size/2), out_channels=num_classes)

    def forward(self, x):
        origin_input = x
        B = x.shape[0]
        outs=[]
        for i in range(self.num_stages):
            patch_embed = getattr(self, f"patch_embed{i + 1}")
            block = getattr(self, f"block{i + 1}")
            norm = getattr(self, f"norm{i + 1}")

            x, H, W, D = patch_embed(x)
            for blk in block:
                x = blk(x, H, W, D)
            x = norm(x)
            x = x.reshape(B, H, W,D, -1).permute(0, 4, 1, 2,3).contiguous()

            # print(x.shape)

            outs.append(x)

     
        # enc1 = self.encoder2(outs[0])
        # enc2 = self.encoder3(outs[1])
        # enc3 = self.encoder4(outs[2])
        # enc4 = self.encoder5(outs[3])
        # dec2 = self.decoder4(enc4, enc3)
        # dec1 = self.decoder3(dec2, enc2)
        # dec0 = self.decoder2(dec1, enc1)
        # output = self.out(self.outup(dec0))


        # return output
        return outs


 


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv3d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W,D):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W, D)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)

        return x

class UnetrUpBlock_trilinear(nn.Module):
    """
    An upsampling module that can be used for UNETR: "Hatamizadeh et al.,
    UNETR: Transformers for 3D Medical Image Segmentation <https://arxiv.org/abs/2103.10504>"
    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[Sequence[int], int],
        upsample_kernel_size: Union[Sequence[int], int],
        norm_name: Union[Tuple, str],
        res_block: bool = False,
    ) -> None:
        """
        Args:
            spatial_dims: number of spatial dimensions.
            in_channels: number of input channels.
            out_channels: number of output channels.
            kernel_size: convolution kernel size.
            upsample_kernel_size: convolution kernel size for transposed convolution layers.
            norm_name: feature normalization type and arguments.
            res_block: bool argument to determine if residual block is used.

        """

        super().__init__()
        upsample_stride = upsample_kernel_size
        # self.transp_conv = get_conv_layer(
        #     spatial_dims,
        #     in_channels,
        #     out_channels,
        #     kernel_size=upsample_kernel_size,
        #     stride=upsample_stride,
        #     conv_only=True,
        #     is_transposed=True,
        # )
        self.conv1 = get_conv_layer(
            spatial_dims,
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=1,
            dropout=None,
            act=None,
            norm=None,
            conv_only=False,
        )
        if res_block:
            self.conv_block = UnetResBlock(
                spatial_dims,
                out_channels + out_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=1,
                norm_name=norm_name,
            )
        else:
            self.conv_block = UnetBasicBlock(  # type: ignore
                spatial_dims,
                out_channels + out_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=1,
                norm_name=norm_name,
            )

    def forward(self, inp, skip):
        # number of channels for skip should equals to out_channels
        # out = self.transp_conv(inp)
        out = self.conv1(inp)
        out = F.interpolate(out, scale_factor=2, mode="trilinear")

        out = torch.cat((out, skip), dim=1)
        out = self.conv_block(out)
        return out





if __name__ == '__main__':
    import torch
    # net = SMT(img_size=96, in_chans=1, num_classes=2,
    #     embed_dims=[30*2, 60*2, 120*2, 240*2], ca_num_heads=[3, 3, 3, -1], sa_num_heads=[-1, -1, 8, 16], mlp_ratios=[2, 2, 2, 2], 
    #     qkv_bias=True, depths=[2, 2, 4, 2], ca_attentions=[1, 1, 1, 0], head_conv=3, expand_ratio=2)  
    # model.default_cfg = _cfg()

    net = GFnet(img_size=96, in_chans=1, num_classes=2,
        embed_dims=[30*2, 60*2, 120*2, 240*2], mlp_ratios=[2, 2, 2, 2], 
         depths=[2, 2, 2, 2], head_conv=3, expand_ratio=2).cuda()
    model = net.cuda()
    input = torch.rand(1, 1, 96,96, 96).cuda()
    output = model(input)
   
    # print(output.shape)
    print(len(output))

    # print(output.shape)
    # print(model)

    ### thop cal ###
    # input_shape = (1, 3, 384, 384) # 输入的形状
    # input_data = torch.randn(*input_shape)
    # macs, params = profile(model, inputs=(input_data,))
    # print(f"FLOPS: {macs / 1e9:.2f}G")
    # print(f"params: {params / 1e6:.2f}M")

    ### ptflops cal ###
    # flops_count, params_count = get_model_complexity_info(model,(3,224,224), as_strings=True, print_per_layer_stat=False)

    # print('flops: ', flops_count)
    # print('params: ', params_count)