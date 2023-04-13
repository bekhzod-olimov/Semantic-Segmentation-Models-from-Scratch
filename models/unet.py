# Import libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

class UNetBlock(nn.Module):
    
    """
    
    This class gets several arguments and formulates a convolution block of UNet model.
    
    Arguments:
    
        in_chs   - number of channels of the input volume, int;
        out_chs  - number of channels of the output volume, int;
        ks       - kernel size of the convolution operation, int;
        p        - padding value for the convolution operation, int.
        
    Output:
    
        out      - output volume from a convolution block of UNet, tensor.
    
    """

    def __init__(self, in_chs: int, out_chs: int, ks: int = 3, p: int = 1):
        super().__init__()
        
        # Get kernel size and padding value
        self.ks, self.p = ks, p
        
        # Initialize the first and the second convolution blocks
        self.block_1 = self.get_conv_block(in_chs = in_chs, out_chs = out_chs)
        self.block_2 = self.get_conv_block(in_chs = out_chs, out_chs = out_chs)

    def get_conv_block(self, in_chs, out_chs):
        
        return nn.Sequential(nn.Conv2d(in_channels = in_chs, out_channels = out_chs, kernel_size = self.ks, padding = self.p)
                             nn.BatchNorm2d(out_chs), 
                             nn.ReLU(inplace = True))
    
    def forward(self, x): return self.block_2(self.block_1(x))


class DownSampling(nn.Module):
    
    def __init__(self, in_chs, out_chs):
        super().__init__()
        
        self.downsample_block = nn.Sequential(  nn.MaxPool2d(2), UNetBlock(in_chs, out_chs) )

    def forward(self, x): return self.downsample_block(x)


class UpSampling(nn.Module):

    def __init__(self, in_chs, out_chs, mode, upsample=None):
        super().__init__()
        
        if mode in ['bilinear', 'nearest']: 
            upsample = True
            up_mode = mode
        self.upsample = nn.Upsample(scale_factor=2, mode=up_mode) if upsample else nn.ConvTranspose2d(in_chs, in_chs // 2, kernel_size=2, stride=2)
        self.conv = UNetBlock(in_chs, out_chs)

    def forward(self, inp1, inp2):
        
        inp1 = self.upsample(inp1)
        pad_y = inp2.size()[2] - inp1.size()[2]
        pad_x = inp2.size()[3] - inp1.size()[3]
        inp1 = F.pad(inp1, [pad_x // 2, pad_x - pad_x // 2, pad_y // 2, pad_y - pad_y // 2])
        
        return self.conv(torch.cat([inp2, inp1], dim=1))

class FinalConv(nn.Module):
    def __init__(self, in_chs, out_chs):
        super().__init__()
        
        self.conv = nn.Conv2d(in_chs, out_chs, kernel_size = 1)

    def forward(self, inp): return self.conv(inp)
    
# class UNet(nn.Module):
    
#     def __init__(self, in_chs, n_cls, out_chs, depth, up_method):
#         super().__init__()
        
#         assert up_method in ['bilinear', 'nearest', 'tr_conv'], "Please choose a proper method for upsampling"
#         self.depth = depth
#         self.init_block = UNetBlock(in_chs, out_chs)
#         factor = 2 if up_method in ['bilinear', 'nearest'] else 1 
#         self.in_chs, self.n_cls, self.depth, = in_chs, n_cls, depth
#         self.init_block = UNetBlock(in_chs, out_chs)
        
#         self.enc_block_1 = DownSampling(out_chs, out_chs * 2)
#         self.enc_block_2 = DownSampling(out_chs * 2, out_chs * 4)
#         self.enc_block_3 = DownSampling(out_chs * 4, out_chs * 8)
#         self.enc_block_4 = DownSampling(out_chs * 8, out_chs * 16 // factor)
        
        
#         self.dec_block_1 = UpSampling(out_chs * 16, out_chs * 8 // factor, up_method) 
#         self.dec_block_2 = UpSampling(out_chs * 8, out_chs * 4 // factor, up_method) 
#         self.dec_block_3 = UpSampling(out_chs * 4, out_chs * 2 // factor, up_method)
#         self.dec_block_4 = UpSampling(out_chs * 2, (out_chs // factor) * 2 if up_method in ['bilinear', 'nearest'] else (out_chs // factor), up_method)
#         self.final_conv = FinalConv(out_chs, n_cls)
        
#     def forward(self, inp):
        
#         init_conv = self.init_block(inp)
        
#         enc_1 = self.enc_block_1(init_conv)
#         enc_2 = self.enc_block_2(enc_1)
#         enc_3 = self.enc_block_3(enc_2)
#         enc_4 = self.enc_block_4(enc_3)
        
#         dec_1 = self.dec_block_1(enc_4, enc_3)
#         dec_2 = self.dec_block_2(dec_1, enc_2)
#         dec_3 = self.dec_block_3(dec_2, enc_1)
#         dec_4 = self.dec_block_4(dec_3, init_conv)
        
#         return self.final_conv(dec_4)


class UNet(nn.Module):
    
    def __init__(self, in_chs, n_cls, out_chs, depth, up_method):
        super().__init__()
        
        assert up_method in ['bilinear', 'nearest', 'tr_conv'], "Please choose a proper method for upsampling"
        self.depth = depth
        self.init_block = UNetBlock(in_chs, out_chs)
        factor = 2 if up_method in ['bilinear', 'nearest'] else 1 
        
        encoder, decoder = [], []
        
        for idx, (enc, dec) in enumerate(zip(range(depth), reversed(range(depth)))):
            
            enc_in_chs = out_chs * 2 ** enc # 64, 128; 128,256; 256,512; 512,512
            
            enc_out_chs = 2 * out_chs * 2 ** (enc - 1) if (idx == (depth - 1) and up_method in ['bilinear', 'nearest']) else 2 * out_chs * 2 ** enc
                
            encoder += [DownSampling(enc_in_chs, int(enc_out_chs))]
            dec_in_chs = 2 * out_chs * 2 ** dec if idx == 0 else dec_out_chs
            dec_out_chs = out_chs * 2 ** dec if idx != (depth - 1) else factor * out_chs * 2 ** dec
            decoder += [UpSampling(dec_in_chs, dec_out_chs // factor, up_method)]
        
        self.encoder = nn.Sequential(*encoder)
        self.decoder = nn.Sequential(*decoder)
        
        self.final_conv = FinalConv(out_chs, n_cls)
        
    def forward(self, inp):
        
        outputs = [self.init_block(inp)]
        
        for idx, block in enumerate(self.encoder):
            out = outputs[idx] if idx == 0 else encoded
            encoded = block(out)
            outputs.append(encoded)
        
        for idx, block in enumerate(self.decoder):
            encoded = outputs[self.depth - idx - 1]
            decoded = decoded if idx != 0 else outputs[-(idx + 1)]
            decoded = block(decoded, encoded)
            
        out = self.final_conv(decoded)
        
        return out
    
# model = UNet(3,23, 64, 4, 'bilinear')
# print(model)
# a = torch.rand(1,3,704, 1056)
# print(model(a).shape)
# print(sum(p.numel() for p in model.parameters()) / 1e6)
