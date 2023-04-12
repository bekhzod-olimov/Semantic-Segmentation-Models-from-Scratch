# Import  libraries
import torch, time
from torch import nn
from torchvision.ops import StochasticDepth
from einops import rearrange
from typing import List, Iterable

class LayerNorm2d(nn.LayerNorm):
    
    """
    
    This function gets a tensor volume, rearranges it, applies LayerNorm and rearranges to its original shape.
    
    """
    
    def forward(self, x):
        
        """
        
        This function does feedforward of the LayerNorm2d class.
        
        Argument:
        
            x    - input volume to the class, tensor.
            
        Output:
            
            x    - rearranged and LayerNorm applied volume, tensor.
        
        """
        
        x = rearrange(x, "b c h w -> b h w c") # (1, 224, 224, 3)
        x = super().forward(x) # (1, 224, 224, 3)
        x = rearrange(x, "b h w c -> b c h w") # (1, 3, 224, 224)
        
        return x
    
class OverlapPatchMerging(nn.Sequential):
    
    """
    
    This class gets several arguments and returns overlap patch merging function.
    
    Arguments:
    
        in_channels  - number of channels of the input volume, int;
        out_channels - number of channels of the output volume, int;
        overlap_size - size of the overlap operation, int.
        
    Output:
    
        overlap patch merging function.
        
    """
    
    def __init__(self, in_channels: int, out_channels: int, patch_size: int, overlap_size: int):
        super().__init__(
            nn.Conv2d(
                in_channels,              # 3
                out_channels,             # 64 
                kernel_size = patch_size, # 16
                stride = overlap_size,    # 16
                padding = patch_size // 2,# 8
                bias = False
            ),
            LayerNorm2d(out_channels)
        )
        
class EfficientMultiHeadAttention(nn.Module):
    
    """
    
    This class gets several arguments and does multi-head attention layer in a more efficient way.
    
    Arguments:
    
        channels        - number of channels, int;
        reduction_ratio - factor to reduce the input volume, int;
        num_heads       - number of heads of the attention layer, int.
        
    Output:
    
        out             - output of the multi head attention layer, tensor.
    
    """
    
    def __init__(self, channels: int, reduction_ratio: int = 1, num_heads: int = 8):
        super().__init__()
        self.reducer = nn.Sequential(nn.Conv2d(channels, channels, kernel_size = reduction_ratio, stride = reduction_ratio), LayerNorm2d(channels))
        self.att = nn.MultiheadAttention(channels, num_heads = num_heads, batch_first = True)

    def forward(self, x):
        _, _, h, w = x.shape
        reduced_x = self.reducer(x) # (1,16,64,64) -> ( 64 - 8  + 0 ) / 8 + 1 = 8; -> (1,16,8,8)
        # attention needs tensor of shape (batch, sequence_length, channels)
        reduced_x = rearrange(reduced_x, "b c h w -> b (h w) c") # (1,16,8,8) -> (1, (8*8), 16)
        x = rearrange(x, "b c h w -> b (h w) c") # (1,16,64,64) -> (1, (64 * 64), 16)
        # query, key, value input to the attention layer
        # out = self.att(x, reduced_x, reduced_x) # [0](Attention output) -> (1, 4096, 16); [1](Attention output weights) -> (1, 4096, 64)
        out = self.att(x, reduced_x, reduced_x)[0]
        out = rearrange(out, "b (h w) c -> b c h w", h = h, w = w) # (1,16,64,64)
        
        return out

class MixMLP(nn.Sequential):
    
    """
    
    This class initializes mixed MLP layer and returns it.
    
    Arguments:
    
        channels  - channels of the input volume, int;
        expansion - a factor to upsample, int.
        
    Output:
    
        mixed MLP layer.
    
    """
    
    def __init__(self, channels: int, expansion: int = 4):
        super().__init__(
            # dense layer
            nn.Conv2d(channels, channels, kernel_size=1),
            # depth wise conv
            nn.Conv2d(
                channels,
                channels * expansion,
                kernel_size = 3,
                groups = channels,
                padding = 1,
            ),
            nn.GELU(),
            # dense layer
            nn.Conv2d(channels * expansion, channels, kernel_size=1),
        )
        

class ResidualAdd(nn.Module):
    
    """
    
    This class does residual add for skip connections.
    
    Argument:
    
        x - input volume, tensor;
    
    Output:
        
        x - output volume from residual block, tensor.
    
    """
    
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        
        out = self.fn(x, **kwargs)
        
        return x + out

class SegFormerEncoderBlock(nn.Sequential):
    
    """
    
    This class creates an encoder block of the SegFormer network.
    
    Arguments:
    
        channels        - number of input channels, int;
        reduction_ratio - factor to reduce an input volume, int;
        num_heads       - number of attention heads, int;
        mlp_expansion   - factor to increase MixMLP, int;
        drop_path_prob  - dropout probability, float.
        
    Output:
    
        Encoder Block, torch sequential object.
    
    """
    
    def __init__(
        self,
        channels: int,
        reduction_ratio: int = 1,
        num_heads: int = 8,
        mlp_expansion: int = 4,
        drop_path_prob: float = .0
    ):
        super().__init__(
            ResidualAdd(
                nn.Sequential(
                    LayerNorm2d(channels),
                    EfficientMultiHeadAttention(channels, reduction_ratio, num_heads),
                )
            ),
            ResidualAdd(
                nn.Sequential(
                    LayerNorm2d(channels),
                    MixMLP(channels, expansion = mlp_expansion),
                    StochasticDepth(p = drop_path_prob, mode = "batch")
                )
            ),
        )


class SegFormerEncoderStage(nn.Sequential):
    
    """
    
    This class gets several arguments and creates an encoder model of the Segformer network.
    
    Arguments:
    
        in_channels     - number of channels of an input volume, int;
        out_channels    - number of channels of an output volume, int;
        patch_size      - size of a patch, int;
        overlap_size    - size for overlap matching, int;
        drop_probs      - probabilities for dropout, list -> int;
        depth           - an encoder model depth, int;
        reduction_ratio - a factor to reduce dimensions of an input volume, int;
        num_heads       - number of attention heads, int;
        mlp_expansion   - factor to increase MixMLP, int;
        
    Output:
    
        Encoder Model, torch sequential object.
    
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        patch_size: int,
        overlap_size: int,
        drop_probs: List[int],
        depth: int = 2,
        reduction_ratio: int = 1,
        num_heads: int = 8,
        mlp_expansion: int = 4,
    ):
        super().__init__()
        self.overlap_patch_merge = OverlapPatchMerging(in_channels, out_channels, patch_size, overlap_size)
        
        self.blocks = nn.Sequential(
                *[
                    SegFormerEncoderBlock(
                        out_channels, reduction_ratio, num_heads, mlp_expansion, drop_probs[i]
                    )
                    for i in range(depth)
                ]
            )
        self.norm = LayerNorm2d(out_channels)

def chunks(data: Iterable, sizes: List[int]):
    
    """
    
    This function gets iterable with numbers and sizes list and returns slices of the list based on the sizes information.
    
    Arguments:
    
        data   - data to be chunked, iterable object;
        sizes  - size information to be chunked, list -> int.
        
    Output:
    
        chunk  - chunk of the data, generator object.
        
    """
    curr = 0
    for size in sizes:
        chunk = data[curr: curr + size]
        curr += size
        yield chunk

class SegFormerEncoder(nn.Module):
    
    """
    
    This class creates an encoder network of SegFormer.
    
    Arguments:
    
        in_channels      - number of channels of an input volume, int;
        widths           - width of the encoder network, list -> int;
        depths           - depth of the encoder network, list -> int;
        all_num_heads    - total number of attention heads, list -> int;
        patch_sizes      - total sizes of a patch, list -> int;
        overlap_sizes    - total sizes for overlap matching, list -> int;
        reduction_ratios - total factors to reduce dimensions of an input volume, list -> int;
        mlp_expansions   - total factors to increase MixMLP, list -> int;
        drop_prob        - probability value for dropout, float;
        
    Output:
    
        An Encoder Network, torch sequential object.
        
    """
    
    def __init__(
        self,
        in_channels: int,
        widths: List[int],
        depths: List[int],
        all_num_heads: List[int],
        patch_sizes: List[int],
        overlap_sizes: List[int],
        reduction_ratios: List[int],
        mlp_expansions: List[int],
        drop_prob: float = .0
    ):
        super().__init__()
        
        # Dropout probabilites for based on the depth of the network
        drop_probs =  [x.item() for x in torch.linspace(0, drop_prob, sum(depths))]
        
        # Initialize stages of the encoder stage
        self.stages = nn.ModuleList(
            [
                SegFormerEncoderStage(*args)
                for args in zip(
                    [in_channels, *widths],
                    widths,
                    patch_sizes,
                    overlap_sizes,
                    chunks(drop_probs, sizes = depths),
                    depths,
                    reduction_ratios,
                    all_num_heads,
                    mlp_expansions
                )
            ]
        )
        
    def forward(self, x):
        
        # Initialize list for the features of each stage of the encoder network
        features = []
        
        # Go through every stage of the encoder network
        for stage in self.stages:
            x = stage(x)
            features.append(x)
        return features   
    
class SegFormerSegmentationHead(nn.Module):
    
    """
    
    This class initializes Segmentation Head Network and returns its output volume.
    
    Arguments:
    
        channels     - number of channels of the input volume, int;
        num_classes  - number of classes in the dataset, int;
        num_features - number of features in the segmentation head network, int.
        
    Output:
    
        x            - output volume from the Segmentation Head network, tensor.
        
    """
    
    def __init__(self, channels: int, num_classes: int, num_features: int = 4):
        super().__init__()
        self.sequence = nn.Sequential(
            nn.Conv2d(channels * num_features, channels, kernel_size = 1, bias = False),
            nn.ReLU(), 
            nn.BatchNorm2d(channels) 
        )
        self.out_layer = nn.Conv2d(channels, num_classes, kernel_size = 1)

    def forward(self, features):
        
        """
        
        This function gets input features and passes them through Segmentation Head Network.
        
        Argument:
        
            features    - input features volume, tensor;
        
        Output:
             
            x           - output volume from the Segmentation Head Network, tensor.
        
        """
        
        # Concatenate features using the dimension #1
        x = torch.cat(features, dim = 1)
        
        # Pass through sequence of layers
        x = self.sequence(x)
        
        # Return output volume
        return self.out_layer(x)
    
class SegFormerDecoderBlock(nn.Sequential):
    
    
    """
    
    This class initializes a block of the Decoder Network of SegFormer.
    
    Arguments:
    
        in_channels  - number of channels of the input volume to a convolution layer, int;
        out_channels - number of channels of the output volume from the convolution layer, int;
        scale_factor - a factor used to upsample the input volume, int.       
    
    """
    
    def __init__(self, in_channels: int, out_channels: int, scale_factor: int = 2):
        super().__init__(nn.UpsamplingBilinear2d(scale_factor = scale_factor), nn.Conv2d(in_channels, out_channels, kernel_size = 1))
        
class SegFormerDecoder(nn.Module):
    
    """
    
    This class initializes a Decoder network of SegFormer.
    
    Arguments:
    
        out_channels  - number of channels of the output volume from the convolution layer, int;
        widths        - values for the width in the Decoder Network, list -> int;
        scale_factors - factor values used to upsample the input volumes, list -> int.       
        
    Output:
    
        new_features  - features from the decoder of the SegFormer model, list.
    
    """
    
    def __init__(self, out_channels: int, widths: List[int], scale_factors: List[int]):
        super().__init__()
        self.stages = nn.ModuleList(
            [
                SegFormerDecoderBlock(in_channels, out_channels, scale_factor) for in_channels, scale_factor in zip(widths, scale_factors)
            ]
        )
    
    def forward(self, features):
        
        """
        
        This function gets input features and passes them through decoder network of the SegFormer and returns new decoded features.
        
        Argument:
        
            features     - input features, list;
        
        Output:
        
            new_features  - features from the decoder of the SegFormer model, list.
        
        """
        
        new_features = []
        
        # Go through every feature and stage
        for feature, stage in zip(features,self.stages):
            
            # Decode the features
            x = stage(feature)
            # Add the decoded features to the list
            new_features.append(x)
            
        return new_features
    
class SegFormer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        widths: List[int],
        depths: List[int],
        all_num_heads: List[int],
        patch_sizes: List[int],
        overlap_sizes: List[int],
        reduction_ratios: List[int],
        mlp_expansions: List[int],
        decoder_channels: int,
        scale_factors: List[int],
        num_classes: int,
        drop_prob: float = 0.0,
    ):

        super().__init__()
        self.encoder = SegFormerEncoder(
            in_channels,
            widths,
            depths,
            all_num_heads,
            patch_sizes,
            overlap_sizes,
            reduction_ratios,
            mlp_expansions,
            drop_prob,
        )
        self.decoder = SegFormerDecoder(decoder_channels, widths[::-1], scale_factors)
        self.head = SegFormerSegmentationHead(
            decoder_channels, num_classes, num_features=len(widths)
        )

    def forward(self, x):
        features = self.encoder(x)
        features = self.decoder(features[::-1])
        segmentation = self.head(features)
        return segmentation
    
# segformer = SegFormer(
#     in_channels=3,
#     widths=[64, 128, 256, 512],
#     depths=[3, 4, 6, 3],
#     all_num_heads=[1, 2, 4, 8],
#     patch_sizes=[7, 3, 3, 3],
#     overlap_sizes=[4, 2, 2, 2],
#     reduction_ratios=[8, 4, 2, 1],
#     mlp_expansions=[4, 4, 4, 4],
#     decoder_channels=256,
#     scale_factors=[8, 4, 2, 1],
#     num_classes=23,
# )

# segmentation = segformer(torch.randn(1,3,704, 1056))
# print(segmentation.shape)
