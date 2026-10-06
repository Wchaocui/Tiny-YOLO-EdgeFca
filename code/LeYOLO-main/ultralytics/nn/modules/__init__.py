# Ultralytics YOLO 🚀, AGPL-3.0 license
"""
Ultralytics modules.

Example:
    Visualize a module with Netron.
    ```python
    from ultralytics.nn.modules import *
    import torch
    import os

    x = torch.ones(1, 128, 40, 40)
    m = Conv(128, 128)
    f = f'{m._get_name()}.onnx'
    torch.onnx.export(m, x, f)
    os.system(f'onnxsim {f} {f} && open {f}')
    ```
"""
from .block import (
    C1,
    C2,
    C3,
    C3TR,
    DFL,
    SPP,
    SPPF,
    Bottleneck,
    BottleneckCSP,
    C2f,
    C3Ghost,
    C3x,
    GhostBottleneck,
    HGBlock,
    HGStem,
    Proto,
    RepC3,
    ResNetLayer,
    InvertedBottleneck,
    MobileNetV3_BLOCK,Conv_NEXT,ExtraDW,
    C3k2,C2PSA,TOPKSPP,freup,SKAttention,ASPP,CARAFE,
)
from .conv import (
    CBAM,
    ChannelAttention,
    Concat,
    Conv,
    Conv2,
    ConvTranspose,
    DWConv,
    DWConvTranspose2d,
    Focus,
    GhostConv,
    LightConv,
    RepConv,
    ChannelAttention,
    SpatialAttention,
    mn_conv,ConvNeXtBlock,EMA

)
from .head import OBB, Classify, Detect, Pose, RTDETRDecoder, Segment
from .transformer import (
    AIFI,
    MLP,
    DeformableTransformerDecoder,
    DeformableTransformerDecoderLayer,
    LayerNorm2d,
    MLPBlock,
    MSDeformAttn,
    TransformerBlock,
    TransformerEncoderLayer,
    TransformerLayer,
)
from ultralytics.nn.modules.gamma import gamma_trans,gamma_trans3,stem_conv
from ultralytics.nn.modules.FEN import FEN
from ultralytics.nn.modules.Edge_fca import EdgeFcaV2
from ultralytics.nn.modules.Edge_fca_adp import EdgeFcaV2_Adaptive
from ultralytics.nn.modules.HWB import HWB,CALayer,SALayer
from ultralytics.nn.modules.fcanet import MultiSpectralAttentionLayer as fcanet_layer
from ultralytics.nn.modules.fcanet import *
from ultralytics.nn.modules.smoothblock import *
from ultralytics.nn.modules.penet import PENet
from ultralytics.nn.modules.ii_block import IIBlock
__all__ = (
    "Conv",
    "Conv2",
    "LightConv",
    "RepConv",
    "DWConv",
    "DWConvTranspose2d",
    "ConvTranspose",
    "Focus",
    "GhostConv",
    "ChannelAttention",
    "SpatialAttention",
    "CBAM",
    "Concat",
    "TransformerLayer",
    "TransformerBlock",
    "MLPBlock",
    "LayerNorm2d",
    "DFL",
    "HGBlock",
    "HGStem",
    "SPP",
    "SPPF",
    "C1",
    "C2",
    "C3",
    "C2f",
    "C3x",
    "C3TR",
    "C3Ghost",
    "GhostBottleneck",
    "Bottleneck",
    "BottleneckCSP",
    "Proto",
    "Detect",
    "Segment",
    "Pose",
    "Classify",
    "TransformerEncoderLayer",
    "RepC3",
    "RTDETRDecoder",
    "AIFI",
    "DeformableTransformerDecoder",
    "DeformableTransformerDecoderLayer",
    "MSDeformAttn",
    "MLP",
    "ResNetLayer",
    "OBB",
    "gamma_trans","gamma_trans3","stem_conv",
    "C3k2","C2PSA",
    "Conv_NEXT","ExtraDW","TOPKSPP",
    "FEN","freup",
    "HWB","EdgeFcaLayer","EMSSHead","EnhancedEdgeFcaLayer","ConvNeXtBlock",#"fcanet_layer","PENet",
    "SKAttention","ASPP","EMA","CARAFE",'IIBlock',"EdgeFcaV2","EdgeFcaV2_Adaptive"

)
