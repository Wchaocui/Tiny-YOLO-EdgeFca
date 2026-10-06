from ultralytics import YOLO
import torch
import random
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from timm.models.convnext import  ConvNeXtBlock
def set_seed(seed):
    random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # 确保PyTorch使用的是确定性算法
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
# 设置固定的随机种子
set_seed(43)
# 创建YOLO模型实例



if __name__ == '__main__':
    model = YOLO('runs/detect/tiny_m_topkedge21/weights/tiny_m.pt')
    # model = YOLO("ultralytics/cfg/cfg/Tinyolo_snod_s_3fca_C2PSA.yaml")
    # model.val(source="ultralytics/cfg/datasets/slli.yaml", **{'split': 'test'}, device='cuda:0')
    #
    # model.train(data="ultralytics/cfg/datasets/exdark_yolo.yaml", epochs=600, batch=64,
    #             workers=4,device="cuda:0",patience = 0)
    image_folder = '/home/s0433/Downloads/data/spark-2022-stream-1/SLLFS/images/test'
    # image_folder= '/home/s0433/local/projects_py/halfConv/augmentzeroDCE/data/result1/SLLI'
    # image_folder ='/home/s0433/local/data/exdark_yolo/images/test'
    model.predict(source=image_folder,**{'save':True},device='cuda')
    # metrics = model.val(source="ultralytics/cfg/datasets/slli.yaml", **{'split': 'test'}, device='cuda:0')
    # # print(metrics.box.map)
    # print(model)

    # 加载last.pt文件
    # modelAll = torch.load('runs/detect/train4/weights/best.pt', map_location='cpu')
    #
    # # 查看模型结构
    # print(modelAll['model'])
    #
    # # 查看训练参数
    # print(modelAll['train_args'])
