from ultralytics import YOLO
from ultralytics import RTDETR
import torch
import random
def set_seed(seed):
    random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # 确保PyTorch使用的是确定性算法
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
# 设置固定的随机种子

    # 模型训练
model = YOLO('ultralytics/cfg/models/v10/yolov10n_edge.yaml')
# model = YOLO('runs/detect/train2/weights/best.pt')
# model = YOLO('runs/detect/yolov8/weights/best.pt')
# model = RTDETR('ultralytics/cfg/models/rt-detr/rtdetr-r18.yaml')
# model = RTDETR('runs/detect/train/weights/best.pt')
model.train(data="ultralytics/cfg/datasets/SLLFS.yaml", epochs=300, batch=32,patience=300, workers=4,device=0,amp=True)
# model.train(data="ultralytics/cfg/datasets/speedplus.yaml", epochs=100, batch=32,patience=100, workers=4,device=0,amp=True)

    # #模型验证

model.val(source="ultralytics/cfg/datasets/SLLFS.yaml",**{'split':'test'},device='cpu')
# model.val(source="ultralytics/cfg/datasets/speedplus.yaml",**{'split':'test'},device='cpu')

    # #模型推理
    # model = YOLO('runs/detect/train/weights/best.pt')
    # model.predict(source="/home/s0433/local/data/exdark_yolo1"
    #                      "exdarkimages/test",**{'save':True})