from ultralytics import YOLO
from ultralytics import RTDETR
import torch
import random
# ✅ 修正
def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(67)
# 设置固定的随机种子

    # 模型训练
# model = YOLO('ultralytics/cfg/models/v9/yolov9t_edgefca.yaml')
# model = YOLO('ultralytics/cfg/models/v10/yolov10n.yaml')
# model = YOLO('ultralytics/cfg/models/11/yolo11.yaml')
model = YOLO('ultralytics/cfg/models/12/yolo12n.yaml')
# model = YOLO('ultralytics/cfg/models/v3/EMV-yolov3-tiny.yaml')
# model = YOLO('/hy-tmp/ultralytics-main/runs/detect/train/weights/best.pt')
# model = YOLO('ultralytics/cfg/models/v10/yolov10n.yaml')
# model = RTDETR('ultralytics/cfg/models/rt-detr/rtdetr-r18.yaml')
# model = RTDETR('runs/detect/SPEED+rtdetr-r34/weights/best.pt')
model.train(data="ultralytics/cfg/datasets/SLLFS.yaml", epochs=300, batch=32,nbs=32,patience=300, workers=4,device=0,amp=True,seed=67)
# model.train(data="ultralytics/cfg/datasets/speedplus.yaml", epochs=100, batch=32,nbs=32,
#             patie nce=100, workers=4,device=0,amp=True)
# model.train(data="ultralytics/cfg/datasets/speedtrain_bbox.yaml", epochs=100, batch=32,nbs=32,
            # patience=100, workers=4,device=0,amp=True)
# model.train(resume=True)
    # #模型验证
model.val(data="ultralytics/cfg/datasets/SLLFS.yaml",**{'split':'test'},device='cpu')
# model.val(data="ultralytics/cfg/datasets/speedplus.yaml",**{'split':'test'},device='cpu')
# model.val(data="ultralytics/cfg/datasets/speedtrain_bbox.yaml",**{'split':'test'},device='cpu')

    # #模型推理
    # model = YOLO('runs/detect/train/weights/best.pt')
    # model.predict(source="/home/s0433/local/data/exdark_yolo1"
    #                      "exdarkimages/test",**{'save':True})