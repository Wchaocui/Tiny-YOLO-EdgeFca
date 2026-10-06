from ultralytics import YOLO
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
set_seed(5)
if __name__ == '__main__':
    #模型训练
    model = YOLO("ultralytics/cfg/cfg/tiny_s_topkedge21.yaml")
    # model = YOLO('runs/detect/5seed-/weights/best.pt')
    model.train(data="ultralytics/cfg/datasets/SLLFS.yaml", epochs=300, batch=32,
                workers=4,patience=300, device=0,seed=5)
    # model.train(data="ultralytics/cfg/datasets/speedplus.yaml", epochs=100, batch=32,
    #             workers=4,patience=100, device=0,amp=True)
    # model.train(data="ultralytics/cfg/datasets/speedtrain_bbox.yaml", epochs=100, batch=32,
                # workers=4,patience=100, device=0,amp=True)
    # 有预训练


    #模型验证
    # model = YOLO('runs/detect/train4/weights/last.pt')
    # 冻结前 10 层
    # freeze_layers = [f"model.{i}" for i in range(18)]
    # for name, param in model.model.named_parameters():
    #     if any(layer in name for layer in freeze_layers):
    #         param.requires_grad = False
    model.val(data="ultralytics/cfg/datasets/SLLFS.yaml", **{'split': 'test'}, device='cpu')
    # model.val(data="ultralytics/cfg/datasets/speedplus.yaml", **{'split': 'test'}, device='cpu')
    # model.val(data="ultralytics/cfg/datasets/speedtrain_bbox.yaml", **{'split': 'test'}, device='cpu')




