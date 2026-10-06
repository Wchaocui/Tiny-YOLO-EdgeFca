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
set_seed(67)
if __name__ == '__main__':
    #模型训练
    model = YOLO("ultralytics/cfg/cfg/tiny_m_topkedge21.yaml")#tiny_s_topkedge21.yaml
    # model = YOLO('runs/detect/train3/weights/last.pt')
    model.train(resume=True,data="ultralytics/cfg/datasets/SLLFS.yaml", epochs=300,
                batch=32, patience=300,workers=4,device=0,seed=67)
    # model.train(data="ultralytics/cfg/datasets/speedtrain_bbox.yaml", epochs=100, batch=32,
    #             workers=4,patience=100, device=0,amp=True)
    # model.train(data="ultralytics/cfg/datasets/speedplus.yaml", epochs=100, batch=32,
    #             workers=4,patience=100, device=0,amp=True)
    # 模型验证
    # model = YOLO('runs/detect/tiny_m_topkedge21/weights/tiny_m.pt')


    # # 冻结前 10 层
    # freeze_layers = [f"model.{i}" for i in range(16)]
    # for name, param in model.model.named_parameters():
    #     if any(layer in name for layer in freeze_layers):
    #         param.requires_grad = False

    model.val(data="ultralytics/cfg/datasets/SLLFS.yaml", **{'split': 'test'}, device='cpu')
    # model.val(data="ultralytics/cfg/datasets/speedplus.yaml", **{'split': 'test'}, device='cpu')
    # model.val(data="ultralytics/cfg/datasets/speedtrain_bbox.yaml", **{'split': 'test'}, device='cpu')



