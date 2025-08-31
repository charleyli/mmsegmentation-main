import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.optim import SGD
from torch.utils.data import DataLoader

from mmengine.evaluator import BaseMetric
from mmengine.model import BaseModel
from mmengine.runner import Runner


from cnnnet import CNNNet
from vit import ViT


class MMViT(BaseModel):
    def __init__(self):
        super().__init__()
        self.backbone = ViT()   # 用ViT作为backbone

    def forward(self, imgs, labels=None, mode='loss'):
        x = self.backbone(imgs)  # 前向传播得到 logits
        if mode == 'loss':
            return {'loss': F.cross_entropy(x, labels)}   # 训练时需要loss
        elif mode == 'predict':
            return x, labels   # 验证/测试时返回预测值和标签




class MMCNNNet(BaseModel):
    def __init__(self):
        super().__init__()
        self.backbone = CNNNet()   # 用CNNNet作为backbone

    def forward(self, imgs, labels=None, mode='loss'):
        x = self.backbone(imgs)  # 前向传播得到 logits
        if mode == 'loss':
            return {'loss': F.cross_entropy(x, labels)}   # 训练时需要loss
        elif mode == 'predict':
            return x, labels   # 验证/测试时返回预测值和标签


class MMResNet18(BaseModel):
    def __init__(self):
        super().__init__()
        self.resnet = torchvision.models.resnet18()

    def forward(self, imgs, labels, mode):
        x = self.resnet(imgs)
        if mode == 'loss':
            return {'loss': F.cross_entropy(x, labels)}
        elif mode == 'predict':
            return x, labels


class Accuracy(BaseMetric):
    def process(self, data_batch, data_samples):
        score, gt = data_samples
        self.results.append({
            'batch_size': len(gt),
            'correct': (score.argmax(dim=1) == gt).sum().cpu(),
        })

    def compute_metrics(self, results):
        total_correct = sum(item['correct'] for item in results)
        total_size = sum(item['batch_size'] for item in results)
        return dict(accuracy=100 * total_correct / total_size)


norm_cfg = dict(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
train_dataloader = DataLoader(batch_size=128,
                              shuffle=True,
                              dataset=torchvision.datasets.CIFAR10(
                                  'data/cifar10',
                                  train=True,
                                  download=True,
                                  transform=transforms.Compose([
                                    #   transforms.RandomCrop(32, padding=4),
                                    #   transforms.RandomHorizontalFlip(),
                                    #   transforms.ToTensor(),
                                    #   transforms.Normalize(**norm_cfg),                         
                                      transforms.RandomCrop(32, padding=4),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                                  ])))

val_dataloader = DataLoader(batch_size=128,
                            shuffle=False,
                            dataset=torchvision.datasets.CIFAR10(
                                'data/cifar10',
                                train=True,
                                download=True,
                                transform=transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Normalize(**norm_cfg),
                                ])))

runner = Runner(
    # model=MMResNet18(),
    # model=MMCNNNet(),
    model=MMViT(),  
    param_scheduler=dict(type='CosineAnnealingLR',by_epoch=True,T_max=200),
    work_dir='./work_dir',
    train_dataloader=train_dataloader,
    optim_wrapper=dict(optimizer=dict(type=SGD, lr=0.1, momentum=0.9,weight_decay=5e-4)),
    train_cfg=dict(by_epoch=True, max_epochs=5, val_interval=1),
    val_dataloader=val_dataloader,
    val_cfg=dict(),
    val_evaluator=dict(type=Accuracy),
)
runner.train()