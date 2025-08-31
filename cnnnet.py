import torch.nn as nn
import torch.nn.functional as F
import torch
from torchsummary import summary
class CNNNet(nn.Module):
    def __init__(self):
        super(CNNNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32) #32是通道数，不改变维度
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64) #64是通道数，不改变维度
        self.pool = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128) #128是通道数，不改变维度
        self.conv4 = nn.Conv2d(128, 128, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(128) #128是通道数，不改变维度
        self.fc1 = nn.Linear(128*4*4, 256)
        self.dropout = nn.Dropout(0.5) # 每个元素有50%的概率变成0,不改变维度
        self.fc2 = nn.Linear(256, 10)
    def forward(self,x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool(x)

        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

def testNet(workNet):
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 使用GPU或者CPU
    model = workNet.to(DEVICE)

    x = torch.randn(1,3,32,32).to(DEVICE)
    y = model(x)
    print(y.size())

    summary(model,(3,32,32))

if __name__ == "__main__":
    # model = CNNNet()
    # testNet(model)
    exit
