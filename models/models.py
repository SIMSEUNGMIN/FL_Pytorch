import torch.nn.functional as F
from torch import nn


def get_model(dataset_name):
    # if dataset_name == "MNIST":
    #     return Mnist_Net()
    if dataset_name == "FashionMNIST":
        return Mnist_Net()
    # if dataset_name == "CIFAR-10":
    #     return Mnist_Net()


class Mnist_Net(nn.Module):
    # AlexNet
    def __init__(self):
        super(Mnist_Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=96, kernel_size=11, stride=4) #for gray scale
        # self.conv1 = nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4)
        self.conv2 = nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, padding=1) # conv5와 fc1 사이에 view 들어간다.
        self.fc1 = nn.Linear(in_features=256 * 6 * 6, out_features=4096) # fc layer
        self.fc2 = nn.Linear(in_features=4096, out_features=4096)
        self.fc3 = nn.Linear(in_features=4096, out_features=10)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        
        x = x.view(x.size(0), -1) # 4차원을 1차원으로 펼쳐주는 층 (역할) -> flatten
        
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.5)
        
        x = F.relu(self.fc2(x))
        x = F.dropout(x, p=0.5)
    
        x = F.log_softmax(self.fc3(x), dim=1)
        
        return x 

    # 모델 참조 https://jschang.tistory.com/4
    # def __init__(self):
    #     super(Mnist_Net, self).__init__()
    #     self.layer1 = nn.Sequential( # 순차적인 레이어 쌓게 함
    #         # Convolution + ReLU + max Pool
    #         nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, stride=1, padding=2),
    #         # Wout = (Win - FilterSize + 2*Padding)/Stride + 1
    #         nn.ReLU(),
    #         nn.MaxPool2d(kernel_size=2, stride=2)
    #     )
    #     self.layer2 = nn.Sequential( # 순차적인 레이어 쌓게 함
    #         # Convolution + ReLU + max Pool
    #         nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2),
    #         nn.ReLU(),
    #         nn.MaxPool2d(kernel_size=2, stride=2)
    #     )
    #     self.dropout = nn.Dropout() # over-fitting 방지 
    #     self.fc1 = nn.Linear(in_features=7*7*64, out_features=1000)
    #     self.fc2 = nn.Linear(in_features=1000, out_features=10)

    # def forward(self, x):
    #     x = self.layer1(x)
    #     x = self.layer2(x)
    #     x = x.reshape(x.size(0), -1)
    #     x = self.dropout(x) # 오버피팅을 막기 위해 학습 과정시 일부 뉴런을 생략하는 기능 
    #     x = self.fc1(x)
    #     x = self.fc2(x)
    #     return x


    # for MNIST
    # def __init__(self):
    #     super(Mnist_Net, self).__init__()
    #     self.conv1 = nn.Conv2d(1, 32, 5)
    #     self.pool = nn.MaxPool2d(2, 2)
    #     self.conv2 = nn.Conv2d(32, 64, 5)
    #     self.fc = nn.Sequential(
    #         nn.Linear(64 * 4 * 4, 512),
    #         nn.ReLU(),
    #         nn.Linear(512, 10)
    #     )

    # def forward(self, x):
    #     x = self.pool(F.relu(self.conv1(x)))
    #     x = self.pool(F.relu(self.conv2(x)))
    #     x = x.view(-1, 4 * 4 * 64)
    #     x = self.fc(x)
    #     return x
