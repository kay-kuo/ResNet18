import torch.nn as nn
import torch

# define residual block (shortcut)
class BasicBlock(nn.Module):
    
    def __init__(self, inputs, outputs, stride=1) -> None:
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=inputs, out_channels=outputs, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(outputs)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(in_channels=outputs, out_channels=outputs, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(outputs)
        

        self.downsample = None

        if stride != 1 or inputs != outputs:
            
            self.downsample = nn.Sequential(
                nn.Conv2d(inputs, outputs, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outputs),
            )

        self.stride = stride

    def forward(self, X):
        
        identity = X

        # layer 1
        out = self.conv1(X)
        out = self.bn1(out)
        out = self.relu(out)

        # layer 2
        out = self.conv2(out)
        out = self.bn2(out)


        if self.downsample is not None:
            identity = self.downsample(X)

        out += identity
        out = self.relu(out)

        return out


# define resnet18
class OurResNet18(nn.Module):

    def __init__(self)-> None:
        super().__init__()

        self.inputs = 64

        # 
        self.conv1 = nn.Conv2d(3, self.inputs, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)

        self.layer1 = self.make_layer(64, 2, stride=1)
        self.layer2 = self.make_layer(128, 2, stride=2)
        self.layer3 = self.make_layer(256, 2, stride=2)
        self.layer4 = self.make_layer(512, 2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, 10)

    def make_layer(self, channels, blocks, stride):

        layers = []

        layers.append(BasicBlock(self.inputs, channels, stride))
        self.inputs = channels
        
        for _ in range(1, blocks):
            layers.append(BasicBlock(self.inputs, channels, stride=1))
          
        return nn.Sequential(*layers)

    def forward(self, X):

        # input layer
        out = self.conv1(X)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.maxpool(out)

        # hidden layers
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        # output layer
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)

        return out
    