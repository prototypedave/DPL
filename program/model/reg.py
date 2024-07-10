import torch.nn as nn
from torchvision import models

class DPL(nn.Module):
    def __init__(self, num_input_channels=16):
        super(DPL, self).__init__()
        self.resnet = models.resnet34(weights='ResNet34_Weights.IMAGENET1K_V1')
        self.resnet.conv1 = nn.Conv2d(num_input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.resnet.bn1 = nn.BatchNorm2d(64)
        self.resnet.relu = nn.ReLU(inplace=True)
        self.resnet.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.resnet.layer1 = self.resnet.layer1
        self.resnet.layer2 = self.resnet.layer2
        self.resnet.layer3 = self.resnet.layer3
        self.resnet.layer4 = self.resnet.layer4

        self.extra_layer = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )
        
        # Upsampling layers
        self.upsample1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2) 
        self.upsample2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2) 
        self.upsample3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)  
        self.upsample4 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)   
        self.upsample5 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)   

        self.conv_final = nn.Conv2d(16, 1, kernel_size=3, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)
        
        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)
        
        x = self.extra_layer(x)

        x = self.upsample1(x)
        x = self.upsample2(x)
        x = self.upsample3(x)
        x = self.upsample4(x)
        x = self.upsample5(x)
        
        x = self.conv_final(x)
        x = self.sigmoid(x)
        
        return x