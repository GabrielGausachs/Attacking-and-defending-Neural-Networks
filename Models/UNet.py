import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
import torch.nn.functional as F

class C2(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(C2, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)
    
class C3(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(C3, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 2, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)

class DUNET(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, features=[64, 128, 256, 256]):

        super(DUNET, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        #self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Down part of UNET
        for i,feature in enumerate(features):
            if i == 0: # First layer sequence
                self.downs.append(C2(in_channels, feature))
                in_channels = feature
            else:
                self.downs.append(C3(in_channels, feature))
                in_channels = feature

        # Up part of UNET
        reversed_features = list(reversed(features))
        for i,feature in enumerate(reversed_features):
            if i == 3:
                self.ups.append(C2(feature, out_channels))
            else:
                self.ups.append(C3(feature, reversed_features[i+1]))
        
        self.bottleneck = C3(features[-1], features[-1])
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)

        x = self.bottleneck(x)

        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups)):
            x = F.interpolate(x,size = skip_connections[idx].shape, mode='bilinear', align_corners=True)

            concat_skip = torch.cat((skip_connections[idx], x), dim=1)
            x = self.ups[idx](concat_skip)

        return self.final_conv(x)
    
    def freeze_except_final(self):
        for param in self.parameters():
            param.requires_grad = False
        for param in self.final_conv.parameters():
            param.requires_grad = True
