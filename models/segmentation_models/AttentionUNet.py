import torch
import torch.nn as nn
import torchvision.models as models

def basic_block(in_channels, out_channels):
    block = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )
    return block

class UpConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpConv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x     
    
class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(AttentionGate,self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size = 1, stride = 1, padding = 0, bias = True),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size = 1, stride = 1, padding = 0, bias = True),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size = 1, stride = 1, padding = 0, bias = True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)

        return x*psi
    
class AttentionUNet(nn.Module):
    def __init__(self, in_channel = 3, out_channel = 1):
        super(AttentionUNet, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.max_pool = nn.MaxPool2d(kernel_size = 2, stride = 2)
        self.conv1 = basic_block(3, 64)
        self.conv2 = basic_block(64, 128)
        self.conv3 = basic_block(128, 256)
        self.conv4 = basic_block(256, 512) 
        self.conv5 = basic_block(512, 1024)

        self.up5 = UpConv(1024, 512)
        self.att5 = AttentionGate(F_g = 512, F_l = 512, F_int = 256)
        self.up_conv5 = basic_block(1024, 512)

        self.up4 = UpConv(512, 256)
        self.att4 = AttentionGate(F_g = 256, F_l = 256, F_int = 128)
        self.up_conv4 = basic_block(512, 256)

        self.up3 = UpConv(256, 128)
        self.att3 = AttentionGate(F_g = 128, F_l = 128, F_int = 64)
        self.up_conv3 = basic_block(256, 128)

        self.up2 = UpConv(128, 64)
        self.att2 = AttentionGate(F_g = 64, F_l = 64, F_int = 32)
        self.up_conv2 = basic_block(128, 64)

        self.out = nn.Conv2d(64, out_channel, kernel_size=1, stride=1, padding=0)

    def forward(self, x: torch.Tensor):
        # Encoding 
        x1 = self.conv1(x)
        x2 = self.max_pool(x1)
        x2 = self.conv2(x2)
        x3 = self.max_pool(x2)
        x3 = self.conv3(x3)
        x4 = self.max_pool(x3)
        x4 = self.conv4(x4)
        x5 = self.max_pool(x4)
        x5 = self.conv5(x5)
        
        #Decoding
        d5 = self.up5(x5)
        x4 = self.att5(g=d5,x=x4)
        d5 = torch.cat((x4,d5),dim=1)        
        d5 = self.up_conv5(d5)
        
        d4 = self.up4(d5)
        x3 = self.att4(g=d4,x=x3)
        d4 = torch.cat((x3,d4),dim=1)
        d4 = self.up_conv4(d4)

        d3 = self.up3(d4)
        x2 = self.att3(g=d3,x=x2)
        d3 = torch.cat((x2,d3),dim=1)
        d3 = self.up_conv3(d3)

        d2 = self.up2(d3)
        x1 = self.att2(g=d2,x=x1)
        d2 = torch.cat((x1,d2),dim=1)
        d2 = self.up_conv2(d2)

        d1 = self.out(d2)

        return d1  