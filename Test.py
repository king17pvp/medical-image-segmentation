import torch
import torch.nn as nn
import torchvision.models as models
from models.segmentation_models.AttentionUNet import *
from models.segmentation_models.R2U_Net import *
from models.segmentation_models.R2AttU_Net import *
from models.segmentation_models.ResnetUnet import *

model0 = AttentionUNet()
model0.load_state_dict(torch.load('weights/AttUNet.pt', map_location=torch.device('cpu')))
print(model0)

model1 = R2U_Net()
model1.load_state_dict(torch.load('weights/R2UNet.pt', map_location=torch.device('cpu')))
print(model1)

model2 = R2AttU_Net()
model2.load_state_dict(torch.load('weights/R2AttUNet.pt', map_location=torch.device('cpu')))
print(model2)

model3 = ResNetUnet()
model3.load_state_dict(torch.load('weights/ResNetUNet.pt', map_location=torch.device('cpu')))