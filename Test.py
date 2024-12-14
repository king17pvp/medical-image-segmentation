import torch
import torch.nn as nn
import torchvision.models as models
from models.segmentation_models.AttentionUNet import *
from models.segmentation_models.R2U_Net import *
from models.segmentation_models.R2AttU_Net import *
from models.segmentation_models.ResnetUnet import *
from models.classification_models.ResNet import *

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

model4 = ResNet18()
model4.load_state_dict(torch.load('weights/ResNet18.pt', map_location = torch.device('cpu')), strict= False)
print(model4)

model5 = ResNet50()
model5.load_state_dict(torch.load('weights/ResNet50.pt', map_location = torch.device('cpu')), strict= False)
dummy = torch.randn((1, 3, 256, 256))
print(model5(dummy))
print(model5(dummy))
