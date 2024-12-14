import torch
import torch.nn as nn
import torchvision.models as models
from models.segmentation_models.AttentionUNet import *
from models.segmentation_models.R2U_Net import *
from models.segmentation_models.R2AttU_Net import *

model0 = AttentionUNet()
ckpt = torch.load('weights/AttentionUNetWeight.pt', map_location=torch.device('cpu'))
model0.load_state_dict(torch.load('weights/AttentionUNetWeight.pt', map_location=torch.device('cpu'))['model_state_dict'])
print(model0)

model1 = R2U_Net()
model1.load_state_dict(torch.load('weights/R2UNetWeight.pt', map_location=torch.device('cpu'))['model_state_dict'])
print(model1)

model2 = R2AttU_Net()
model2.load_state_dict(torch.load('weights/AttentionR2UNetWeight.pt', map_location=torch.device('cpu'))['model_state_dict'])
print(model2)