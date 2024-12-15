from models.classification_models.ResNet import *
from models.segmentation_models.ResnetUnet import *
from models.segmentation_models.AttentionUNet import *
from models.segmentation_models.R2U_Net import *
from models.segmentation_models.R2AttU_Net import *
import torch

model4 = ResNet18()
model4.load_state_dict(torch.load('weights/classification_models/ResNet18.pt', map_location = torch.device('cpu')), strict=True)

print(model4)

dummy = torch.randn((1, 3, 256, 256))
out1 = model4(dummy)
out2 = model4(dummy)

print(out1, out2)
