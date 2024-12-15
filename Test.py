import torch
import torch.nn as nn
import torchvision.models as models
from models.segmentation_models.AttentionUNet import *
from models.segmentation_models.R2U_Net import *
from models.segmentation_models.R2AttU_Net import *
from models.segmentation_models.ResnetUnet import *
from models.classification_models.ResNet import *
from models.classification_models.VGG import *
# model0 = AttentionUNet()
# model0.load_state_dict(torch.load('weights/AttUNet.pt', map_location=torch.device('cpu')))
# print(model0)

# model1 = R2U_Net()
# model1.load_state_dict(torch.load('weights/R2UNet.pt', map_location=torch.device('cpu')))
# print(model1)

# model2 = R2AttU_Net()
# model2.load_state_dict(torch.load('weights/R2AttUNet.pt', map_location=torch.device('cpu')))
# print(model2)

# model3 = ResNetUnet()
# model3.load_state_dict(torch.load('weights/ResNetUNet.pt', map_location=torch.device('cpu')))

base_model = models.vgg16(pretrained=False)
base_features = nn.Sequential(*list(base_model.features))

# Adding layers
best_model = nn.Sequential(
    base_features,
    nn.AdaptiveAvgPool2d((1, 1)),
    nn.Flatten(),
    nn.Linear(512, 256),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(256, 256),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(256, 3),
    nn.Softmax(dim=1)
)

best_model.load_state_dict(torch.load('weights/classification_models/vgg16.pth', map_location = torch.device('cpu')))
print(best_model)
# model6 = VGG19(3)
# model6.load_state_dict(torch.load('weights/classification_models/VGG19.pt', map_location = torch.device('cpu')), strict = False)
# # torch.save(model6.state_dict(), 'weights/classification_models/VGG19.pt')
# print(model6)
# model6.eval()
# dummy = torch.randn((1, 3, 256, 256))
# dummy1 = torch.randn((1, 3, 256, 256))
# t = model6(dummy)
# v = model6(dummy)
# print(t, v)