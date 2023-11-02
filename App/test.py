import torch
import torchvision.models as models
import torch.nn as nn
resnet50 = models.resnet50()
num_classes = 39
state_dict = torch.load("ResNet50.pt")
resnet50.fc = nn.Linear(resnet50.fc.in_features, num_classes)
