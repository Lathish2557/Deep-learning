### Author: Silvija Kokalj-Filipovic
### Nov 2023
import torch
from torch.nn import Linear, Conv2d, Sequential, Dropout, GELU
from torch.nn import Softmax, Flatten, MaxPool2d, ReLU, CrossEntropyLoss
from torch.optim import SGD
import torchvision.models as models
import torch.nn as nn
from pytorch_resnetI import PyTorch_ResnetI
import torch.nn.init as init

class resnetI50(PyTorch_ResnetI):
    """
    Resnet Classifier used for the various ImageNet related datasets.
    This is resnet18 based - others should follow the pattern
    """
    def __init__(self, train_loader, valid_loader, num_classes, save_path, stop_early = False):
        super(resnetI50, self).__init__(train_loader,
                                    valid_loader,
                                    num_classes,
                                    save_path,
                                    stop_early)

    
    def _define_resnet_body(self):

        orig_model = models.resnet50(weights = models.ResNet50_Weights.IMAGENET1K_V1)
        layers = list(orig_model.children())[:-1]  #get all the layers except the last one
        model = nn.Sequential(*layers)
        return model

    def _define_linear_layers(self):
        linear_1 = Linear(2048, out_features=self._num_classes)
        init.xavier_normal_(linear_1.weight)
        init.zeros_(linear_1.bias)

        linear_layers = nn.Sequential(
            Flatten(),
            linear_1 
        )

        return linear_layers

    
    def _define_optimizer(self):
        return SGD(self.parameters(), lr = 0.01, momentum=0.9)

    def _define_loss_function(self):
        return CrossEntropyLoss()
