from __future__ import absolute_import

from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
from torch.nn import init
import torch
import torchvision
import math

from .resnet import *


__all__ = ["End2End_AvgPooling"]


class AvgPooling(nn.Module):
    def __init__(self, input_feature_size, embeding_fea_size=1024, num_classes=0, dropout=0.5):
        super(self.__class__, self).__init__()

        # embeding
        self.embeding_fea_size = embeding_fea_size
        self.embeding = nn.Linear(input_feature_size, embeding_fea_size)
        self.embeding_bn = nn.BatchNorm1d(embeding_fea_size)
        self.num_classes = num_classes
        init.kaiming_normal_(self.embeding.weight, mode='fan_out')
        init.constant_(self.embeding.bias, 0)
        init.constant_(self.embeding_bn.weight, 1)
        init.constant_(self.embeding_bn.bias, 0)
        self.drop = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.embeding_fea_size, self.num_classes)
        init.normal_(self.classifier.weight, std=0.001)
        init.constant_(self.classifier.bias, 0)

    def forward(self, inputs):
#        net = inputs.mean(dim = 1)            
#        eval_feas = F.normalize(net, p=2, dim=1)
        net = self.embeding(inputs)
#        net = self.embeding_bn(net)
        net_norm = F.normalize(net, p=2, dim=1)
#        net_drop = self.drop(net_norm) 
        logits = self.classifier(F.relu(net))       
        return net_norm, logits

class End2End_AvgPooling(nn.Module):

    def __init__(self, dropout=0,  embeding_fea_size=1024, num_classes=0, fixed_layer=True):
        super(self.__class__, self).__init__()
        self.CNN = resnet50(dropout=dropout, fixed_layer=fixed_layer)
        self.avg_pooling = AvgPooling(input_feature_size=2048, embeding_fea_size=embeding_fea_size, 
                                      num_classes=num_classes, dropout=dropout)

    def forward(self, x):
        assert len(x.data.shape) == 5
        # reshape (batch, samples, ...) ==> (batch * samples, ...)
        oriShape = x.data.shape
        x = x.view(-1, oriShape[2], oriShape[3], oriShape[4])
        
        # resnet encoding
        resnet_feature = self.CNN(x)

        # reshape back into (batch, samples, ...)
        resnet_feature = resnet_feature.view(oriShape[0], oriShape[1], -1)
#        print(resnet_feature.shape)

        # avg pooling
        output = self.avg_pooling(resnet_feature)
        return output
