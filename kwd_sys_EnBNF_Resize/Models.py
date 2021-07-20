import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader

########################################################################
#                                ConvNet
########################################################################
# Architecture taken from original code here: https://github.com/idiap/CNN_QbE_STD/blob/master/Model_Query_Detection_DTW_CNN.py
class ConvNet(nn.Module):
    def __init__(self, max_height=100, max_width=300, dropout=0.1,depth=30):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels = 1, out_channels = 30, kernel_size = 3)
        self.conv2 = nn.Conv2d(in_channels = 30, out_channels = 30, kernel_size = 3)
        self.maxpool  = nn.MaxPool2d(kernel_size = 2, stride= 2)

        self.conv4 = nn.Conv2d(in_channels = 30, out_channels = 30, kernel_size = 3)
        self.conv5 = nn.Conv2d(in_channels = 30, out_channels = 30, kernel_size = 3)
        # 6 = maxpool

        self.conv7 = nn.Conv2d(in_channels = 30, out_channels = 30, kernel_size = 3)
        self.conv8 = nn.Conv2d(in_channels = 30, out_channels = 30, kernel_size = 3)
        # 9 = maxpool

        self.conv10 = nn.Conv2d(in_channels = 30, out_channels = 30, kernel_size = 3)
        self.conv11 = nn.Conv2d(in_channels = 30, out_channels = 15, kernel_size = 1)
        # 12 = maxpool (output size = M x depth/2 x 3 x 47)

        self.length = 15* 3* 16
        self.fc1 = nn.Linear(self.length, 60)
        # self.fc1 = nn.Linear(15*6*59, 60)
        self.fc2 = nn.Linear(60, 1)

        self.dout_layer = nn.Dropout(dropout)

    def forward(self, x):
        x = F.relu(self.dout_layer(self.conv1(x)))
        x = F.relu(self.dout_layer(self.conv2(x)))
        x = self.maxpool(x)
        
        x = F.relu(self.dout_layer(self.conv4(x)))
        x = F.relu(self.dout_layer(self.conv5(x)))
        x = self.maxpool(x)

        x = F.relu(self.dout_layer(self.conv7(x)))
        x = F.relu(self.dout_layer(self.conv8(x)))
        x = self.maxpool(x)

        x = F.relu(self.dout_layer(self.conv10(x)))
        x = F.relu(self.dout_layer(self.conv11(x)))
        x = self.maxpool(x)
        # print(x.size())

        # x = x.view(-1, self.length)
        x = x.view(-1, x.size()[1]*x.size()[2]*x.size()[3])
        x = F.relu(self.dout_layer(self.fc1(x)))
        x = self.fc2(x)
        x = torch.sigmoid(x)

        return x
########################################################################
#                                VGG
########################################################################
# VGG code adapted from: https://github.com/MLSpeech/speech_yolo/blob/master/model_speech_yolo.py
class VGG(nn.Module):
    def __init__(self, vgg_name):
        cfg = {
            'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
            'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
            'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
            'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
        }

        def _make_layers(cfg, kernel=3):
            layers = []
            in_channels = 1
            for x in cfg:
                if x == 'M':
                    layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
                else:
                    layers += [nn.Conv2d(in_channels, x, kernel_size=kernel, padding=1),
                                nn.BatchNorm2d(x),
                                nn.ReLU(inplace=True),
                                nn.Dropout(p = 0.1)]
                    in_channels = x
            layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
            return nn.Sequential(*layers)

        super(VGG, self).__init__()
        self.features = _make_layers(cfg[vgg_name])
        self.fc1 = nn.Linear(38400, 512)
        self.fc2 = nn.Linear(512, 1)
        self.dout_layer = nn.Dropout(0.1)

    def forward(self, x):
        # out = self.features(x)
        for m in self.features.children():
            # x_in = x.shape
            x = m(x)
            # print("%s -> %s" % (x_in, x.shape))

        out = x
        out = out.view(out.size(0), -1)
        out = self.dout_layer(self.fc1(out))
        out = self.fc2(out)
        return torch.sigmoid(out)

class VGG11(VGG):
    def __init__(self,max_height=100, max_width=300):
        VGG.__init__(self, 'VGG11')

conv_block = nn.Sequential(nn.Conv2d(3,64,kernel_size=7, stride=2, padding=3, bias=False), #112,112
                            nn.BatchNorm2d(64),
                            nn.ReLU(inplace=True),
                            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)) # 56,56


########################################################################
#                                ResNet
########################################################################
# ResNet code adapted from: https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py

class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class SEBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, reduction=8):
        super(SEBasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.se = SELayer(planes, reduction)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.relu(out)
        out = self.bn1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out

class SELayer(nn.Module):
    def __init__(self, channel, reduction=8):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
                nn.Linear(channel, channel // reduction),
                nn.ReLU(inplace=True),
                nn.Linear(channel // reduction, channel),
                nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class ResNet(nn.Module):

    def __init__(self, block, layers, encoder='ASP', num_classes=1):
        super().__init__()
        
        self.inplanes = 16

        self.conv1 = nn.Conv2d(in_channels = 1, out_channels = self.inplanes, kernel_size=7, stride=(2, 1), padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(block, 16, layers[0])
        self.layer2 = self._make_layer(block, 32, layers[1], stride=(2, 2))
        self.layer3 = self._make_layer(block, 64, layers[2], stride=(2, 2))
        self.layer4 = self._make_layer(block, 128,layers[3], stride=(1, 1))
        
        self.encoder = encoder
        if self.encoder == 'AVG':
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(512 , num_classes)
        elif self.encoder == "ASP":
            self.sap_linear = nn.Linear(128, 128)
            self.attention = self.new_parameter(128, 1)
            self.fc = nn.Linear(128 * 2 , num_classes)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None  

        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes, 1, stride, bias=False),
                nn.BatchNorm2d(planes))
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes    
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)
    
    def new_parameter(self, *size):
        out = nn.Parameter(torch.FloatTensor(*size))
        nn.init.xavier_normal_(out)
        return out    

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        # x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        out5 = torch.mean(x, dim=2, keepdim=True)

        if self.encoder == 'AVG':
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
        elif self.encoder == "ASP":
            x = out5.permute(0,3,1,2).squeeze(-1)                   # x:[1, 135, 128]
            h = torch.tanh(self.sap_linear(x))                      # h:[1, 135, 128]
            w = torch.matmul(h, self.attention).squeeze(dim=2)      # w:[1, 135]
            w = F.softmax(w, dim=1).view(x.size(0), x.size(1), 1)   # w:[1, 135, 1]
            mu = torch.sum(x * w, dim=1)                            # mu:[1, 128]
            rh = torch.sqrt( ( torch.sum((x**2) * w, dim=1) - mu**2 ).clamp(min=1e-5) )
            x = torch.cat((mu,rh),1)
        x = self.fc(x)
        return torch.sigmoid(x)

class ResNet34(ResNet):
    def __init__(self,max_height=100, max_width=300):
        layers=[3, 4, 6, 3]
        ResNet.__init__(self, BasicBlock, layers)

class ResNetSE34L(ResNet):
    def __init__(self,max_height=100, max_width=300):
        layers=[3, 4, 6, 3]
        ResNet.__init__(self, SEBasicBlock, layers)


########################################################################
#                                BiLSTM
########################################################################
class TimeDistributed(nn.Module):
    def __init__(self, module, batch_first=False):
        super(TimeDistributed, self).__init__()
        self.module = module
        self.batch_first = batch_first

    def forward(self, x):
        if len(x.size()) <= 2:
            return self.module(x)

        # reshape input data --> (samples * timesteps, input_size)
        # squash timesteps
        x_reshaped = x.contiguous().view(-1, x.size(-1))
        y = self.module(x_reshaped)

        # We have to reshape Y
        if self.batch_first:
            # (samples, timesteps, output_size)
            y = y.contiguous().view(x.size(0), -1, y.size(-1))
        else:
            # (timesteps, samples, output_size)
            y = y.contiguous().view(-1, x.size(1), y.size(-1))
        return y


class BiLSTM(nn.Module):

    def __init__(self, max_height=100, max_width=300):
        '''
        The constructor of this class.
        It initiates a few things:
         
        - LSTM layers according to configuration in config/config.yaml
        - Initiates the weights and biases
        - Regular feed forward layer
        '''
        super(BiLSTM, self).__init__()    
        self.bLSTM = nn.LSTM(input_size=128, 
                                hidden_size=128, 
                                num_layers=1,
                                dropout=0.2,
                                bidirectional=False
                                ) #batch_first=True

        ########################################################
        ## init the forget gate weight=-3, other gata weight=0 #
        ########################################################        
        for name, param in self.bLSTM.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0) # let all bias=0
                with torch.no_grad():
                    param[100:200].fill_(-3)  # init the forget gate weight=-3
            elif 'weight' in name:
                nn.init.xavier_normal_(param) # use xavier init weights

        self.conv1d = nn.Conv1d(in_channels=100, out_channels=128, kernel_size=3)
        self.BN = nn.BatchNorm1d(128)
        self.relu = nn.ReLU(inplace=True)
        self.conv2d = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3)
        self.BN2 = nn.BatchNorm1d(128)
        self.maxpooling = nn.MaxPool1d(kernel_size=2, stride=2)
        self.fc1 = TimeDistributed(nn.Linear(128, 50), batch_first=True)
        self.elu = nn.ELU()
        self.dropout = nn.Dropout(p=0.2)
        self.fc2 = TimeDistributed(nn.Linear(50, 1), batch_first=True)


    def forward(self, x):
        '''
        :param x: a matrix of N x M (number of speakers, utteraces)
        :type x: torch.Tensor
        '''
        x = x.squeeze()

        #============================================================================================================
        # CNN
        #============================================================================================================
        # x = x.data.permute(0,2,1)
        x = self.conv1d(x)
        x = self.BN(x)
        x = self.relu(x)
        #x = self.conv2d(x)
        #x = self.BN2(x)
        #x = self.relu(x)
        x = x.permute(0, 2, 1)

        #============================================================================================================
        # bLSTM
        #============================================================================================================
        x, (h, _) = self.bLSTM(x)
        x = self.fc1(x) # x.shape: [batch, frames, 50]
        x = self.elu(x)
        x = self.dropout(x)
        frame_score = self.fc2(x)
        average_score = F.adaptive_avg_pool2d(frame_score, (1,1)) # average_score.shape: [batch, 1, 1]
        average_score = average_score.view(-1,1)
        average_score = torch.sigmoid(average_score)
        return average_score


if __name__ == "__main__":
    model = BiLSTM()
    input = torch.randn(10, 100, 257)
    a = model(input)
