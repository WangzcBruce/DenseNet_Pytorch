import torch
import torch.nn as nn


class DenseBlock_layer(nn.Module):
    # DenseBlock中有很多layer，比如6个【1*1conv， 3*3conv】
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super(DenseBlock_layer, self).__init__()
        # 先进行batchnorm2d操作， 之后relu激活， conv1*1操作，
        # batchnorm2d操作， 之后relu激活， conv3*3操作，
        # bn_size*growth_rate不太懂？？？
        self.DenseBlock_layer_module = [
            nn.BatchNorm2d(num_input_features),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_input_features, bn_size*growth_rate, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(bn_size*growth_rate),
            nn.ReLU(inplace=True),
            nn.Conv2d(bn_size*growth_rate, growth_rate, kernel_size=3, stride=1, padding=1, bias=False)
        ]
        self.DenseBlock_layer_module = nn.Sequential(*self.DenseBlock_layer_module)
        self.drop_rate = drop_rate


    def forward(self, x):
        out = self.DenseBlock_layer_module(x)
        # 每次经过一个DenseBlock——layer后都进行dropout
        if self.drop_rate > 0:
            out = nn.Dropout(x, self.drop_rate)
        # dropout的结果和原有输入进行合并
        return torch.cat([x, out], dim=1)



class DenseBlock(nn.Module):
    # DenseBlock 有很多DenseBlock——Layer组成
    # num_layers：个数 num_input_features：每一层输入特征通道数量
    # bn_size, growth_rate？不太懂
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate):
        super(DenseBlock, self).__init__()
        self.DenseBlock_Module = [
            DenseBlock_layer(num_input_features + i * growth_rate, growth_rate, bn_size, drop_rate) for i in range(num_layers)
        ]
        self.DenseBlock_Module = nn.Sequential(*self.DenseBlock_Module)
    def forward(self, x):
        return self.DenseBlock_Module(x)



class Transition(nn.Module):
    # DenseBlock之后要有transition模块进行过度，降低尺寸大小
    def __init__(self, num_input_features, num_output_features):
        super(Transition, self).__init__()
        self.Transition_Module = [
            nn.BatchNorm2d(num_input_features),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_input_features, num_output_features, kernel_size=1, stride=1, bias=False),
            nn.AvgPool2d(kernel_size=2, stride=2)
        ]
        self.Transition_Module = nn.Sequential(*self.Transition_Module)
    def forward(self, x):
        return self.Transition_Module(x)


class DenseNet(nn.Module):
    def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16),
                 num_init_features=64, bn_size=4, drop_rate=0, num_classes=1000):
        super(DenseNet, self).__init__()
        # 首先进行预处理， 尺寸缩小1/2（向上取整）， 通道数量变为num_init_features
        self.conv1 = nn.Conv2d(3, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(num_init_features)
        self.relu1 = nn.ReLU(inplace=True)
        # 尺寸缩小1/2（向上取整）
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        num_features = num_init_features
        Dense_stage = []
        # block_config=(6, 12, 24, 16)
        # DenseBlock每个stage由6，12，24，16个DenseBlock——layer重复组成
        # 每个stage之间由transition模块过度，进行下采样
        for i, num_layers in enumerate(block_config):
            # Denseblock由num_layers个Denseblock——layer组成，输入通道数量num_features
            # 输出通道数量每经过一个layer就增加growth——rate个
            block = DenseBlock(num_layers=num_layers, num_input_features=num_features,
                                bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
            Dense_stage.append(block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = Transition(num_input_features=num_features, num_output_features=num_features // 2)
                Dense_stage.append(trans)
                num_features = num_features // 2
        self.Dense_stage = nn.Sequential(*Dense_stage)
        self.avgpool = nn.AvgPool2d(kernel_size=7)
        self.linear = nn.Linear(num_features, num_classes)
    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.Dense_stage(x)
        x = self.avgpool(x)
        x = self.linear(x)
        return x



print(DenseNet())