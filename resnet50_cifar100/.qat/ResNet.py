# GENETARED BY NNDCT, DO NOT EDIT!

import torch
import pytorch_nndct as py_nndct
class ResNet(torch.nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        self.module_0 = py_nndct.nn.Input() #ResNet::input_0
        self.module_1 = py_nndct.nn.quant_input() #ResNet::ResNet/QuantStub[quant_stub]/input.1
        self.module_2 = py_nndct.nn.Conv2d(in_channels=3, out_channels=64, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #ResNet::ResNet/Conv2d[conv1]/input.3
        self.module_3 = py_nndct.nn.ReLU(inplace=True) #ResNet::ResNet/ReLU[relu]/input.7
        self.module_4 = py_nndct.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #ResNet::ResNet/Sequential[layer1]/Bottleneck[0]/Conv2d[conv1]/input.9
        self.module_5 = py_nndct.nn.ReLU(inplace=True) #ResNet::ResNet/Sequential[layer1]/Bottleneck[0]/ReLU[relu1]/input.13
        self.module_6 = py_nndct.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #ResNet::ResNet/Sequential[layer1]/Bottleneck[0]/Conv2d[conv2]/input.15
        self.module_7 = py_nndct.nn.ReLU(inplace=True) #ResNet::ResNet/Sequential[layer1]/Bottleneck[0]/ReLU[relu2]/input.19
        self.module_8 = py_nndct.nn.Conv2d(in_channels=64, out_channels=256, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #ResNet::ResNet/Sequential[layer1]/Bottleneck[0]/Conv2d[conv3]/input.21
        self.module_9 = py_nndct.nn.Conv2d(in_channels=64, out_channels=256, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #ResNet::ResNet/Sequential[layer1]/Bottleneck[0]/Sequential[downsample]/Conv2d[0]/input.23
        self.module_10 = py_nndct.nn.Add() #ResNet::ResNet/Sequential[layer1]/Bottleneck[0]/Add[skip_add]/input.25
        self.module_11 = py_nndct.nn.ReLU(inplace=True) #ResNet::ResNet/Sequential[layer1]/Bottleneck[0]/ReLU[relu3]/input.27
        self.module_12 = py_nndct.nn.Conv2d(in_channels=256, out_channels=64, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #ResNet::ResNet/Sequential[layer1]/Bottleneck[1]/Conv2d[conv1]/input.29
        self.module_13 = py_nndct.nn.ReLU(inplace=True) #ResNet::ResNet/Sequential[layer1]/Bottleneck[1]/ReLU[relu1]/input.33
        self.module_14 = py_nndct.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #ResNet::ResNet/Sequential[layer1]/Bottleneck[1]/Conv2d[conv2]/input.35
        self.module_15 = py_nndct.nn.ReLU(inplace=True) #ResNet::ResNet/Sequential[layer1]/Bottleneck[1]/ReLU[relu2]/input.39
        self.module_16 = py_nndct.nn.Conv2d(in_channels=64, out_channels=256, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #ResNet::ResNet/Sequential[layer1]/Bottleneck[1]/Conv2d[conv3]/input.41
        self.module_17 = py_nndct.nn.Add() #ResNet::ResNet/Sequential[layer1]/Bottleneck[1]/Add[skip_add]/input.43
        self.module_18 = py_nndct.nn.ReLU(inplace=True) #ResNet::ResNet/Sequential[layer1]/Bottleneck[1]/ReLU[relu3]/input.45
        self.module_19 = py_nndct.nn.Conv2d(in_channels=256, out_channels=64, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #ResNet::ResNet/Sequential[layer1]/Bottleneck[2]/Conv2d[conv1]/input.47
        self.module_20 = py_nndct.nn.ReLU(inplace=True) #ResNet::ResNet/Sequential[layer1]/Bottleneck[2]/ReLU[relu1]/input.51
        self.module_21 = py_nndct.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #ResNet::ResNet/Sequential[layer1]/Bottleneck[2]/Conv2d[conv2]/input.53
        self.module_22 = py_nndct.nn.ReLU(inplace=True) #ResNet::ResNet/Sequential[layer1]/Bottleneck[2]/ReLU[relu2]/input.57
        self.module_23 = py_nndct.nn.Conv2d(in_channels=64, out_channels=256, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #ResNet::ResNet/Sequential[layer1]/Bottleneck[2]/Conv2d[conv3]/input.59
        self.module_24 = py_nndct.nn.Add() #ResNet::ResNet/Sequential[layer1]/Bottleneck[2]/Add[skip_add]/input.61
        self.module_25 = py_nndct.nn.ReLU(inplace=True) #ResNet::ResNet/Sequential[layer1]/Bottleneck[2]/ReLU[relu3]/input.63
        self.module_26 = py_nndct.nn.Conv2d(in_channels=256, out_channels=128, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #ResNet::ResNet/Sequential[layer2]/Bottleneck[0]/Conv2d[conv1]/input.65
        self.module_27 = py_nndct.nn.ReLU(inplace=True) #ResNet::ResNet/Sequential[layer2]/Bottleneck[0]/ReLU[relu1]/input.69
        self.module_28 = py_nndct.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=[3, 3], stride=[2, 2], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #ResNet::ResNet/Sequential[layer2]/Bottleneck[0]/Conv2d[conv2]/input.71
        self.module_29 = py_nndct.nn.ReLU(inplace=True) #ResNet::ResNet/Sequential[layer2]/Bottleneck[0]/ReLU[relu2]/input.75
        self.module_30 = py_nndct.nn.Conv2d(in_channels=128, out_channels=512, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #ResNet::ResNet/Sequential[layer2]/Bottleneck[0]/Conv2d[conv3]/input.77
        self.module_31 = py_nndct.nn.Conv2d(in_channels=256, out_channels=512, kernel_size=[1, 1], stride=[2, 2], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #ResNet::ResNet/Sequential[layer2]/Bottleneck[0]/Sequential[downsample]/Conv2d[0]/input.79
        self.module_32 = py_nndct.nn.Add() #ResNet::ResNet/Sequential[layer2]/Bottleneck[0]/Add[skip_add]/input.81
        self.module_33 = py_nndct.nn.ReLU(inplace=True) #ResNet::ResNet/Sequential[layer2]/Bottleneck[0]/ReLU[relu3]/input.83
        self.module_34 = py_nndct.nn.Conv2d(in_channels=512, out_channels=128, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #ResNet::ResNet/Sequential[layer2]/Bottleneck[1]/Conv2d[conv1]/input.85
        self.module_35 = py_nndct.nn.ReLU(inplace=True) #ResNet::ResNet/Sequential[layer2]/Bottleneck[1]/ReLU[relu1]/input.89
        self.module_36 = py_nndct.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #ResNet::ResNet/Sequential[layer2]/Bottleneck[1]/Conv2d[conv2]/input.91
        self.module_37 = py_nndct.nn.ReLU(inplace=True) #ResNet::ResNet/Sequential[layer2]/Bottleneck[1]/ReLU[relu2]/input.95
        self.module_38 = py_nndct.nn.Conv2d(in_channels=128, out_channels=512, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #ResNet::ResNet/Sequential[layer2]/Bottleneck[1]/Conv2d[conv3]/input.97
        self.module_39 = py_nndct.nn.Add() #ResNet::ResNet/Sequential[layer2]/Bottleneck[1]/Add[skip_add]/input.99
        self.module_40 = py_nndct.nn.ReLU(inplace=True) #ResNet::ResNet/Sequential[layer2]/Bottleneck[1]/ReLU[relu3]/input.101
        self.module_41 = py_nndct.nn.Conv2d(in_channels=512, out_channels=128, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #ResNet::ResNet/Sequential[layer2]/Bottleneck[2]/Conv2d[conv1]/input.103
        self.module_42 = py_nndct.nn.ReLU(inplace=True) #ResNet::ResNet/Sequential[layer2]/Bottleneck[2]/ReLU[relu1]/input.107
        self.module_43 = py_nndct.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #ResNet::ResNet/Sequential[layer2]/Bottleneck[2]/Conv2d[conv2]/input.109
        self.module_44 = py_nndct.nn.ReLU(inplace=True) #ResNet::ResNet/Sequential[layer2]/Bottleneck[2]/ReLU[relu2]/input.113
        self.module_45 = py_nndct.nn.Conv2d(in_channels=128, out_channels=512, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #ResNet::ResNet/Sequential[layer2]/Bottleneck[2]/Conv2d[conv3]/input.115
        self.module_46 = py_nndct.nn.Add() #ResNet::ResNet/Sequential[layer2]/Bottleneck[2]/Add[skip_add]/input.117
        self.module_47 = py_nndct.nn.ReLU(inplace=True) #ResNet::ResNet/Sequential[layer2]/Bottleneck[2]/ReLU[relu3]/input.119
        self.module_48 = py_nndct.nn.Conv2d(in_channels=512, out_channels=128, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #ResNet::ResNet/Sequential[layer2]/Bottleneck[3]/Conv2d[conv1]/input.121
        self.module_49 = py_nndct.nn.ReLU(inplace=True) #ResNet::ResNet/Sequential[layer2]/Bottleneck[3]/ReLU[relu1]/input.125
        self.module_50 = py_nndct.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #ResNet::ResNet/Sequential[layer2]/Bottleneck[3]/Conv2d[conv2]/input.127
        self.module_51 = py_nndct.nn.ReLU(inplace=True) #ResNet::ResNet/Sequential[layer2]/Bottleneck[3]/ReLU[relu2]/input.131
        self.module_52 = py_nndct.nn.Conv2d(in_channels=128, out_channels=512, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #ResNet::ResNet/Sequential[layer2]/Bottleneck[3]/Conv2d[conv3]/input.133
        self.module_53 = py_nndct.nn.Add() #ResNet::ResNet/Sequential[layer2]/Bottleneck[3]/Add[skip_add]/input.135
        self.module_54 = py_nndct.nn.ReLU(inplace=True) #ResNet::ResNet/Sequential[layer2]/Bottleneck[3]/ReLU[relu3]/input.137
        self.module_55 = py_nndct.nn.Conv2d(in_channels=512, out_channels=256, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #ResNet::ResNet/Sequential[layer3]/Bottleneck[0]/Conv2d[conv1]/input.139
        self.module_56 = py_nndct.nn.ReLU(inplace=True) #ResNet::ResNet/Sequential[layer3]/Bottleneck[0]/ReLU[relu1]/input.143
        self.module_57 = py_nndct.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=[3, 3], stride=[2, 2], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #ResNet::ResNet/Sequential[layer3]/Bottleneck[0]/Conv2d[conv2]/input.145
        self.module_58 = py_nndct.nn.ReLU(inplace=True) #ResNet::ResNet/Sequential[layer3]/Bottleneck[0]/ReLU[relu2]/input.149
        self.module_59 = py_nndct.nn.Conv2d(in_channels=256, out_channels=1024, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #ResNet::ResNet/Sequential[layer3]/Bottleneck[0]/Conv2d[conv3]/input.151
        self.module_60 = py_nndct.nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=[1, 1], stride=[2, 2], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #ResNet::ResNet/Sequential[layer3]/Bottleneck[0]/Sequential[downsample]/Conv2d[0]/input.153
        self.module_61 = py_nndct.nn.Add() #ResNet::ResNet/Sequential[layer3]/Bottleneck[0]/Add[skip_add]/input.155
        self.module_62 = py_nndct.nn.ReLU(inplace=True) #ResNet::ResNet/Sequential[layer3]/Bottleneck[0]/ReLU[relu3]/input.157
        self.module_63 = py_nndct.nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #ResNet::ResNet/Sequential[layer3]/Bottleneck[1]/Conv2d[conv1]/input.159
        self.module_64 = py_nndct.nn.ReLU(inplace=True) #ResNet::ResNet/Sequential[layer3]/Bottleneck[1]/ReLU[relu1]/input.163
        self.module_65 = py_nndct.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #ResNet::ResNet/Sequential[layer3]/Bottleneck[1]/Conv2d[conv2]/input.165
        self.module_66 = py_nndct.nn.ReLU(inplace=True) #ResNet::ResNet/Sequential[layer3]/Bottleneck[1]/ReLU[relu2]/input.169
        self.module_67 = py_nndct.nn.Conv2d(in_channels=256, out_channels=1024, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #ResNet::ResNet/Sequential[layer3]/Bottleneck[1]/Conv2d[conv3]/input.171
        self.module_68 = py_nndct.nn.Add() #ResNet::ResNet/Sequential[layer3]/Bottleneck[1]/Add[skip_add]/input.173
        self.module_69 = py_nndct.nn.ReLU(inplace=True) #ResNet::ResNet/Sequential[layer3]/Bottleneck[1]/ReLU[relu3]/input.175
        self.module_70 = py_nndct.nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #ResNet::ResNet/Sequential[layer3]/Bottleneck[2]/Conv2d[conv1]/input.177
        self.module_71 = py_nndct.nn.ReLU(inplace=True) #ResNet::ResNet/Sequential[layer3]/Bottleneck[2]/ReLU[relu1]/input.181
        self.module_72 = py_nndct.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #ResNet::ResNet/Sequential[layer3]/Bottleneck[2]/Conv2d[conv2]/input.183
        self.module_73 = py_nndct.nn.ReLU(inplace=True) #ResNet::ResNet/Sequential[layer3]/Bottleneck[2]/ReLU[relu2]/input.187
        self.module_74 = py_nndct.nn.Conv2d(in_channels=256, out_channels=1024, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #ResNet::ResNet/Sequential[layer3]/Bottleneck[2]/Conv2d[conv3]/input.189
        self.module_75 = py_nndct.nn.Add() #ResNet::ResNet/Sequential[layer3]/Bottleneck[2]/Add[skip_add]/input.191
        self.module_76 = py_nndct.nn.ReLU(inplace=True) #ResNet::ResNet/Sequential[layer3]/Bottleneck[2]/ReLU[relu3]/input.193
        self.module_77 = py_nndct.nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #ResNet::ResNet/Sequential[layer3]/Bottleneck[3]/Conv2d[conv1]/input.195
        self.module_78 = py_nndct.nn.ReLU(inplace=True) #ResNet::ResNet/Sequential[layer3]/Bottleneck[3]/ReLU[relu1]/input.199
        self.module_79 = py_nndct.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #ResNet::ResNet/Sequential[layer3]/Bottleneck[3]/Conv2d[conv2]/input.201
        self.module_80 = py_nndct.nn.ReLU(inplace=True) #ResNet::ResNet/Sequential[layer3]/Bottleneck[3]/ReLU[relu2]/input.205
        self.module_81 = py_nndct.nn.Conv2d(in_channels=256, out_channels=1024, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #ResNet::ResNet/Sequential[layer3]/Bottleneck[3]/Conv2d[conv3]/input.207
        self.module_82 = py_nndct.nn.Add() #ResNet::ResNet/Sequential[layer3]/Bottleneck[3]/Add[skip_add]/input.209
        self.module_83 = py_nndct.nn.ReLU(inplace=True) #ResNet::ResNet/Sequential[layer3]/Bottleneck[3]/ReLU[relu3]/input.211
        self.module_84 = py_nndct.nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #ResNet::ResNet/Sequential[layer3]/Bottleneck[4]/Conv2d[conv1]/input.213
        self.module_85 = py_nndct.nn.ReLU(inplace=True) #ResNet::ResNet/Sequential[layer3]/Bottleneck[4]/ReLU[relu1]/input.217
        self.module_86 = py_nndct.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #ResNet::ResNet/Sequential[layer3]/Bottleneck[4]/Conv2d[conv2]/input.219
        self.module_87 = py_nndct.nn.ReLU(inplace=True) #ResNet::ResNet/Sequential[layer3]/Bottleneck[4]/ReLU[relu2]/input.223
        self.module_88 = py_nndct.nn.Conv2d(in_channels=256, out_channels=1024, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #ResNet::ResNet/Sequential[layer3]/Bottleneck[4]/Conv2d[conv3]/input.225
        self.module_89 = py_nndct.nn.Add() #ResNet::ResNet/Sequential[layer3]/Bottleneck[4]/Add[skip_add]/input.227
        self.module_90 = py_nndct.nn.ReLU(inplace=True) #ResNet::ResNet/Sequential[layer3]/Bottleneck[4]/ReLU[relu3]/input.229
        self.module_91 = py_nndct.nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #ResNet::ResNet/Sequential[layer3]/Bottleneck[5]/Conv2d[conv1]/input.231
        self.module_92 = py_nndct.nn.ReLU(inplace=True) #ResNet::ResNet/Sequential[layer3]/Bottleneck[5]/ReLU[relu1]/input.235
        self.module_93 = py_nndct.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #ResNet::ResNet/Sequential[layer3]/Bottleneck[5]/Conv2d[conv2]/input.237
        self.module_94 = py_nndct.nn.ReLU(inplace=True) #ResNet::ResNet/Sequential[layer3]/Bottleneck[5]/ReLU[relu2]/input.241
        self.module_95 = py_nndct.nn.Conv2d(in_channels=256, out_channels=1024, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #ResNet::ResNet/Sequential[layer3]/Bottleneck[5]/Conv2d[conv3]/input.243
        self.module_96 = py_nndct.nn.Add() #ResNet::ResNet/Sequential[layer3]/Bottleneck[5]/Add[skip_add]/input.245
        self.module_97 = py_nndct.nn.ReLU(inplace=True) #ResNet::ResNet/Sequential[layer3]/Bottleneck[5]/ReLU[relu3]/input.247
        self.module_98 = py_nndct.nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #ResNet::ResNet/Sequential[layer4]/Bottleneck[0]/Conv2d[conv1]/input.249
        self.module_99 = py_nndct.nn.ReLU(inplace=True) #ResNet::ResNet/Sequential[layer4]/Bottleneck[0]/ReLU[relu1]/input.253
        self.module_100 = py_nndct.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=[3, 3], stride=[2, 2], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #ResNet::ResNet/Sequential[layer4]/Bottleneck[0]/Conv2d[conv2]/input.255
        self.module_101 = py_nndct.nn.ReLU(inplace=True) #ResNet::ResNet/Sequential[layer4]/Bottleneck[0]/ReLU[relu2]/input.259
        self.module_102 = py_nndct.nn.Conv2d(in_channels=512, out_channels=2048, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #ResNet::ResNet/Sequential[layer4]/Bottleneck[0]/Conv2d[conv3]/input.261
        self.module_103 = py_nndct.nn.Conv2d(in_channels=1024, out_channels=2048, kernel_size=[1, 1], stride=[2, 2], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #ResNet::ResNet/Sequential[layer4]/Bottleneck[0]/Sequential[downsample]/Conv2d[0]/input.263
        self.module_104 = py_nndct.nn.Add() #ResNet::ResNet/Sequential[layer4]/Bottleneck[0]/Add[skip_add]/input.265
        self.module_105 = py_nndct.nn.ReLU(inplace=True) #ResNet::ResNet/Sequential[layer4]/Bottleneck[0]/ReLU[relu3]/input.267
        self.module_106 = py_nndct.nn.Conv2d(in_channels=2048, out_channels=512, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #ResNet::ResNet/Sequential[layer4]/Bottleneck[1]/Conv2d[conv1]/input.269
        self.module_107 = py_nndct.nn.ReLU(inplace=True) #ResNet::ResNet/Sequential[layer4]/Bottleneck[1]/ReLU[relu1]/input.273
        self.module_108 = py_nndct.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #ResNet::ResNet/Sequential[layer4]/Bottleneck[1]/Conv2d[conv2]/input.275
        self.module_109 = py_nndct.nn.ReLU(inplace=True) #ResNet::ResNet/Sequential[layer4]/Bottleneck[1]/ReLU[relu2]/input.279
        self.module_110 = py_nndct.nn.Conv2d(in_channels=512, out_channels=2048, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #ResNet::ResNet/Sequential[layer4]/Bottleneck[1]/Conv2d[conv3]/input.281
        self.module_111 = py_nndct.nn.Add() #ResNet::ResNet/Sequential[layer4]/Bottleneck[1]/Add[skip_add]/input.283
        self.module_112 = py_nndct.nn.ReLU(inplace=True) #ResNet::ResNet/Sequential[layer4]/Bottleneck[1]/ReLU[relu3]/input.285
        self.module_113 = py_nndct.nn.Conv2d(in_channels=2048, out_channels=512, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #ResNet::ResNet/Sequential[layer4]/Bottleneck[2]/Conv2d[conv1]/input.287
        self.module_114 = py_nndct.nn.ReLU(inplace=True) #ResNet::ResNet/Sequential[layer4]/Bottleneck[2]/ReLU[relu1]/input.291
        self.module_115 = py_nndct.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #ResNet::ResNet/Sequential[layer4]/Bottleneck[2]/Conv2d[conv2]/input.293
        self.module_116 = py_nndct.nn.ReLU(inplace=True) #ResNet::ResNet/Sequential[layer4]/Bottleneck[2]/ReLU[relu2]/input.297
        self.module_117 = py_nndct.nn.Conv2d(in_channels=512, out_channels=2048, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #ResNet::ResNet/Sequential[layer4]/Bottleneck[2]/Conv2d[conv3]/input.299
        self.module_118 = py_nndct.nn.Add() #ResNet::ResNet/Sequential[layer4]/Bottleneck[2]/Add[skip_add]/input.301
        self.module_119 = py_nndct.nn.ReLU(inplace=True) #ResNet::ResNet/Sequential[layer4]/Bottleneck[2]/ReLU[relu3]/input.303
        self.module_120 = py_nndct.nn.AdaptiveAvgPool2d(output_size=1) #ResNet::ResNet/AdaptiveAvgPool2d[avgpool]/8308
        self.module_121 = py_nndct.nn.Module('flatten') #ResNet::ResNet/input
        self.module_122 = py_nndct.nn.Linear(in_features=2048, out_features=100, bias=True) #ResNet::ResNet/Linear[fc]/inputs
        self.module_123 = py_nndct.nn.dequant_output() #ResNet::ResNet/DeQuantStub[dequant_stub]/8313

    def forward(self, *args):
        output_module_0 = self.module_0(input=args[0])
        output_module_0 = self.module_1(input=output_module_0)
        output_module_0 = self.module_2(output_module_0)
        output_module_0 = self.module_3(output_module_0)
        output_module_4 = self.module_4(output_module_0)
        output_module_4 = self.module_5(output_module_4)
        output_module_4 = self.module_6(output_module_4)
        output_module_4 = self.module_7(output_module_4)
        output_module_4 = self.module_8(output_module_4)
        output_module_9 = self.module_9(output_module_0)
        output_module_4 = self.module_10(input=output_module_4, other=output_module_9, alpha=1)
        output_module_4 = self.module_11(output_module_4)
        output_module_12 = self.module_12(output_module_4)
        output_module_12 = self.module_13(output_module_12)
        output_module_12 = self.module_14(output_module_12)
        output_module_12 = self.module_15(output_module_12)
        output_module_12 = self.module_16(output_module_12)
        output_module_12 = self.module_17(input=output_module_12, other=output_module_4, alpha=1)
        output_module_12 = self.module_18(output_module_12)
        output_module_19 = self.module_19(output_module_12)
        output_module_19 = self.module_20(output_module_19)
        output_module_19 = self.module_21(output_module_19)
        output_module_19 = self.module_22(output_module_19)
        output_module_19 = self.module_23(output_module_19)
        output_module_19 = self.module_24(input=output_module_19, other=output_module_12, alpha=1)
        output_module_19 = self.module_25(output_module_19)
        output_module_26 = self.module_26(output_module_19)
        output_module_26 = self.module_27(output_module_26)
        output_module_26 = self.module_28(output_module_26)
        output_module_26 = self.module_29(output_module_26)
        output_module_26 = self.module_30(output_module_26)
        output_module_31 = self.module_31(output_module_19)
        output_module_26 = self.module_32(input=output_module_26, other=output_module_31, alpha=1)
        output_module_26 = self.module_33(output_module_26)
        output_module_34 = self.module_34(output_module_26)
        output_module_34 = self.module_35(output_module_34)
        output_module_34 = self.module_36(output_module_34)
        output_module_34 = self.module_37(output_module_34)
        output_module_34 = self.module_38(output_module_34)
        output_module_34 = self.module_39(input=output_module_34, other=output_module_26, alpha=1)
        output_module_34 = self.module_40(output_module_34)
        output_module_41 = self.module_41(output_module_34)
        output_module_41 = self.module_42(output_module_41)
        output_module_41 = self.module_43(output_module_41)
        output_module_41 = self.module_44(output_module_41)
        output_module_41 = self.module_45(output_module_41)
        output_module_41 = self.module_46(input=output_module_41, other=output_module_34, alpha=1)
        output_module_41 = self.module_47(output_module_41)
        output_module_48 = self.module_48(output_module_41)
        output_module_48 = self.module_49(output_module_48)
        output_module_48 = self.module_50(output_module_48)
        output_module_48 = self.module_51(output_module_48)
        output_module_48 = self.module_52(output_module_48)
        output_module_48 = self.module_53(input=output_module_48, other=output_module_41, alpha=1)
        output_module_48 = self.module_54(output_module_48)
        output_module_55 = self.module_55(output_module_48)
        output_module_55 = self.module_56(output_module_55)
        output_module_55 = self.module_57(output_module_55)
        output_module_55 = self.module_58(output_module_55)
        output_module_55 = self.module_59(output_module_55)
        output_module_60 = self.module_60(output_module_48)
        output_module_55 = self.module_61(input=output_module_55, other=output_module_60, alpha=1)
        output_module_55 = self.module_62(output_module_55)
        output_module_63 = self.module_63(output_module_55)
        output_module_63 = self.module_64(output_module_63)
        output_module_63 = self.module_65(output_module_63)
        output_module_63 = self.module_66(output_module_63)
        output_module_63 = self.module_67(output_module_63)
        output_module_63 = self.module_68(input=output_module_63, other=output_module_55, alpha=1)
        output_module_63 = self.module_69(output_module_63)
        output_module_70 = self.module_70(output_module_63)
        output_module_70 = self.module_71(output_module_70)
        output_module_70 = self.module_72(output_module_70)
        output_module_70 = self.module_73(output_module_70)
        output_module_70 = self.module_74(output_module_70)
        output_module_70 = self.module_75(input=output_module_70, other=output_module_63, alpha=1)
        output_module_70 = self.module_76(output_module_70)
        output_module_77 = self.module_77(output_module_70)
        output_module_77 = self.module_78(output_module_77)
        output_module_77 = self.module_79(output_module_77)
        output_module_77 = self.module_80(output_module_77)
        output_module_77 = self.module_81(output_module_77)
        output_module_77 = self.module_82(input=output_module_77, other=output_module_70, alpha=1)
        output_module_77 = self.module_83(output_module_77)
        output_module_84 = self.module_84(output_module_77)
        output_module_84 = self.module_85(output_module_84)
        output_module_84 = self.module_86(output_module_84)
        output_module_84 = self.module_87(output_module_84)
        output_module_84 = self.module_88(output_module_84)
        output_module_84 = self.module_89(input=output_module_84, other=output_module_77, alpha=1)
        output_module_84 = self.module_90(output_module_84)
        output_module_91 = self.module_91(output_module_84)
        output_module_91 = self.module_92(output_module_91)
        output_module_91 = self.module_93(output_module_91)
        output_module_91 = self.module_94(output_module_91)
        output_module_91 = self.module_95(output_module_91)
        output_module_91 = self.module_96(input=output_module_91, other=output_module_84, alpha=1)
        output_module_91 = self.module_97(output_module_91)
        output_module_98 = self.module_98(output_module_91)
        output_module_98 = self.module_99(output_module_98)
        output_module_98 = self.module_100(output_module_98)
        output_module_98 = self.module_101(output_module_98)
        output_module_98 = self.module_102(output_module_98)
        output_module_103 = self.module_103(output_module_91)
        output_module_98 = self.module_104(input=output_module_98, other=output_module_103, alpha=1)
        output_module_98 = self.module_105(output_module_98)
        output_module_106 = self.module_106(output_module_98)
        output_module_106 = self.module_107(output_module_106)
        output_module_106 = self.module_108(output_module_106)
        output_module_106 = self.module_109(output_module_106)
        output_module_106 = self.module_110(output_module_106)
        output_module_106 = self.module_111(input=output_module_106, other=output_module_98, alpha=1)
        output_module_106 = self.module_112(output_module_106)
        output_module_113 = self.module_113(output_module_106)
        output_module_113 = self.module_114(output_module_113)
        output_module_113 = self.module_115(output_module_113)
        output_module_113 = self.module_116(output_module_113)
        output_module_113 = self.module_117(output_module_113)
        output_module_113 = self.module_118(input=output_module_113, other=output_module_106, alpha=1)
        output_module_113 = self.module_119(output_module_113)
        output_module_113 = self.module_120(output_module_113)
        output_module_113 = self.module_121(input=output_module_113, start_dim=1, end_dim=3)
        output_module_113 = self.module_122(output_module_113)
        output_module_113 = self.module_123(input=output_module_113)
        return output_module_113
