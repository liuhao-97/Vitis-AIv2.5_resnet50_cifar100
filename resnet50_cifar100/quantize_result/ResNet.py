# GENETARED BY NNDCT, DO NOT EDIT!

import torch
import pytorch_nndct as py_nndct
class ResNet(torch.nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        self.module_0 = py_nndct.nn.Input() #ResNet::input_0
        self.module_1 = py_nndct.nn.Conv2d(in_channels=3, out_channels=64, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #ResNet::ResNet/Sequential[conv1]/Conv2d[0]/input.3
        self.module_2 = py_nndct.nn.ReLU(inplace=True) #ResNet::ResNet/Sequential[conv1]/ReLU[2]/input.7
        self.module_3 = py_nndct.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #ResNet::ResNet/Sequential[conv2_x]/BottleNeck[0]/Sequential[residual_function]/Conv2d[0]/input.9
        self.module_4 = py_nndct.nn.ReLU(inplace=True) #ResNet::ResNet/Sequential[conv2_x]/BottleNeck[0]/Sequential[residual_function]/ReLU[2]/input.13
        self.module_5 = py_nndct.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #ResNet::ResNet/Sequential[conv2_x]/BottleNeck[0]/Sequential[residual_function]/Conv2d[3]/input.15
        self.module_6 = py_nndct.nn.ReLU(inplace=True) #ResNet::ResNet/Sequential[conv2_x]/BottleNeck[0]/Sequential[residual_function]/ReLU[5]/input.19
        self.module_7 = py_nndct.nn.Conv2d(in_channels=64, out_channels=256, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #ResNet::ResNet/Sequential[conv2_x]/BottleNeck[0]/Sequential[residual_function]/Conv2d[6]/input.21
        self.module_8 = py_nndct.nn.Conv2d(in_channels=64, out_channels=256, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #ResNet::ResNet/Sequential[conv2_x]/BottleNeck[0]/Sequential[shortcut]/Conv2d[0]/input.23
        self.module_9 = py_nndct.nn.Add() #ResNet::ResNet/Sequential[conv2_x]/BottleNeck[0]/input.25
        self.module_10 = py_nndct.nn.ReLU(inplace=True) #ResNet::ResNet/Sequential[conv2_x]/BottleNeck[0]/input.27
        self.module_11 = py_nndct.nn.Conv2d(in_channels=256, out_channels=64, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #ResNet::ResNet/Sequential[conv2_x]/BottleNeck[1]/Sequential[residual_function]/Conv2d[0]/input.29
        self.module_12 = py_nndct.nn.ReLU(inplace=True) #ResNet::ResNet/Sequential[conv2_x]/BottleNeck[1]/Sequential[residual_function]/ReLU[2]/input.33
        self.module_13 = py_nndct.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #ResNet::ResNet/Sequential[conv2_x]/BottleNeck[1]/Sequential[residual_function]/Conv2d[3]/input.35
        self.module_14 = py_nndct.nn.ReLU(inplace=True) #ResNet::ResNet/Sequential[conv2_x]/BottleNeck[1]/Sequential[residual_function]/ReLU[5]/input.39
        self.module_15 = py_nndct.nn.Conv2d(in_channels=64, out_channels=256, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #ResNet::ResNet/Sequential[conv2_x]/BottleNeck[1]/Sequential[residual_function]/Conv2d[6]/input.41
        self.module_16 = py_nndct.nn.Add() #ResNet::ResNet/Sequential[conv2_x]/BottleNeck[1]/input.43
        self.module_17 = py_nndct.nn.ReLU(inplace=True) #ResNet::ResNet/Sequential[conv2_x]/BottleNeck[1]/input.45
        self.module_18 = py_nndct.nn.Conv2d(in_channels=256, out_channels=64, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #ResNet::ResNet/Sequential[conv2_x]/BottleNeck[2]/Sequential[residual_function]/Conv2d[0]/input.47
        self.module_19 = py_nndct.nn.ReLU(inplace=True) #ResNet::ResNet/Sequential[conv2_x]/BottleNeck[2]/Sequential[residual_function]/ReLU[2]/input.51
        self.module_20 = py_nndct.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #ResNet::ResNet/Sequential[conv2_x]/BottleNeck[2]/Sequential[residual_function]/Conv2d[3]/input.53
        self.module_21 = py_nndct.nn.ReLU(inplace=True) #ResNet::ResNet/Sequential[conv2_x]/BottleNeck[2]/Sequential[residual_function]/ReLU[5]/input.57
        self.module_22 = py_nndct.nn.Conv2d(in_channels=64, out_channels=256, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #ResNet::ResNet/Sequential[conv2_x]/BottleNeck[2]/Sequential[residual_function]/Conv2d[6]/input.59
        self.module_23 = py_nndct.nn.Add() #ResNet::ResNet/Sequential[conv2_x]/BottleNeck[2]/input.61
        self.module_24 = py_nndct.nn.ReLU(inplace=True) #ResNet::ResNet/Sequential[conv2_x]/BottleNeck[2]/input.63
        self.module_25 = py_nndct.nn.Conv2d(in_channels=256, out_channels=128, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #ResNet::ResNet/Sequential[conv3_x]/BottleNeck[0]/Sequential[residual_function]/Conv2d[0]/input.65
        self.module_26 = py_nndct.nn.ReLU(inplace=True) #ResNet::ResNet/Sequential[conv3_x]/BottleNeck[0]/Sequential[residual_function]/ReLU[2]/input.69
        self.module_27 = py_nndct.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=[3, 3], stride=[2, 2], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #ResNet::ResNet/Sequential[conv3_x]/BottleNeck[0]/Sequential[residual_function]/Conv2d[3]/input.71
        self.module_28 = py_nndct.nn.ReLU(inplace=True) #ResNet::ResNet/Sequential[conv3_x]/BottleNeck[0]/Sequential[residual_function]/ReLU[5]/input.75
        self.module_29 = py_nndct.nn.Conv2d(in_channels=128, out_channels=512, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #ResNet::ResNet/Sequential[conv3_x]/BottleNeck[0]/Sequential[residual_function]/Conv2d[6]/input.77
        self.module_30 = py_nndct.nn.Conv2d(in_channels=256, out_channels=512, kernel_size=[1, 1], stride=[2, 2], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #ResNet::ResNet/Sequential[conv3_x]/BottleNeck[0]/Sequential[shortcut]/Conv2d[0]/input.79
        self.module_31 = py_nndct.nn.Add() #ResNet::ResNet/Sequential[conv3_x]/BottleNeck[0]/input.81
        self.module_32 = py_nndct.nn.ReLU(inplace=True) #ResNet::ResNet/Sequential[conv3_x]/BottleNeck[0]/input.83
        self.module_33 = py_nndct.nn.Conv2d(in_channels=512, out_channels=128, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #ResNet::ResNet/Sequential[conv3_x]/BottleNeck[1]/Sequential[residual_function]/Conv2d[0]/input.85
        self.module_34 = py_nndct.nn.ReLU(inplace=True) #ResNet::ResNet/Sequential[conv3_x]/BottleNeck[1]/Sequential[residual_function]/ReLU[2]/input.89
        self.module_35 = py_nndct.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #ResNet::ResNet/Sequential[conv3_x]/BottleNeck[1]/Sequential[residual_function]/Conv2d[3]/input.91
        self.module_36 = py_nndct.nn.ReLU(inplace=True) #ResNet::ResNet/Sequential[conv3_x]/BottleNeck[1]/Sequential[residual_function]/ReLU[5]/input.95
        self.module_37 = py_nndct.nn.Conv2d(in_channels=128, out_channels=512, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #ResNet::ResNet/Sequential[conv3_x]/BottleNeck[1]/Sequential[residual_function]/Conv2d[6]/input.97
        self.module_38 = py_nndct.nn.Add() #ResNet::ResNet/Sequential[conv3_x]/BottleNeck[1]/input.99
        self.module_39 = py_nndct.nn.ReLU(inplace=True) #ResNet::ResNet/Sequential[conv3_x]/BottleNeck[1]/input.101
        self.module_40 = py_nndct.nn.Conv2d(in_channels=512, out_channels=128, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #ResNet::ResNet/Sequential[conv3_x]/BottleNeck[2]/Sequential[residual_function]/Conv2d[0]/input.103
        self.module_41 = py_nndct.nn.ReLU(inplace=True) #ResNet::ResNet/Sequential[conv3_x]/BottleNeck[2]/Sequential[residual_function]/ReLU[2]/input.107
        self.module_42 = py_nndct.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #ResNet::ResNet/Sequential[conv3_x]/BottleNeck[2]/Sequential[residual_function]/Conv2d[3]/input.109
        self.module_43 = py_nndct.nn.ReLU(inplace=True) #ResNet::ResNet/Sequential[conv3_x]/BottleNeck[2]/Sequential[residual_function]/ReLU[5]/input.113
        self.module_44 = py_nndct.nn.Conv2d(in_channels=128, out_channels=512, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #ResNet::ResNet/Sequential[conv3_x]/BottleNeck[2]/Sequential[residual_function]/Conv2d[6]/input.115
        self.module_45 = py_nndct.nn.Add() #ResNet::ResNet/Sequential[conv3_x]/BottleNeck[2]/input.117
        self.module_46 = py_nndct.nn.ReLU(inplace=True) #ResNet::ResNet/Sequential[conv3_x]/BottleNeck[2]/input.119
        self.module_47 = py_nndct.nn.Conv2d(in_channels=512, out_channels=128, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #ResNet::ResNet/Sequential[conv3_x]/BottleNeck[3]/Sequential[residual_function]/Conv2d[0]/input.121
        self.module_48 = py_nndct.nn.ReLU(inplace=True) #ResNet::ResNet/Sequential[conv3_x]/BottleNeck[3]/Sequential[residual_function]/ReLU[2]/input.125
        self.module_49 = py_nndct.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #ResNet::ResNet/Sequential[conv3_x]/BottleNeck[3]/Sequential[residual_function]/Conv2d[3]/input.127
        self.module_50 = py_nndct.nn.ReLU(inplace=True) #ResNet::ResNet/Sequential[conv3_x]/BottleNeck[3]/Sequential[residual_function]/ReLU[5]/input.131
        self.module_51 = py_nndct.nn.Conv2d(in_channels=128, out_channels=512, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #ResNet::ResNet/Sequential[conv3_x]/BottleNeck[3]/Sequential[residual_function]/Conv2d[6]/input.133
        self.module_52 = py_nndct.nn.Add() #ResNet::ResNet/Sequential[conv3_x]/BottleNeck[3]/input.135
        self.module_53 = py_nndct.nn.ReLU(inplace=True) #ResNet::ResNet/Sequential[conv3_x]/BottleNeck[3]/input.137
        self.module_54 = py_nndct.nn.Conv2d(in_channels=512, out_channels=256, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #ResNet::ResNet/Sequential[conv4_x]/BottleNeck[0]/Sequential[residual_function]/Conv2d[0]/input.139
        self.module_55 = py_nndct.nn.ReLU(inplace=True) #ResNet::ResNet/Sequential[conv4_x]/BottleNeck[0]/Sequential[residual_function]/ReLU[2]/input.143
        self.module_56 = py_nndct.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=[3, 3], stride=[2, 2], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #ResNet::ResNet/Sequential[conv4_x]/BottleNeck[0]/Sequential[residual_function]/Conv2d[3]/input.145
        self.module_57 = py_nndct.nn.ReLU(inplace=True) #ResNet::ResNet/Sequential[conv4_x]/BottleNeck[0]/Sequential[residual_function]/ReLU[5]/input.149
        self.module_58 = py_nndct.nn.Conv2d(in_channels=256, out_channels=1024, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #ResNet::ResNet/Sequential[conv4_x]/BottleNeck[0]/Sequential[residual_function]/Conv2d[6]/input.151
        self.module_59 = py_nndct.nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=[1, 1], stride=[2, 2], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #ResNet::ResNet/Sequential[conv4_x]/BottleNeck[0]/Sequential[shortcut]/Conv2d[0]/input.153
        self.module_60 = py_nndct.nn.Add() #ResNet::ResNet/Sequential[conv4_x]/BottleNeck[0]/input.155
        self.module_61 = py_nndct.nn.ReLU(inplace=True) #ResNet::ResNet/Sequential[conv4_x]/BottleNeck[0]/input.157
        self.module_62 = py_nndct.nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #ResNet::ResNet/Sequential[conv4_x]/BottleNeck[1]/Sequential[residual_function]/Conv2d[0]/input.159
        self.module_63 = py_nndct.nn.ReLU(inplace=True) #ResNet::ResNet/Sequential[conv4_x]/BottleNeck[1]/Sequential[residual_function]/ReLU[2]/input.163
        self.module_64 = py_nndct.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #ResNet::ResNet/Sequential[conv4_x]/BottleNeck[1]/Sequential[residual_function]/Conv2d[3]/input.165
        self.module_65 = py_nndct.nn.ReLU(inplace=True) #ResNet::ResNet/Sequential[conv4_x]/BottleNeck[1]/Sequential[residual_function]/ReLU[5]/input.169
        self.module_66 = py_nndct.nn.Conv2d(in_channels=256, out_channels=1024, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #ResNet::ResNet/Sequential[conv4_x]/BottleNeck[1]/Sequential[residual_function]/Conv2d[6]/input.171
        self.module_67 = py_nndct.nn.Add() #ResNet::ResNet/Sequential[conv4_x]/BottleNeck[1]/input.173
        self.module_68 = py_nndct.nn.ReLU(inplace=True) #ResNet::ResNet/Sequential[conv4_x]/BottleNeck[1]/input.175
        self.module_69 = py_nndct.nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #ResNet::ResNet/Sequential[conv4_x]/BottleNeck[2]/Sequential[residual_function]/Conv2d[0]/input.177
        self.module_70 = py_nndct.nn.ReLU(inplace=True) #ResNet::ResNet/Sequential[conv4_x]/BottleNeck[2]/Sequential[residual_function]/ReLU[2]/input.181
        self.module_71 = py_nndct.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #ResNet::ResNet/Sequential[conv4_x]/BottleNeck[2]/Sequential[residual_function]/Conv2d[3]/input.183
        self.module_72 = py_nndct.nn.ReLU(inplace=True) #ResNet::ResNet/Sequential[conv4_x]/BottleNeck[2]/Sequential[residual_function]/ReLU[5]/input.187
        self.module_73 = py_nndct.nn.Conv2d(in_channels=256, out_channels=1024, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #ResNet::ResNet/Sequential[conv4_x]/BottleNeck[2]/Sequential[residual_function]/Conv2d[6]/input.189
        self.module_74 = py_nndct.nn.Add() #ResNet::ResNet/Sequential[conv4_x]/BottleNeck[2]/input.191
        self.module_75 = py_nndct.nn.ReLU(inplace=True) #ResNet::ResNet/Sequential[conv4_x]/BottleNeck[2]/input.193
        self.module_76 = py_nndct.nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #ResNet::ResNet/Sequential[conv4_x]/BottleNeck[3]/Sequential[residual_function]/Conv2d[0]/input.195
        self.module_77 = py_nndct.nn.ReLU(inplace=True) #ResNet::ResNet/Sequential[conv4_x]/BottleNeck[3]/Sequential[residual_function]/ReLU[2]/input.199
        self.module_78 = py_nndct.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #ResNet::ResNet/Sequential[conv4_x]/BottleNeck[3]/Sequential[residual_function]/Conv2d[3]/input.201
        self.module_79 = py_nndct.nn.ReLU(inplace=True) #ResNet::ResNet/Sequential[conv4_x]/BottleNeck[3]/Sequential[residual_function]/ReLU[5]/input.205
        self.module_80 = py_nndct.nn.Conv2d(in_channels=256, out_channels=1024, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #ResNet::ResNet/Sequential[conv4_x]/BottleNeck[3]/Sequential[residual_function]/Conv2d[6]/input.207
        self.module_81 = py_nndct.nn.Add() #ResNet::ResNet/Sequential[conv4_x]/BottleNeck[3]/input.209
        self.module_82 = py_nndct.nn.ReLU(inplace=True) #ResNet::ResNet/Sequential[conv4_x]/BottleNeck[3]/input.211
        self.module_83 = py_nndct.nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #ResNet::ResNet/Sequential[conv4_x]/BottleNeck[4]/Sequential[residual_function]/Conv2d[0]/input.213
        self.module_84 = py_nndct.nn.ReLU(inplace=True) #ResNet::ResNet/Sequential[conv4_x]/BottleNeck[4]/Sequential[residual_function]/ReLU[2]/input.217
        self.module_85 = py_nndct.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #ResNet::ResNet/Sequential[conv4_x]/BottleNeck[4]/Sequential[residual_function]/Conv2d[3]/input.219
        self.module_86 = py_nndct.nn.ReLU(inplace=True) #ResNet::ResNet/Sequential[conv4_x]/BottleNeck[4]/Sequential[residual_function]/ReLU[5]/input.223
        self.module_87 = py_nndct.nn.Conv2d(in_channels=256, out_channels=1024, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #ResNet::ResNet/Sequential[conv4_x]/BottleNeck[4]/Sequential[residual_function]/Conv2d[6]/input.225
        self.module_88 = py_nndct.nn.Add() #ResNet::ResNet/Sequential[conv4_x]/BottleNeck[4]/input.227
        self.module_89 = py_nndct.nn.ReLU(inplace=True) #ResNet::ResNet/Sequential[conv4_x]/BottleNeck[4]/input.229
        self.module_90 = py_nndct.nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #ResNet::ResNet/Sequential[conv4_x]/BottleNeck[5]/Sequential[residual_function]/Conv2d[0]/input.231
        self.module_91 = py_nndct.nn.ReLU(inplace=True) #ResNet::ResNet/Sequential[conv4_x]/BottleNeck[5]/Sequential[residual_function]/ReLU[2]/input.235
        self.module_92 = py_nndct.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #ResNet::ResNet/Sequential[conv4_x]/BottleNeck[5]/Sequential[residual_function]/Conv2d[3]/input.237
        self.module_93 = py_nndct.nn.ReLU(inplace=True) #ResNet::ResNet/Sequential[conv4_x]/BottleNeck[5]/Sequential[residual_function]/ReLU[5]/input.241
        self.module_94 = py_nndct.nn.Conv2d(in_channels=256, out_channels=1024, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #ResNet::ResNet/Sequential[conv4_x]/BottleNeck[5]/Sequential[residual_function]/Conv2d[6]/input.243
        self.module_95 = py_nndct.nn.Add() #ResNet::ResNet/Sequential[conv4_x]/BottleNeck[5]/input.245
        self.module_96 = py_nndct.nn.ReLU(inplace=True) #ResNet::ResNet/Sequential[conv4_x]/BottleNeck[5]/input.247
        self.module_97 = py_nndct.nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #ResNet::ResNet/Sequential[conv5_x]/BottleNeck[0]/Sequential[residual_function]/Conv2d[0]/input.249
        self.module_98 = py_nndct.nn.ReLU(inplace=True) #ResNet::ResNet/Sequential[conv5_x]/BottleNeck[0]/Sequential[residual_function]/ReLU[2]/input.253
        self.module_99 = py_nndct.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=[3, 3], stride=[2, 2], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #ResNet::ResNet/Sequential[conv5_x]/BottleNeck[0]/Sequential[residual_function]/Conv2d[3]/input.255
        self.module_100 = py_nndct.nn.ReLU(inplace=True) #ResNet::ResNet/Sequential[conv5_x]/BottleNeck[0]/Sequential[residual_function]/ReLU[5]/input.259
        self.module_101 = py_nndct.nn.Conv2d(in_channels=512, out_channels=2048, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #ResNet::ResNet/Sequential[conv5_x]/BottleNeck[0]/Sequential[residual_function]/Conv2d[6]/input.261
        self.module_102 = py_nndct.nn.Conv2d(in_channels=1024, out_channels=2048, kernel_size=[1, 1], stride=[2, 2], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #ResNet::ResNet/Sequential[conv5_x]/BottleNeck[0]/Sequential[shortcut]/Conv2d[0]/input.263
        self.module_103 = py_nndct.nn.Add() #ResNet::ResNet/Sequential[conv5_x]/BottleNeck[0]/input.265
        self.module_104 = py_nndct.nn.ReLU(inplace=True) #ResNet::ResNet/Sequential[conv5_x]/BottleNeck[0]/input.267
        self.module_105 = py_nndct.nn.Conv2d(in_channels=2048, out_channels=512, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #ResNet::ResNet/Sequential[conv5_x]/BottleNeck[1]/Sequential[residual_function]/Conv2d[0]/input.269
        self.module_106 = py_nndct.nn.ReLU(inplace=True) #ResNet::ResNet/Sequential[conv5_x]/BottleNeck[1]/Sequential[residual_function]/ReLU[2]/input.273
        self.module_107 = py_nndct.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #ResNet::ResNet/Sequential[conv5_x]/BottleNeck[1]/Sequential[residual_function]/Conv2d[3]/input.275
        self.module_108 = py_nndct.nn.ReLU(inplace=True) #ResNet::ResNet/Sequential[conv5_x]/BottleNeck[1]/Sequential[residual_function]/ReLU[5]/input.279
        self.module_109 = py_nndct.nn.Conv2d(in_channels=512, out_channels=2048, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #ResNet::ResNet/Sequential[conv5_x]/BottleNeck[1]/Sequential[residual_function]/Conv2d[6]/input.281
        self.module_110 = py_nndct.nn.Add() #ResNet::ResNet/Sequential[conv5_x]/BottleNeck[1]/input.283
        self.module_111 = py_nndct.nn.ReLU(inplace=True) #ResNet::ResNet/Sequential[conv5_x]/BottleNeck[1]/input.285
        self.module_112 = py_nndct.nn.Conv2d(in_channels=2048, out_channels=512, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #ResNet::ResNet/Sequential[conv5_x]/BottleNeck[2]/Sequential[residual_function]/Conv2d[0]/input.287
        self.module_113 = py_nndct.nn.ReLU(inplace=True) #ResNet::ResNet/Sequential[conv5_x]/BottleNeck[2]/Sequential[residual_function]/ReLU[2]/input.291
        self.module_114 = py_nndct.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #ResNet::ResNet/Sequential[conv5_x]/BottleNeck[2]/Sequential[residual_function]/Conv2d[3]/input.293
        self.module_115 = py_nndct.nn.ReLU(inplace=True) #ResNet::ResNet/Sequential[conv5_x]/BottleNeck[2]/Sequential[residual_function]/ReLU[5]/input.297
        self.module_116 = py_nndct.nn.Conv2d(in_channels=512, out_channels=2048, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #ResNet::ResNet/Sequential[conv5_x]/BottleNeck[2]/Sequential[residual_function]/Conv2d[6]/input.299
        self.module_117 = py_nndct.nn.Add() #ResNet::ResNet/Sequential[conv5_x]/BottleNeck[2]/input.301
        self.module_118 = py_nndct.nn.ReLU(inplace=True) #ResNet::ResNet/Sequential[conv5_x]/BottleNeck[2]/input.303
        self.module_119 = py_nndct.nn.AdaptiveAvgPool2d(output_size=1) #ResNet::ResNet/AdaptiveAvgPool2d[avg_pool]/9562
        self.module_120 = py_nndct.nn.Module('shape') #ResNet::ResNet/9564
        self.module_121 = py_nndct.nn.Module('reshape') #ResNet::ResNet/input
        self.module_122 = py_nndct.nn.Linear(in_features=2048, out_features=100, bias=True) #ResNet::ResNet/Linear[fc]/9570

    def forward(self, *args):
        output_module_0 = self.module_0(input=args[0])
        output_module_0 = self.module_1(output_module_0)
        output_module_0 = self.module_2(output_module_0)
        output_module_3 = self.module_3(output_module_0)
        output_module_3 = self.module_4(output_module_3)
        output_module_3 = self.module_5(output_module_3)
        output_module_3 = self.module_6(output_module_3)
        output_module_3 = self.module_7(output_module_3)
        output_module_8 = self.module_8(output_module_0)
        output_module_3 = self.module_9(input=output_module_3, other=output_module_8, alpha=1)
        output_module_3 = self.module_10(output_module_3)
        output_module_11 = self.module_11(output_module_3)
        output_module_11 = self.module_12(output_module_11)
        output_module_11 = self.module_13(output_module_11)
        output_module_11 = self.module_14(output_module_11)
        output_module_11 = self.module_15(output_module_11)
        output_module_11 = self.module_16(input=output_module_11, other=output_module_3, alpha=1)
        output_module_11 = self.module_17(output_module_11)
        output_module_18 = self.module_18(output_module_11)
        output_module_18 = self.module_19(output_module_18)
        output_module_18 = self.module_20(output_module_18)
        output_module_18 = self.module_21(output_module_18)
        output_module_18 = self.module_22(output_module_18)
        output_module_18 = self.module_23(input=output_module_18, other=output_module_11, alpha=1)
        output_module_18 = self.module_24(output_module_18)
        output_module_25 = self.module_25(output_module_18)
        output_module_25 = self.module_26(output_module_25)
        output_module_25 = self.module_27(output_module_25)
        output_module_25 = self.module_28(output_module_25)
        output_module_25 = self.module_29(output_module_25)
        output_module_30 = self.module_30(output_module_18)
        output_module_25 = self.module_31(input=output_module_25, other=output_module_30, alpha=1)
        output_module_25 = self.module_32(output_module_25)
        output_module_33 = self.module_33(output_module_25)
        output_module_33 = self.module_34(output_module_33)
        output_module_33 = self.module_35(output_module_33)
        output_module_33 = self.module_36(output_module_33)
        output_module_33 = self.module_37(output_module_33)
        output_module_33 = self.module_38(input=output_module_33, other=output_module_25, alpha=1)
        output_module_33 = self.module_39(output_module_33)
        output_module_40 = self.module_40(output_module_33)
        output_module_40 = self.module_41(output_module_40)
        output_module_40 = self.module_42(output_module_40)
        output_module_40 = self.module_43(output_module_40)
        output_module_40 = self.module_44(output_module_40)
        output_module_40 = self.module_45(input=output_module_40, other=output_module_33, alpha=1)
        output_module_40 = self.module_46(output_module_40)
        output_module_47 = self.module_47(output_module_40)
        output_module_47 = self.module_48(output_module_47)
        output_module_47 = self.module_49(output_module_47)
        output_module_47 = self.module_50(output_module_47)
        output_module_47 = self.module_51(output_module_47)
        output_module_47 = self.module_52(input=output_module_47, other=output_module_40, alpha=1)
        output_module_47 = self.module_53(output_module_47)
        output_module_54 = self.module_54(output_module_47)
        output_module_54 = self.module_55(output_module_54)
        output_module_54 = self.module_56(output_module_54)
        output_module_54 = self.module_57(output_module_54)
        output_module_54 = self.module_58(output_module_54)
        output_module_59 = self.module_59(output_module_47)
        output_module_54 = self.module_60(input=output_module_54, other=output_module_59, alpha=1)
        output_module_54 = self.module_61(output_module_54)
        output_module_62 = self.module_62(output_module_54)
        output_module_62 = self.module_63(output_module_62)
        output_module_62 = self.module_64(output_module_62)
        output_module_62 = self.module_65(output_module_62)
        output_module_62 = self.module_66(output_module_62)
        output_module_62 = self.module_67(input=output_module_62, other=output_module_54, alpha=1)
        output_module_62 = self.module_68(output_module_62)
        output_module_69 = self.module_69(output_module_62)
        output_module_69 = self.module_70(output_module_69)
        output_module_69 = self.module_71(output_module_69)
        output_module_69 = self.module_72(output_module_69)
        output_module_69 = self.module_73(output_module_69)
        output_module_69 = self.module_74(input=output_module_69, other=output_module_62, alpha=1)
        output_module_69 = self.module_75(output_module_69)
        output_module_76 = self.module_76(output_module_69)
        output_module_76 = self.module_77(output_module_76)
        output_module_76 = self.module_78(output_module_76)
        output_module_76 = self.module_79(output_module_76)
        output_module_76 = self.module_80(output_module_76)
        output_module_76 = self.module_81(input=output_module_76, other=output_module_69, alpha=1)
        output_module_76 = self.module_82(output_module_76)
        output_module_83 = self.module_83(output_module_76)
        output_module_83 = self.module_84(output_module_83)
        output_module_83 = self.module_85(output_module_83)
        output_module_83 = self.module_86(output_module_83)
        output_module_83 = self.module_87(output_module_83)
        output_module_83 = self.module_88(input=output_module_83, other=output_module_76, alpha=1)
        output_module_83 = self.module_89(output_module_83)
        output_module_90 = self.module_90(output_module_83)
        output_module_90 = self.module_91(output_module_90)
        output_module_90 = self.module_92(output_module_90)
        output_module_90 = self.module_93(output_module_90)
        output_module_90 = self.module_94(output_module_90)
        output_module_90 = self.module_95(input=output_module_90, other=output_module_83, alpha=1)
        output_module_90 = self.module_96(output_module_90)
        output_module_97 = self.module_97(output_module_90)
        output_module_97 = self.module_98(output_module_97)
        output_module_97 = self.module_99(output_module_97)
        output_module_97 = self.module_100(output_module_97)
        output_module_97 = self.module_101(output_module_97)
        output_module_102 = self.module_102(output_module_90)
        output_module_97 = self.module_103(input=output_module_97, other=output_module_102, alpha=1)
        output_module_97 = self.module_104(output_module_97)
        output_module_105 = self.module_105(output_module_97)
        output_module_105 = self.module_106(output_module_105)
        output_module_105 = self.module_107(output_module_105)
        output_module_105 = self.module_108(output_module_105)
        output_module_105 = self.module_109(output_module_105)
        output_module_105 = self.module_110(input=output_module_105, other=output_module_97, alpha=1)
        output_module_105 = self.module_111(output_module_105)
        output_module_112 = self.module_112(output_module_105)
        output_module_112 = self.module_113(output_module_112)
        output_module_112 = self.module_114(output_module_112)
        output_module_112 = self.module_115(output_module_112)
        output_module_112 = self.module_116(output_module_112)
        output_module_112 = self.module_117(input=output_module_112, other=output_module_105, alpha=1)
        output_module_112 = self.module_118(output_module_112)
        output_module_112 = self.module_119(output_module_112)
        output_module_120 = self.module_120(input=output_module_112, dim=0)
        output_module_121 = self.module_121(input=output_module_112, shape=[output_module_120,-1])
        output_module_121 = self.module_122(output_module_121)
        return output_module_121
