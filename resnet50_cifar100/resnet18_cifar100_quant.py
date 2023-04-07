# https://github.com/Xilinx/Vitis-AI/tree/2.5/src/RNN/rnn_quantizer/example
import pytorch_nndct
from pytorch_nndct.apis import torch_quantizer, dump_xmodel
from pytorch_nndct.apis import Inspector

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from torch import optim
from torchvision.utils import save_image
from torchvision.datasets import CIFAR100


import argparse
import os
import shutil
import time


import torch.optim
import torchvision.datasets as datasets


from pytorch_nndct import nn as nndct_nn
from pytorch_nndct.nn.modules import functional
from pytorch_nndct import QatProcessor

import sys
import argparse
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from torch.utils.data import DataLoader

import os
import re
import sys
import argparse
import time
import pdb
import random
from pytorch_nndct.apis import torch_quantizer
from conf import settings
from utils import (
    get_network,
    get_training_dataloader,
    get_test_dataloader,
    WarmUpLR,
    most_recent_folder,
    most_recent_weights,
    last_epoch,
    best_acc_weights,
)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 downsample=None,
                 groups=1,
                 base_width=64,
                 dilation=1,
                 norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError(
                'BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError(
                "Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

        self.skip_add = functional.Add()
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.skip_add(out, identity)
        # out = out+identity
        out = self.relu2(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 downsample=None,
                 groups=1,
                 base_width=64,
                 dilation=1,
                 norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu1 = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

        self.skip_add = functional.Add()
        self.relu2 = nn.ReLU(inplace=True)
        self.relu3 = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        # The original code was:
        # out += identity
        # Replace '+=' with Add module cause we want to quantize add op.
        out = self.skip_add(out, identity)
        out = self.relu3(out)

        return out


class ResNet(nn.Module):

    def __init__(self,
                 block,
                 layers,
                 num_classes=100,
                 zero_init_residual=False,
                 groups=1,
                 width_per_group=64,
                 replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group

        self.conv1 = nn.Conv2d(
            3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(
            block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(
            block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(
            block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        self.quant_stub = nndct_nn.QuantStub()
        self.dequant_stub = nndct_nn.DeQuantStub()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(self.inplanes, planes, stride, downsample, self.groups,
                  self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.quant_stub(x)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        x = self.dequant_stub(x)
        return x


def _resnet(arch, block, layers, pretrained, progress, **kwargs):
    model = ResNet(block, layers, **kwargs)
    if pretrained:
        model.load_state_dict(torch.load(args.pretrained))
    return model


def resnet18(pretrained=False, progress=True, **kwargs):
    r"""ResNet-18 model from
      `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>'_
      Args:
          pretrained (bool): If True, returns a model pre-trained on ImageNet
          progress (bool): If True, displays a progress bar of the download to stderr
      """
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress,
                   **kwargs)



class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions
      for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].flatten().float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def evaluate(model, val_loader, loss_fn):

    model.eval()
    model = model.to(device)
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    total = 0
    Loss = 0
    for iteraction, (images, labels) in tqdm(
            enumerate(val_loader), total=len(val_loader)):
        images = images.to(device)
        labels = labels.to(device)
        # pdb.set_trace()
        outputs = model(images)
        loss = loss_fn(outputs, labels)
        Loss += loss.item()
        total += images.size(0)
        acc1, acc5 = accuracy(outputs, labels, topk=(1, 5))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))
    return top1.avg, top5.avg, Loss / total


def quantization(title='optimize',
                 model_name='',
                 file_path=''):

    data_dir = args.data_dir
    quant_mode = args.quant_mode
    finetune = args.fast_finetune
    deploy = args.deploy
    batch_size = args.batch_size
    subset_len = args.subset_len
    inspect = args.inspect
    config_file = args.config_file
    if quant_mode != 'test' and deploy:
        deploy = False
        print(r'Warning: Exporting xmodel needs to be done in quantization test mode, turn off it in this running!')
    if deploy and (batch_size != 1 or subset_len != 1):
        print(r'Warning: Exporting xmodel needs batch size to be 1 and only 1 iteration of inference, change them automatically!')
        batch_size = 1
        subset_len = 1

    
    model=resnet18()
    model.load_state_dict(torch.load('resnet18-200-regular.pth'))

    input = torch.randn([32, 3, 32, 32])
    if quant_mode == 'float':
        quant_model = model
        if inspect:
            import sys
            from pytorch_nndct.apis import Inspector
            # create inspector
            # inspector = Inspector('0x20200000000002a')
            inspector = Inspector("DPUCAHX8H_ISA2")  # by name U55C
    
            # inspector = Inspector("0x603000b16013831") # by fingerprint
            # inspector = Inspector("DPUCAHX8L_ISA0_SP") # by target name

            
            # start to inspect
            inspector.inspect(quant_model, (input,), device=device,image_format='svg')
            sys.exit()
    else:
        # new api
        ####################################################################################
        quantizer = torch_quantizer(
            quant_mode, model, (input), device=device, quant_config_file=config_file)

        quant_model = quantizer.quant_model
        #####################################################################################

    # to get loss value after evaluation
    loss_fn = torch.nn.CrossEntropyLoss().to(device)

    # val_loader, _ = load_data(
    #     subset_len=subset_len,
    #     train=False,
    #     batch_size=batch_size,
    #     sample_method='random',
    #     data_dir=data_dir,
    #     model_name=model_name)
    test_dataset = CIFAR100(root='./data', train=False, transform=transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ]))

    test_loader = torch.utils.data.DataLoader(
        CIFAR100(root='./data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=100, shuffle=False, num_workers=4)
    val_loader = test_loader

    # fast finetune model or load finetuned parameter before test
    if finetune == True:
        ft_loader, _ = load_data(
            subset_len=5120,
            train=False,
            batch_size=batch_size,
            sample_method='random',
            data_dir=data_dir,
            model_name=model_name)
        if quant_mode == 'calib':
            quantizer.fast_finetune(
                evaluate, (quant_model, ft_loader, loss_fn))
        elif quant_mode == 'test':
            quantizer.load_ft_param()

    # record  modules float model accuracy
    # add modules float model accuracy here
    acc_org1 = 0.0
    acc_org5 = 0.0
    loss_org = 0.0

    # register_modification_hooks(model_gen, train=False)
    acc1_gen, acc5_gen, loss_gen = evaluate(quant_model, val_loader, loss_fn)

    # logging accuracy
    print('loss: %g' % (loss_gen))
    print('top-1 / top-5 accuracy: %g / %g' % (acc1_gen, acc5_gen))

    # handle quantization result
    if quant_mode == 'calib':
        quantizer.export_quant_config()
    if deploy:
        quantizer.export_xmodel(deploy_check=False)
        quantizer.export_onnx_model()


def evaluate(net, test_data):
    net.eval()
    with torch.no_grad():
        correct = 0
        test_num = test_data.dataset.data.shape[0]
        total_test_loss = []

        for i, data in enumerate(test_data):
            inputs, labels = data
            inputs = inputs.cuda()
            labels = labels.cuda()
            outputs = net(inputs)
            correct += torch.sum(torch.argmax(outputs, dim=1) == labels)

    net.train()
    return float(correct) / test_num


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--data_dir',
        default="/workspace/resnet50_cifar100/data/",
        help='Data set directory, when quant_mode=calib, it is for calibration, while quant_mode=test it is for evaluation')
    parser.add_argument(
        '--model_dir',
        default="/workspace/resnet50_cifar100/",
        help='Trained model file path. Download pretrained model from the following url and put it in model_dir specified path: https://download.pytorch.org/models/resnet18-5c106cde.pth'
    )
    parser.add_argument(
        '--config_file',
        default=None,
        help='quantization configuration file')
    parser.add_argument(
        '--subset_len',
        default=None,
        type=int,
        help='subset_len to evaluate model, using the whole validation dataset if it is not set')
    parser.add_argument(
        '--batch_size',
        default=32,
        type=int,
        help='input data batch size to evaluate model')
    parser.add_argument('--quant_mode',
                        default='calib',
                        choices=['float', 'calib', 'test'],
                        help='quantization mode. 0: no quantization, evaluate float model, calib: quantize, test: evaluate quantized model')
    parser.add_argument('--fast_finetune',
                        dest='fast_finetune',
                        action='store_true',
                        help='fast finetune model before calibration')
    parser.add_argument('--deploy',
                        dest='deploy',
                        action='store_true',
                        help='export xmodel for deployment')
    parser.add_argument('--inspect',
                        dest='inspect',
                        action='store_true',
                        help='inspect model')
    args, _ = parser.parse_known_args()

    normalize = transforms.Normalize(
        (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))

    train_loader = torch.utils.data.DataLoader(
        CIFAR100(root='./data', train=True, transform=transforms.Compose([
            transforms.RandomCrop(32, 4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]), download=True),
        batch_size=128, shuffle=True, num_workers=4)

    test_dataset = CIFAR100(root='./data', train=False, transform=transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ]))

    test_loader = torch.utils.data.DataLoader(
        CIFAR100(root='./data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=100, shuffle=False, num_workers=4)

    model_name = 'resnet18'
    file_path = os.path.join(args.model_dir, 'resnet18-200-regular.pth')

    feature_test = ' float model evaluation'
    if args.quant_mode != 'float':
        feature_test = ' quantization'
        # force to merge BN with CONV for better quantization accuracy
        args.optimize = 1
        feature_test += ' with optimization'
    else:
        feature_test = ' float model evaluation'
    title = model_name + feature_test

    print("-------- Start {} test ".format(model_name))

    # calibration or evaluation
    quantization(
        title=title,
        model_name=model_name,
        file_path=file_path)

    print("-------- End of {} test ".format(model_name))

    # net = resnet50().to(device)
    # net.load_state_dict(torch.load('resnet50-200-best.pth')) #0.7931
    # # acc = evaluate(net, cifar100_test_loader )
    # # print(acc)
    # inspector = Inspector('0x20200000000002a')
    # inspector = Inspector('DPUCAHX8H_ISA2')
    # input = torch.randn([5, 3, 32, 32])
    # inspector.inspect(model,input)

    # quant_mode = "calib"  # test calib
    # inputs_calib = gen_calib(test_loader)
    # # inputs_calib=torch.tensor(inputs_calib)
    # inputs_calib = torch.randn([1, 3, 32, 32])
    # quantizer = torch_quantizer(quant_mode, model, (inputs_calib))
    # quant_model = quantizer.quant_model
    # # acc1_gen, acc5_gen = evaluate(quant_model, test_loader)

    # if quant_mode == "calib":
    #     quantizer.export_quant_config()
    # if deploy:
    #     quantizer.export_xmodel())
