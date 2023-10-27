'''Train CIFAR10 with PyTorch.'''
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms
import torchvision.models as models

import os
import argparse

from models import *

from tqdm import tqdm
import gradio as gr

# from utils import progress_bar

def main(drop_type):
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--resume', '-r', action='store_true',
                        help='resume from checkpoint')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch

    # Data
    print('==> Preparing data..')
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train)
    trainloader = DataLoader(
        trainset, batch_size=128, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)
    testloader = DataLoader(
        testset, batch_size=100, shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat', 'deer',
            'dog', 'frog', 'horse', 'ship', 'truck')

    # Model
    print('==> Building model..')
    net = models_dict.get(drop_type, None)

    # Make list of models containing either classifer or fc functions
    classifier_models = ['ConvNext_Small', 'ConvNext_Base', 'ConvNext_Large', 'DenseNet', 'EfficientNet_B0', 'MobileNetV2',
                         'MaxVit', 'MnasNet0_5', 'SqueezeNet', 'VGG19']
    fc_models = ['GoogLeNet', 'InceptionNetV3', 'RegNet_X_400MF', 'ResNet18', 'ShuffleNet_V2_X0_5']

    # Check dropdown choice for fc or classifier function implementation
    if net in classifier_models:
        num_ftrs = net.classifier[-1].in_features
        net.classifier[-1] = torch.nn.Linear(num_ftrs, len(classes))
    elif net in fc_models:
        num_ftrs = net.fc.in_features
        net.fc = torch.nn.Linear(num_ftrs, len(classes))
    
    net = net.to(device)

    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
        checkpoint = torch.load('./checkpoint/ckpt.pth')
        net.load_state_dict(checkpoint['net'])
        best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch']

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr,
                        momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

    for epoch in range(start_epoch, start_epoch+200):
        train(epoch, net, trainloader, device, optimizer, criterion)
        acc = test(epoch, net, testloader, device, criterion)
        scheduler.step()
    return acc


# Training
def train(epoch, net, trainloader, device, optimizer, criterion):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in tqdm(enumerate(trainloader)):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        # progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
        #              % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))


def test(epoch, net, testloader, device, criterion):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in tqdm(enumerate(testloader)):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            # progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            #              % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    print(acc)
    # if acc > best_acc:
    #     print('Saving..')
    #     state = {
    #         'net': net.state_dict(),
    #         'acc': acc,
    #         'epoch': epoch,
    #     }
    #     if not os.path.isdir('checkpoint'):
    #         os.mkdir('checkpoint')
    #     torch.save(state, './checkpoint/ckpt.pth')
    #     best_acc = acc
    return acc

models_dict = {
        "ConvNext_Small": models.convnext_small(weights=models.ConvNeXt_Small_Weights.DEFAULT),
        "ConvNext_Base": models.convnext_base(weights=models.ConvNeXt_Base_Weights.DEFAULT),
        "ConvNext_Large": models.convnext_large(weights=models.ConvNeXt_Large_Weights.DEFAULT),
        "DenseNet": models.densenet121(weights=models.DenseNet121_Weights.DEFAULT),
        "EfficientNet_B0": models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT),
        "GoogLeNet": models.googlenet(weights=models.GoogLeNet_Weights.DEFAULT),
        # "InceptionNetV3": models.inception_v3(weights=models.Inception_V3_Weights.DEFAULT),
        # "MaxVit": models.maxvit_t(weights=models.MaxVit_T_Weights.DEFAULT),
        "MnasNet0_5": models.mnasnet0_5(weights=models.MNASNet0_5_Weights.DEFAULT),
        "MobileNetV2": models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT),
        "ResNet18": models.resnet18(weights=models.ResNet18_Weights.DEFAULT),
        "RegNet_X_400MF": models.regnet_x_400mf(weights=models.RegNet_X_400MF_Weights.DEFAULT),
        "ShuffleNet_V2_X0_5": models.shufflenet_v2_x0_5(weights=models.ShuffleNet_V2_X0_5_Weights.DEFAULT),
        "SqueezeNet": models.squeezenet1_0(weights=models.SqueezeNet1_0_Weights.DEFAULT),
        "VGG19": models.vgg19(weights=models.VGG19_Weights.DEFAULT)
}

# Store dictionary keys into list for dropdown menu choices
names = list(models_dict.keys())

with gr.Blocks() as demo:
    #ADD CODE HERE
    with gr.Row():
        inp = gr.Dropdown(names)
    with gr.Row():
        out = gr.Textbox(label="Accuracy")
    with gr.Row():
        btn = gr.Button("Run")
    btn.click(fn=main,inputs=inp,outputs=out)

if __name__ == '__main__':
    demo.launch()
    main()