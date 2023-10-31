'''Train CIFAR10 with PyTorch.'''
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import matplotlib.pyplot as plt
import numpy as np

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
    total_epoch = 1 # how many epochs to run
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
    if drop_type == "resnet":
        #net = ResNet18() # run from computer file instead of from torchvision.models
        #net = models.resnet18
        net = models.resnet18(weights=models.ResNet18_Weights) # to use pretrained weights
        num_ftrs = net.fc.in_features
        # here the size of each output sample is set to 2
        # alternatively, it can be generalized to nn.linear((num_ftrs,len(classes))
        net.fc = torch.nn.Linear(num_ftrs, len(classes))
    elif drop_type == "vgg":
        net = VGG('VGG19')
    elif drop_type == "preact_resnet":
        net = PreActResNet18()
    elif drop_type == "googlenet":
        net = GoogLeNet()
    elif drop_type == "densenet":
        net = DenseNet121()
    elif drop_type == "resnext":
        net = ResNeXt29_2x64d()
    elif drop_type == "mobilenet":
        net = MobileNet()
    elif drop_type == "mobilenetv2":
        net = MobileNetV2()
    elif drop_type == "dpn":
        net = DPN92()
    elif drop_type == "shufflenet":
        net = ShuffleNetG2()
    elif drop_type == "senet":
        net = SENet18()
    elif drop_type == "shufflenetv2":
        net = ShuffleNetV2(1)
    elif drop_type == "efficientnet":
        net = EfficientNetB0()
    elif drop_type == "regnet":
        net = RegNetX_200MF()
    elif drop_type == "dla_simple":
        net = SimpleDLA()
        
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

    img_list = [] # initialize list for image generation
    for epoch in range(start_epoch, start_epoch+total_epoch):
        train(epoch, net, trainloader, device, optimizer, criterion)
        acc = test(epoch, net, testloader, device, criterion)
        scheduler.step()
        if ((epoch-1) % 10 == 0) or (epoch == 0): # generate images every 10 epochs (and the 0th epoch)
            dataiter = iter(testloader)
            imgs, labels = next(dataiter)
            normalized_imgs = (imgs-imgs.min())/(imgs.max()-imgs.min())
            for i in range(10): # generate 10 images per epoch
                gradio_imgs = transforms.functional.to_pil_image(normalized_imgs[i])
                img_list.append(gradio_imgs)

    return acc, img_list


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

path = "./models"
dir_list = os.listdir(path)
dir_list.remove("__init__.py")
dir_list.remove("__pycache__")
files = [file.strip(".py") for file in dir_list]

with gr.Blocks() as demo:
    #ADD CODE HERE
    with gr.Row():
        inp = gr.Dropdown(files)
    with gr.Row():
        out = gr.Textbox(label="Accuracy")
    with gr.Row():
        pics = gr.Gallery(preview=True,selected_index=0,object_fit='contain')
    with gr.Row():
        btn = gr.Button("Run")
    btn.click(fn=main,inputs=inp,outputs=[out,pics])

if __name__ == '__main__':
    demo.launch()
    # main(drop_type = "resnet")
    #main()