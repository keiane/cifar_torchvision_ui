'''Train CIFAR10 with PyTorch.'''
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision.models as models

import torchvision
import torchvision.transforms as transforms

import os
import argparse

from models import *

from tqdm import tqdm
import gradio as gr
# from utils import progress_bar

def main(value1,value2,value3,value4):
    num_epochs = int(value1)
    learn_batch = int(value2)
    test_batch = int(value3)
    optimizer_choose = str(value4)
    print (optimizer_choose)

    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--resume', '-r', action='store_true',
                        help='resume from checkpoint')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    best_acc = 0  # best test accuracy
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
        trainset, batch_size=learn_batch, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)
    testloader = DataLoader(
        testset, batch_size=test_batch, shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat', 'deer',
            'dog', 'frog', 'horse', 'ship', 'truck')

    # Model
    print('==> Building model..')
    # net = ResNet18()
    net = models.resnet18(weights=models.ResNet18_Weights) # to use pretrained weights
    num_ftrs = net.fc.in_features
    net.fc = torch.nn.Linear(num_ftrs, len(classes))
    # net = VGG('VGG19')
    # net = PreActResNet18()
    # net = GoogLeNet()
    # net = DenseNet121()
    # net = ResNeXt29_2x64d()
    # net = MobileNet()
    # net = MobileNetV2()
    # net = DPN92()
    # net = ShuffleNetG2()
    # net = SENet18()
    # net = ShuffleNetV2(1)
    # net = EfficientNetB0()
    # net = RegNetX_200MF()
    # net = SimpleDLA()
    net = net.to(device)

    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
        checkpoint = torch.load('./checkpoint/ckpt.pth')
        net.load_state_dict(checkpoint['net'])
        best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch']

    SGDopt = optim.SGD(net.parameters(), lr=args.lr,momentum=0.9, weight_decay=5e-4)
    Adamopt = optim.Adam(net.parameters(), lr=args.lr, weight_decay=5e-4)

    criterion = nn.CrossEntropyLoss()

    if optimizer_choose == "SGD":
        optimizer = SGDopt
    elif optimizer_choose == "Adam":
        optimizer = Adamopt
    print (optimizer)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

    for epoch in range(start_epoch, start_epoch+num_epochs):
        train(epoch, net, trainloader, device, optimizer, criterion)
        test(epoch, net, testloader, device, criterion)
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
    global acc
    acc = 100.*correct/total
    print (acc)
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
    # hi

list = ["SGD","Adam"]
with gr.Blocks() as demo:
    with gr.Row():
        inp = gr.Slider(label="# of Epochs",minimum=1,maximum=100,step=1,value=1)
        inp1 = gr.Slider(label="Training Batch Size",minimum=1,maximum=1000,step=1,value=128)
        inp2 = gr.Slider(label="Testing Batch Size",minimum=1,maximum=1000,step=1,value=100)
        inp3 = gr.Dropdown(label="Choose Optimizer",choices=list,value="SGD")
        accuracy = gr.Textbox(label = "Accuracy")

    btn = gr.Button("Import")
    btn.click(fn=main, inputs=[inp, inp1, inp2, inp3], outputs = [accuracy])

if __name__ == '__main__':
    demo.launch()
    main()