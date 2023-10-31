###### Train CIFAR10 with PyTorch. ######

### IMPORT DEPENDENCIES

from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision.models as models

import torchvision
import torchvision.transforms as transforms
import torchvision.models as models

import os
import argparse

from models import *

from tqdm import tqdm
import gradio as gr

# from utils import progress_bar



### MAIN FUNCTION

def main(drop_type, epochs_sldr, train_sldr, test_sldr, optimizer):

    if not drop_type:
        gr.Warning("Please select a model from the dropdown.")
        return

    num_epochs = int(epochs_sldr)
    learn_batch = int(train_sldr)
    test_batch = int(test_sldr)
    optimizer_choose = str(optimizer)
    
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--resume', '-r', action='store_true',
                        help='resume from checkpoint')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch

    ## Data
    try:
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

        classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    except Exception as e:
        print(f"Error: {e}")
        gr.Warning(f"Data Loading Error: {e}")

    ## Model
    try:
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

    except Exception as e:
        print(f"Error: {e}")
        gr.Warning(f"Model Building Error: {e}")

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
        acc = test(epoch, net, testloader, device, criterion)
        scheduler.step()
    return acc



### TRAINING

def train(epoch, net, trainloader, device, optimizer, criterion):
    try:
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
    
    except Exception as e:
        print(f"Error: {e}")
        gr.Warning(f"Training Error: {e}")



### TESTING

def test(epoch, net, testloader, device, criterion):
    try:
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
    
    except Exception as e:
        print(f"Error: {e}")
        gr.Warning(f"Testing Error: {e}")



### DEFINE MODELS AND PARAMETERS

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

# Optimizer names
optimizers = ["SGD","Adam"]



### GRADIO APP INTERFACE

## Main app for functionality
with gr.Blocks() as functionApp:
    with gr.Row():
        gr.Markdown("# CIFAR-10 Model Training GUI")
    with gr.Row():
        gr.Markdown("## Model")
    with gr.Row():
        inp = gr.Dropdown(choices=names, label="Training Model", info="Choose one of 13 common models provided in the dropdown to use for training.")
    with gr.Row():
        gr.Markdown("## Parameters")
    with gr.Row():
        epochs_sldr = gr.Slider(label="Number of Epochs", minimum=1, maximum=100, step=1, value=1, info="How many times the model will see the entire dataset during trianing.")
        train_sldr = gr.Slider(label="Training Batch Size", minimum=1, maximum=1000, step=1, value=128, info="The number of training samples processed before the model's internal parameters are updated.")
        test_sldr = gr.Slider(label="Testing Batch Size", minimum=1, maximum=1000, step=1, value=100, info="The number of testing samples processed at once during the evaluation phase.")
        optimizer = gr.Dropdown(label="Optimizer", choices=optimizers, value="SGD", info="The optimization algorithm used to minimize the loss function during training.")
    with gr.Row():
        gr.Markdown("## Training Results")
    with gr.Row():
        accuracy = gr.Textbox(label = "Accuracy", info="The validation accuracy of the trained model (accuracy evaluated on testing data).")
    with gr.Row():
        btn = gr.Button("Run")
        btn.click(fn=main, inputs=[inp, epochs_sldr, train_sldr, test_sldr, optimizer], outputs=[accuracy])

## Documentation app (implemented as second tab)
with gr.Blocks() as documentationApp:
    with gr.Row():
        gr.Markdown("# CIFAR-10 Training Interface Documentation")
    with gr.Row():
        gr.Markdown('''
                    ## Overview
                    This interface facilitates training deep learning models on the CIFAR-10 dataset using PyTorch. Users can select from a 
                    variety of models, set training parameters, and initiate training to evaluate model performance. Here's more about it:
                    ### Model Selection:
                    In the model selection section, users have the option to choose from a variety of predefined models, each with its unique architecture and set of parameters. The available models are tailored for different computational capabilities and objectives, thereby offering a diverse range of options for training on the CIFAR-10 dataset. By providing a selection of models, this interface facilitates a more flexible and tailored approach to exploring and understanding the performance of different neural network architectures on the CIFAR-10 dataset. Users can easily switch between models to observe how each performs and to find the one that best meets their requirements.
                    ### Training Parameters:
                    In the training parameters section, users can customize the training process by adjusting several settings. The number of epochs controls how many times the entire training dataset is passed forward and backward through the neural network. The training and testing batch sizes determine the number of samples that will be propagated through the network at one time, affecting the speed and memory usage of the training process. Lastly, the optimizer selection allows users to choose between different optimization algorithms, namely SGD (Stochastic Gradient Descent) or Adam, which have distinct behaviors and performance characteristics. These parameters collectively allow users to tailor the training process to meet specific computational constraints and performance goals.
                    ### Training Results:
                    In the training results section, users can initiate the training process by clicking the "Run" button. Once pressed, the selected model begins training on the CIFAR-10 dataset using the specified training parameters. The training process includes both forward and backward passes through the network, optimizing the model's weights to minimize the loss function. Upon completion of the training across the defined number of epochs, the interface will evaluate the model on the test dataset and display the achieved accuracy.
                    ### Warnings:
                    Any warnings during training will be displayed in a yellow popup at the top right of the interface.
                    ### Data:
                    The CIFAR-10 dataset used in this interface comprises 60,000 32x32 color images spread across 10 different classes, with a training set of 50,000 images and a testing set of 10,000 images. Before training, the dataset undergoes specific transformations such as random cropping and normalization to augment the data and standardize the pixel values, respectively. These preprocessing steps help in enhancing the model's ability to learn and generalize well from the data. The interface automatically handles the downloading and preparation of the CIFAR-10 dataset, making it effortless for users to start training models without worrying about data management.
                    ''') # Can be collapesed in VSCode to hide paragraphs from view



### LAUNCH APP

if __name__ == '__main__':
    mainApp = gr.TabbedInterface([functionApp, documentationApp], ["Welcome", "Documentation"], theme=gr.themes.Base())
    mainApp.queue()
    mainApp.launch()