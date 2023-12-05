###### Train CIFAR10 with PyTorch. ######

### IMPORT DEPENDENCIES

from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import gradio as gr
import wandb
import math
import numpy as np
import torchvision.models as models
import matplotlib.pyplot as plt

import torchvision
import torchvision.transforms as transforms
import torch.optim.lr_scheduler as lr_scheduler
import os
import argparse

from models import *

from tqdm import tqdm
from PIL import Image
import gradio as gr

# from utils import progress_bar

# CSS theme styling
theme = gr.themes.Base(
    font=[gr.themes.GoogleFont('Montserrat'), 'ui-sans-serif', 'system-ui', 'sans-serif'],
    primary_hue="emerald",
    secondary_hue="emerald",
    neutral_hue="zinc"
).set(
    body_text_color='*neutral_950',
    body_text_color_subdued='*neutral_950',
    block_shadow='*shadow_drop_lg',
    button_shadow='*shadow_drop_lg',
    block_title_text_color='*neutral_950',
    block_title_text_weight='500',
    slider_color='*secondary_600'
)

def normalize(img):
    min_im = np.min(img)
    np_img = img - min_im
    max_im = np.max(np_img)
    np_img /= max_im
    return np_img

### MAIN FUNCTION

def main(drop_type, epochs_sldr, train_sldr, test_sldr, learning_rate, optimizer, sigma_sldr, username, scheduler):

    ## Input protection
    if not drop_type:
        gr.Warning("Please select a model from the dropdown.")
        return
    if not username:
        gr.Warning("Please enter a WandB username.")
        return
    if(epochs_sldr % 1 != 0):
        gr.Warning("Number of epochs must be an integer.")
        return
    if(train_sldr % 1 != 0):
        gr.Warning("Training batch size must be an integer.")
        return
    if(test_sldr % 1 != 0):
        gr.Warning("Testing batch size must be an integer.")
        return

    num_epochs = int(epochs_sldr)
    global learn_batch
    learn_batch = int(train_sldr)
    global test_batch
    test_batch = int(test_sldr)
    learning_rate = float(learning_rate)
    optimizer_choose = str(optimizer)
    sigma = float(sigma_sldr) 
    scheduler_choose = str(scheduler)
    
    # REPLACE ENTITY WITH USERNAME BELOW
    wandb.init(entity=username, project="tutorial")
    
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--resume', '-r', action='store_true',
                        help='resume from checkpoint')
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = 'cuda'
        gr.Info("Cuda detected - running on Cuda")
    elif torch.backends.mps.is_available():
        device = 'mps'
        gr.Info("MPS detected - running on Metal")
    else:
        device = 'cpu'
        gr.Info("No GPU Detected - running on CPU")

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

    SGDopt = optim.SGD(net.parameters(), lr=learning_rate,momentum=0.9, weight_decay=5e-4)
    Adamopt = optim.Adam(net.parameters(), lr=learning_rate, weight_decay=5e-4)

    criterion = nn.CrossEntropyLoss()

    if optimizer_choose == "SGD":
        optimizer = SGDopt
    elif optimizer_choose == "Adam":
        optimizer = Adamopt
    print (f'optimizer: {optimizer}')

    #scheduler = lr_scheduler.LinearLR(optimizer, start_factor=learning_rate, end_factor=0.0001, total_iters=10)
    if scheduler_choose == "CosineAnnealingLR":
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
    elif scheduler_choose == "ReduceLROnPlateau":
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=5)
    elif scheduler_choose == "StepLR":
        scheduler = lr_scheduler.StepLR(optimizer, step_size=30)
    print (f'scheduler: {scheduler_choose}')

    img_list = [] # initialize list for image generation
    img_list2 = []
    for epoch in range(start_epoch, start_epoch+epochs_sldr):
        if sigma == 0:
            train(epoch, net, trainloader, device, optimizer, criterion, sigma)
        else:
            gaussian_fig = train(epoch, net, trainloader, device, optimizer, criterion, sigma)
        acc = test(epoch, net, testloader, device, criterion)

        if scheduler_choose == "ReduceLROnPlateau":
            scheduler.step(metrics=acc)
        elif not scheduler_choose == "None":
            scheduler.step()

        if ((epoch-1) % 10 == 0) or (epoch == 0): # generate images every 10 epochs (and the 0th epoch)
            dataiter = iter(trainloader)
            imgs, labels = next(dataiter)
            normalized_imgs = (imgs-imgs.min())/(imgs.max()-imgs.min())
            for i in range(10): # generate 10 images per epoch
                gradio_imgs = transforms.functional.to_pil_image(normalized_imgs[i])
                img_list.append(gradio_imgs) 
                if sigma != 0:
                    img_list2.append(gaussian_fig)
    
    if sigma == 0:
        return str(acc)+"%", img_list, None
    else:
        return str(acc)+"%", img_list, img_list2



### TRAINING
def train(epoch, net, trainloader, device, optimizer, criterion, sigma, progress=gr.Progress()):
    try:
        print('\nEpoch: %d' % epoch)
        net.train()
        train_loss = 0
        correct = 0
        total = 0
        
        temp_val = 0
        temp2_val = 0

        iter_float = 50000/learn_batch
        iterations = math.ceil(iter_float)
        iter_prog = 0

        for batch_idx, (inputs, targets) in tqdm(enumerate(trainloader)):
            if sigma == 0:
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad()
                outputs = net(inputs)
            else:
                noise = np.random.normal(0, sigma, inputs.shape)
                inputs += torch.tensor(noise)
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad()
                outputs = net(inputs)
                n_inputs = inputs.clone().detach().cpu().numpy()
                if(batch_idx%99 == 0):
                    plt.imshow(normalize(np.transpose(n_inputs[0], (1, 2, 0))))
                    fig_name = "test_input.png"
                    plt.savefig(fig_name)
                    print(f'Figure saved as {fig_name}')
                    gaussian_fig = Image.open(fig_name)

            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            iter_prog = iter_prog + 1 # Iterating iteration amount
            progress(iter_prog/iterations, desc=f"Training Epoch {epoch}", total=iterations)
            

            # progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            #              % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

    except Exception as e:
        print(f"Error: {e}")
        gr.Warning(f"Training Error: {e}")
    if sigma != 0:
        return gaussian_fig


### TESTING

def test(epoch, net, testloader, device, criterion, progress = gr.Progress()):
    try:
        global best_acc
        net.eval()
        test_loss = 0
        correct = 0
        total = 0

        iter_float = 10000/test_batch
        iterations = math.ceil(iter_float)
        iter_prog = 0

        with torch.no_grad():
            for batch_idx, (inputs, targets) in tqdm(enumerate(testloader)):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = net(inputs)
                loss = criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                iter_prog = iter_prog + 1 # Iterating iteration amount
                progress(iter_prog/iterations, desc=f"Testing Epoch {epoch}", total=iterations)

            wandb.log({'epoch': epoch+1, 'loss': test_loss})
            wandb.log({"acc": correct/total})

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


models_dict = {
        "ConvNext_Small": models.convnext_small(weights=models.ConvNeXt_Small_Weights.DEFAULT),
        "ConvNext_Base": models.convnext_base(weights=models.ConvNeXt_Base_Weights.DEFAULT),
        "ConvNext_Large": models.convnext_large(weights=models.ConvNeXt_Large_Weights.DEFAULT),
        "DenseNet": models.densenet121(weights=models.DenseNet121_Weights.DEFAULT),
        #"EfficientNet_B0": models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT),
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

# Scheduler names
schedulers = ["None","CosineAnnealingLR","ReduceLROnPlateau","StepLR"]

### GRADIO APP INTERFACE

def settings(choice):
    if choice == "Advanced":
        advanced = [
            gr.Slider(visible=True),
            gr.Slider(visible=True),
            gr.Slider(visible=True),
            gr.Dropdown(visible=True),
            gr.Dropdown(visible=True),
            gr.Radio(visible=True)
        ]
        return advanced
    else:
        basic = [
            gr.Slider(visible=False),
            gr.Slider(visible=False),
            gr.Slider(visible=False),
            gr.Dropdown(visible=False),
            gr.Dropdown(visible=False),
            gr.Radio(visible=False)
        ]
        return basic

def attacks(choice):
    if choice == "Yes":
        yes = [
            gr.Markdown(visible=True),
            gr.Slider(visible=True),
            gr.Gallery(visible=True)
        ]
        return yes
    if choice == "No":
        no = [
            gr.Markdown(visible=False),
            gr.Slider(visible=False),
            gr.Gallery(visible=False)
        ]
        return no


## Main app for functionality
with gr.Blocks() as functionApp:
    with gr.Row():
        gr.Markdown("# CIFAR-10 Model Training GUI")
    with gr.Row():
        gr.Markdown("## Parameters")
    with gr.Row():
        inp = gr.Dropdown(choices=names, label="Training Model", value="ResNet18", info="Choose one of 13 common models provided in the dropdown to use for training.")
        username = gr.Textbox(label="Weights and Biases", info="Enter your username from the Weights and Biases API")
        epochs_sldr = gr.Slider(label="Number of Epochs", minimum=1, maximum=100, step=1, value=1, info="How many times the model will see the entire dataset during trianing.")
        with gr.Column():
            setting_radio = gr.Radio(["Basic", "Advanced"], label="Settings", value="Basic")
            btn = gr.Button("Run")        
    with gr.Row():
        train_sldr = gr.Slider(visible=False, label="Training Batch Size", minimum=1, maximum=1000, step=1, value=128, info="The number of training samples processed before the model's internal parameters are updated.")
        test_sldr = gr.Slider(visible=False, label="Testing Batch Size", minimum=1, maximum=1000, step=1, value=100, info="The number of testing samples processed at once during the evaluation phase.")
        learning_rate_sldr = gr.Slider(visible=False, label="Learning Rate", minimum=0.0001, maximum=0.1, step=0.0001, value=0.001, info="The learning rate of the optimization program.")
        optimizer = gr.Dropdown(visible=False, label="Optimizer", choices=optimizers, value="SGD", info="The optimization algorithm used to minimize the loss function during training.")
        scheduler = gr.Dropdown(visible=False, label="Scheduler", choices=schedulers, value="CosineAnnealingLR", info="do later")
        use_sigma = gr.Radio(["Yes", "No"], visible=False, label="Use Gaussian Noise", value= "No")
        setting_radio.change(fn=settings, inputs=setting_radio, outputs=[train_sldr, test_sldr, learning_rate_sldr, optimizer, scheduler, use_sigma])
    with gr.Row():
        attack_method = gr.Markdown("## Attacking Methods", visible=False)
    with gr.Row():
        sigma_sldr = gr.Slider(visible=False, label="Gaussian Noise", minimum=0, maximum=1, value=0, step=0.1, info="do later")
    with gr.Row():
        gr.Markdown("## Training Results")
    with gr.Row():
        accuracy = gr.Textbox(label = "Accuracy", info="The validation accuracy of the trained model (accuracy evaluated on testing data).")
        pics = gr.Gallery(preview=True,selected_index=0,object_fit='contain')  
        gaussian_pics = gr.Gallery(visible=False, preview=True, selected_index=0, object_fit='contain')
        use_sigma.change(fn=attacks, inputs=use_sigma, outputs=[attack_method, sigma_sldr, gaussian_pics])
        btn.click(fn=main, inputs=[inp, epochs_sldr, train_sldr, test_sldr, learning_rate_sldr, optimizer, sigma_sldr, username, scheduler], outputs=[accuracy, pics, gaussian_pics])

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
                    ''') # Can be collapesed in VSCode to hide paragraphs from view. Vscode can also wrap text.



### LAUNCH APP

if __name__ == '__main__':
    mainApp = gr.TabbedInterface([functionApp, documentationApp], ["Welcome", "Documentation"], theme=theme)
    mainApp.queue()
    mainApp.launch()
