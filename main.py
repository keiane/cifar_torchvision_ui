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

import torchvision.models as models

import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import torch.optim.lr_scheduler as lr_scheduler
import os
import argparse
import torchattacks

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

def imshow(img, fig_name = "test_input.png"):
    try:
        img = img.clone().detach().cpu().numpy()
    except:
        print('img already numpy')

    plt.imshow(normalize(np.transpose(img, (1, 2, 0))))
    plt.savefig(fig_name)
    print(f'Figure saved as {fig_name}')
    return fig_name

### MAIN FUNCTION

def main(drop_type, epochs_sldr, train_sldr, test_sldr, learning_rate, optimizer, sigma_sldr, adv_attack, username, scheduler):

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
    attack = str(adv_attack)
    scheduler_choose = str(scheduler)
    
    # REPLACE ENTITY WITH USERNAME BELOW
    wandb.init(entity=username, project="model-training")
    
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
    img_list2 = [] # initialize list for gaussian image generation
    img_list3 = [] # initialize list for adversarial attack image generation
    adv_num = 1 # initialize adversarial image number for naming purposes
    global gaussian_num 
    gaussian_num = 1 # initialize gaussian noise image number for naming purposes

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
        #if not epoch == 1: # ignore making images for 1st epoch
        if ((epoch-1) % 10 == 0) or (epoch == 0): # generate images every 10 epochs (and the 0th epoch)
            dataiter = iter(trainloader)
            imgs, labels = next(dataiter)
            normalized_imgs = (imgs-imgs.min())/(imgs.max()-imgs.min())
            atk = torchattacks.PGD(net, eps=0.00015, alpha=0.0000000000000001, steps=7)
            if attack == "Yes":
                if normalized_imgs is None:
                    print("error occured")
                else:
                    print(torch.std(normalized_imgs))
                    atk.set_normalization_used(mean = torch.mean(normalized_imgs,axis=[0,2,3]), std=torch.std(normalized_imgs,axis=[0,2,3])/1.125)
                    adv_images = atk(imgs, labels)
                    fig_name = imshow(adv_images[0], fig_name = f'figures/adversarial_attack{adv_num}.png')
                    attack_fig = Image.open(fig_name)
                    for i in range(1): # generate 1 image per epoch
                        img_list3.append(attack_fig)
                        adv_num = adv_num + 1
            for i in range(10): # generate 10 images per epoch
                gradio_imgs = transforms.functional.to_pil_image(normalized_imgs[i])
                img_list.append(gradio_imgs) 
            if sigma != 0:
                    for i in range(1): # generate 1 image per epoch
                        img_list2.append(gaussian_fig)
                        gaussian_num = gaussian_num + 1

    if (sigma == 0) and (attack == "No"):
        return str(acc)+"%", img_list, None, None
    elif (sigma != 0) and (attack == "No"):
        return str(acc)+"%", img_list, img_list2, None
    elif (sigma == 0) and (attack == "Yes"):
        return str(acc)+"%", img_list, None, img_list3
    else:
        return str(acc)+"%", img_list, img_list2, img_list3



### TRAINING
def train(epoch, net, trainloader, device, optimizer, criterion, sigma, progress=gr.Progress()):
    try:
        print('\nEpoch: %d' % epoch)
        net.train()
        train_loss = 0
        correct = 0
        total = 0

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
                        fig_name = imshow(n_inputs[0], fig_name= f'figures/gaussian_noise{gaussian_num}.png')
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
        #"AlexNet": models.AlexNet(weights=models.AlexNet_Weights.DEFAULT),
        #"ConvNext_Small": models.convnext_small(weights=models.ConvNeXt_Small_Weights.DEFAULT),
        #"ConvNext_Base": models.convnext_base(weights=models.ConvNeXt_Base_Weights.DEFAULT),
        #"ConvNext_Large": models.convnext_large(weights=models.ConvNeXt_Large_Weights.DEFAULT),
        "DenseNet": models.densenet121(weights=models.DenseNet121_Weights.DEFAULT),
        #"EfficientNet_B0": models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT),
        #"GoogLeNet": models.googlenet(weights=models.GoogLeNet_Weights.DEFAULT),
        # "InceptionNetV3": models.inception_v3(weights=models.Inception_V3_Weights.DEFAULT),
        # "MaxVit": models.maxvit_t(weights=models.MaxVit_T_Weights.DEFAULT),
        #"MnasNet0_5": models.mnasnet0_5(weights=models.MNASNet0_5_Weights.DEFAULT),
        #"MobileNetV2": models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT),
        "ResNet18": models.resnet18(weights=models.ResNet18_Weights.DEFAULT),
        "ResNet50": models.resnet50(weights=models.ResNet50_Weights.DEFAULT),
        #"RegNet_X_400MF": models.regnet_x_400mf(weights=models.RegNet_X_400MF_Weights.DEFAULT),
        #"ShuffleNet_V2_X0_5": models.shufflenet_v2_x0_5(weights=models.ShuffleNet_V2_X0_5_Weights.DEFAULT),
        #"SqueezeNet": models.squeezenet1_0(weights=models.SqueezeNet1_0_Weights.DEFAULT),
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
            gr.Radio(visible=True),
            gr.Radio(visible=True)
        ]
        return yes
    if choice == "No":
        no = [
            gr.Markdown(visible=False),
            gr.Radio(visible=False),
            gr.Radio(visible=False)
        ]
        return no
    
def gaussian(choice):
    if choice == "Yes":
        yes = [
            gr.Slider(visible=True),
            gr.Gallery(visible=True),
        ]
        return yes
    else:
        no = [
            gr.Slider(visible=False),
            gr.Gallery(visible=False),
        ]
        return no
def adversarial(choice):
    if choice == "Yes":
        yes = gr.Gallery(visible=True)
        return yes
    else:
        no = gr.Gallery(visible=False)

## Main app for functionality
with gr.Blocks() as functionApp:
    with gr.Row():
        gr.Markdown("# CIFAR-10 Model Training GUI")
    with gr.Row():
        gr.Markdown("## Parameters")
    with gr.Row():
        inp = gr.Dropdown(choices=names, label="Training Model", value="ResNet18", info="Choose one of 13 common models provided in the dropdown to use for training.")
        username = gr.Textbox(label="Weights and Biases", info="Enter your username or team name from the Weights and Biases API.")
        epochs_sldr = gr.Slider(label="Number of Epochs", minimum=1, maximum=100, step=1, value=1, info="How many times the model will see the entire dataset during trianing.")
        with gr.Column():
            setting_radio = gr.Radio(["Basic", "Advanced"], label="Settings", value="Basic")
            btn = gr.Button("Run")        
    with gr.Row():
        train_sldr = gr.Slider(visible=False, label="Training Batch Size", minimum=1, maximum=1000, step=1, value=128, info="The number of training samples processed before the model's internal parameters are updated.")
        test_sldr = gr.Slider(visible=False, label="Testing Batch Size", minimum=1, maximum=1000, step=1, value=100, info="The number of testing samples processed at once during the evaluation phase.")
        learning_rate_sldr = gr.Slider(visible=False, label="Learning Rate", minimum=0.0001, maximum=0.1, step=0.0001, value=0.001, info="The learning rate of the optimization program.")
        optimizer = gr.Dropdown(visible=False, label="Optimizer", choices=optimizers, value="SGD", info="The optimization algorithm used to minimize the loss function during training.")
        scheduler = gr.Dropdown(visible=False, label="Scheduler", choices=schedulers, value="CosineAnnealingLR", info="The scheduler used to iteratively alter learning rate.")
        use_attacks = gr.Radio(["Yes", "No"], visible=False, label="Use Attacking Methods?", value="No")
        setting_radio.change(fn=settings, inputs=setting_radio, outputs=[train_sldr, test_sldr, learning_rate_sldr, optimizer, scheduler, use_attacks])
    with gr.Row():
        attack_method = gr.Markdown("## Attacking Methods", visible=False)
    with gr.Row():
        use_sigma = gr.Radio(["Yes","No"], visible=False, label="Use Gaussian Noise?", value="No")
        sigma_sldr = gr.Slider(visible=False, label="Gaussian Noise", minimum=0, maximum=1, value=0, step=0.1, info="The sigma value of the gaussian noise eqaution. A value of 0 disables gaussian noise.")
        adv_attack = gr.Radio(["Yes","No"], visible=False, label="Use Adversarial Attacks?", value="No")
    with gr.Row():
        gr.Markdown("## Training Results")
    with gr.Row():
        accuracy = gr.Textbox(label = "Accuracy", info="The validation accuracy of the trained model (accuracy evaluated on testing data).")
        pics = gr.Gallery(preview=False,selected_index=0, object_fit='contain', label="Training Images")  
    with gr.Row():
        gaussian_pics = gr.Gallery(visible=False, preview=False, selected_index=0, object_fit='contain', label="Gaussian Noise")
        attack_pics = gr.Gallery(visible=False, preview=False, selected_index=0, object_fit='contain', label="Adversarial Attack")
        use_attacks.change(fn=attacks, inputs=use_attacks, outputs=[attack_method, use_sigma, adv_attack])
        use_sigma.change(fn=gaussian, inputs=use_sigma, outputs=[sigma_sldr, gaussian_pics])
        adv_attack.change(fn=adversarial, inputs=adv_attack, outputs=attack_pics)
        btn.click(fn=main, inputs=[inp, epochs_sldr, train_sldr, test_sldr, learning_rate_sldr, optimizer, sigma_sldr, adv_attack, username, scheduler], outputs=[accuracy, pics, gaussian_pics, attack_pics])

## Documentation app (implemented as second tab)

markdown_file_path = 'documentation.md'
with open(markdown_file_path, 'r') as file:
    markdown_content = file.read()

with gr.Blocks() as documentationApp:
    with gr.Row():
        gr.Markdown("# CIFAR-10 Training Interface Documentation")
    with gr.Row():
        gr.Markdown(markdown_content) # Can be collapesed in VSCode to hide paragraphs from view. Vscode can also wrap text.

### LAUNCH APP

if __name__ == '__main__':
    mainApp = gr.TabbedInterface([functionApp, documentationApp], ["Welcome", "Documentation"], theme=theme)
    mainApp.queue()
    mainApp.launch()
