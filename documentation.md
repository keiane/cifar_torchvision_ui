# Overview
This interface facilitates training deep learning models on the CIFAR-10 dataset using PyTorch. Users can select from a 
variety of models, set training parameters, and initiate training to evaluate model performance. Here's more about it:
# Training Parameters
In the training parameters section, users can customize the training process by adjusting several settings, sorted into basic and advanced. These parameters collectively allow users to tailor the training process to meet specific computational constraints and performance goals.
## Basic Settings
### Model Selection:
In the model selection section, users have the option to choose from a variety of predefined models, each with its unique architecture and set of parameters. The available models are tailored for different computational capabilities and objectives, thereby offering a diverse range of options for training on the CIFAR-10 dataset. By providing a selection of models, this interface facilitates a more flexible and tailored approach to exploring and understanding the performance of different neural network architectures on the CIFAR-10 dataset. Users can easily switch between models to observe how each performs and to find the one that best meets their requirements.
### Weights and Biases:
Username required for weights and biases (wandb) website to save graphs regarding accuracy and loss. If you do not have a wandb account, input "balica15".
### Number of Epochs:
The number of epochs controls how many times the entire training dataset is passed forward and backward through the neural network.
### Run:
Run the program. Once pressed, the selected model begins training on the CIFAR-10 dataset using the specified training parameters. The training process includes both forward and backward passes through the network, optimizing the model's weights to minimize the loss function.
## Advanced Settings
### Training and Testing Batch Sizes:
The training and testing batch sizes determine the number of samples that will be propagated through the network at one time, affecting the speed and memory usage of the training process.
### Learning Rate:
The starting learning rate optimization of the optimizers. The learning rate in the optional schedulers are unable to be edited as they were chosen specifically to heighten accuracy. 
### Optimizer:
The optimizer selection allows users to choose between different optimization algorithms, namely SGD (Stochastic Gradient Descent) or Adam, which have distinct behaviors and performance characteristics.
### Scheduler:
The scheduler selection allows users to choose how they want learning rate to change over the course of the program. There are four options: None, CosineAnnealingLR, ReduceLROnPlateau, and StepLR.
- None: No scheduler. The learning rate remains constant the entire run.
- CosineAnnealingLR: The learning rate of each parameter group is determined using a cosine annealing schedule. 
- ReduceLROnPlateau: The learning rate reduces when a parameter stops improving over a certain interval. In this case, if the accuracy stops improving for five epochs straight, the program will lower the learning rate.
- StepLR: The learning rate decreases at a set rate over a set interval. In this case, every 30 epochs the learning rate decreases by a factor of 0.1.
### Attacking Methods:
If attacking methods are enabled, it reveals two mores settings: gaussian noise and advarsarial attack. 

When gaussian noise is enabled, the user can choose a value for the sigma value of gaussian noise, controlling how much it influences the model. A sigma value of 0 disables gaussian noise, even if the setting is enabled. 

The advarsarial attack is simply a toggle that causes almost unnoticable changes to the pictures the model is looking at, which can cause incorrect results.
# Training Results
Upon completion of the training across the defined number of epochs, the interface will evaluate the model on the test dataset and display the achieved accuracy, 10 testing pictures per every 10 epochs, the gaussian noise on an image (if enabled), and the advarsarial attack result on an image (if enabled).
# Warnings
Any warnings during training will be displayed in a yellow popup at the top right of the interface.
# Data
The CIFAR-10 dataset used in this interface comprises 60,000 32x32 color images spread across 10 different classes, with a training set of 50,000 images and a testing set of 10,000 images. Before training, the dataset undergoes specific transformations such as random cropping and normalization to augment the data and standardize the pixel values, respectively. These preprocessing steps help in enhancing the model's ability to learn and generalize well from the data. The interface automatically handles the downloading and preparation of the CIFAR-10 dataset, making it effortless for users to start training models without worrying about data management.