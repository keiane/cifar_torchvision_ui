# Train CIFAR10 with PyTorch

I'm playing with [PyTorch](http://pytorch.org/) on the CIFAR10 dataset.

## Prerequisites
- Python 3.6+
- PyTorch 1.0+

- Install the dependencies from `requirements.txt` by running `pip install -r requirements.txt`
- Update the `requirements.txt` with `pip list --format=freeze > requirements.txt`

## Dependencies for Weights and Biases
You must install weights and biases using the command below:
`pip install wandb`

Afterwards, you must run this command to login into weights and biases. A GitHub account can be used to login:
`wandb login`

Paste the API key into terminal.

Additionally, change this line in `main.py`:
`wandb.init(entity="balica15", project="tutorial")`

Where "balica15" is replaced by your username that you used to login with into Weights and Biases.


## Training
```
# Start training with: 
python main.py

# You can manually resume the training with: 
python main.py --resume --lr=0.01
```

## Features to Add
| Name                          | Feature      |
| ----------------------------- | ----------- |
| Evelyn Atkins and Ethan White | Input and Error Protection     |
| Keiane Balicanta              | TorchVision Model Dropdown      |
| Henry Conde                   | Weights and Biases API      |
| Matthew Gerace                | Iteration and Batch Size Sliders      |
| Luke Wilkins                  | Image Classification |


## Accuracy
| Model             | Acc.        |
| ----------------- | ----------- |
| [VGG16](https://arxiv.org/abs/1409.1556)              | 92.64%      |
| [ResNet18](https://arxiv.org/abs/1512.03385)          | 93.02%      |
| [ResNet50](https://arxiv.org/abs/1512.03385)          | 93.62%      |
| [ResNet101](https://arxiv.org/abs/1512.03385)         | 93.75%      |
| [RegNetX_200MF](https://arxiv.org/abs/2003.13678)     | 94.24%      |
| [RegNetY_400MF](https://arxiv.org/abs/2003.13678)     | 94.29%      |
| [MobileNetV2](https://arxiv.org/abs/1801.04381)       | 94.43%      |
| [ResNeXt29(32x4d)](https://arxiv.org/abs/1611.05431)  | 94.73%      |
| [ResNeXt29(2x64d)](https://arxiv.org/abs/1611.05431)  | 94.82%      |
| [SimpleDLA](https://arxiv.org/abs/1707.064)           | 94.89%      |
| [DenseNet121](https://arxiv.org/abs/1608.06993)       | 95.04%      |
| [PreActResNet18](https://arxiv.org/abs/1603.05027)    | 95.11%      |
| [DPN92](https://arxiv.org/abs/1707.01629)             | 95.16%      |
| [DLA](https://arxiv.org/pdf/1707.06484.pdf)           | 95.47%      |

