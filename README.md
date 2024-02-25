![image](https://github.com/zibinpan/FedLF/assets/79980563/57207813-903f-456c-8d6d-eda786211686)# Introduction
This repository is the implementation of the paper FedLF: Layer-Wise Fair Federated Learning (accepted by AAAI 2024).

This work presents a promising way to handle FL non-iid problem that the global model easily has imbalanced performance that favors part of clients while performing poorly on others. The proposed FedLF can significantly increase the average model performance and reduce the performance variance across clients.

**What Makes Federated Learning Unfair?**
![image](https://github.com/zibinpan/FedLF/assets/79980563/559b2963-60df-47c5-89ce-3212ed2bdf70)

**Challenges of computing a fair update direction in FL:**
1. Model-level gradient conflict.
2. Improvement bias.
![image](https://github.com/zibinpan/FedLF/assets/79980563/f650a5a1-f71a-493d-938c-4c70788e0a0f)
3. Layer-level gradient conflict.
![image](https://github.com/zibinpan/FedLF/assets/79980563/088b1789-8ae2-489e-9e64-a701ca390fa5)

**Visualization of unfairness:**
![image](https://github.com/zibinpan/FedLF/assets/79980563/8df33c21-1011-462b-a539-b600e22b6461)

## Code of the paper

The code can be run directly in Windows, and all experiments can be replicated by running the code. In the event that our paper is accepted for publication, we plan to release all of our codes as open source. However, for safety concerns, we have encrypted some of the code for review.

## Requirements

The code can be run under **Python 3.6-3.10 Windows x64** with dependencies below:

```
numpy>=1.19.5
pytorch>=1.8.1
torchvision>=0.9.0
cvxopt>=1.3.0
matplotlib>=3.1.3
```

## Basic usage
Copy the following command in the Windows cmd under the same folder of `./gfedplat` as well as `run.py`, and then run (one example is shown as follows). By setting different parameters and run the command, you can replicate the results of all experiments.
```
python run.py --seed 1 --device 0 --module CNN --algorithm FedAvg --dataloader DataLoader_cifar10_dir --N 100 --Diralpha 0.1 --B 50 --C 0.1 --R 3000 --E 1 --lr 0.1 --decay 0.999
```
All parameters can be seen in `./gfedplat/main.py`.

Enjoy yourself!
