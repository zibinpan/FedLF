# Code of the paper

Thank you for taking the time to review this code. The code can be run directly in Windows, and all experiments can be replicated by running the code. In the event that our paper is accepted for publication, we plan to release all of our codes as open source. However, for safety concerns, we have encrypted some of the code for review.

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

Paper Hash code:
29e64b0b81f4ef4d77ddafccaec5adf9

Appendix Hash code:
0d41beb017af3dc2eef8550b32adc1c1
