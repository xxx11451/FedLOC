This repository provides code for FedLOC, a compress algorithm for federated learning based on layer output.

### Requirement

* python 3.9.12
* pytorch 1.12.1
* torchvision 0.13.1
* cuda 10.2
* fedlab 1.1.2

### Command

This example is running these compress methods based on CIFAR10 dataset and vgg11 model, it can use CIFAR100 and FMNIST dataset by change `--data CIFAR100` or `--data FMNIST`.
The experiment result can be shown in the result folder.

```cmake
python server.py --gpu 0 --data CIFAR10 --model vgg11  --method Topk --k 0.01 --alpha 1.0 -lr 0.01
python server.py --gpu 0 --data CIFAR10 --model vgg11  --method FedLOC --k 0.01 --alpha 1.0 -lr 0.01
python server.py --gpu 0 --data CIFAR10 --model vgg11  --method STC --k 0.01 --alpha 1.0 -lr 0.01
python server.py --gpu 0 --data CIFAR10 --model vgg11  --method STCLOC --k 0.01 --alpha 1.0 -lr 0.01
python server.py --gpu 0 --data CIFAR10 --model vgg11  --method SBC --k 0.01 --alpha 1.0 -lr 0.01
python server.py --gpu 0 --data CIFAR10 --model vgg11  --method SBCLOC --k 0.01 --alpha 1.0 -lr 0.01
python server.py --gpu 0 --data CIFAR10 --model vgg11  --method FedCAMS --k 0.01 --alpha 1.0 -lr 0.01
python server.py --gpu 0 --data CIFAR10 --model vgg11  --method FedCAMSLOC --k 0.01 --alpha 1.0 -lr 0.01
```
