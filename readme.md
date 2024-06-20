This repository provides code for FedLOC, a compress algorithm for federated learning based on layer output.

### Requirement

* python 3.9.12
* pytorch 1.12.1
* torchvision 0.13.1
* cuda 10.2
* fedlab 1.1.2

### Command

This example is running these compress methods based on CIFAR10 dataset and vgg11 model,The experiment result can be shown in the `result` folder.
'-data': CIFAR10, CIFAR100, FMNIST
'-model':vgg11, CNN
'-method':Topk, FedLOC, STC, SBC, FedCAMS, STCLOC, SBCLOC, FedCAMSLOC
'-k': sparsity rate
'-alpha': Dirichlet parameter
'-lr': learning rate
'-b': local batch size
'-comm': the number of training rounds
'-e': the number of local epochs
'-nc': the number of total clients
'-pf': the proportion of participants in all of cliants

```cmake
python server.py  --data CIFAR10 --model vgg11  --method Topk --k 0.01 --alpha 1.0 -lr 0.01
python server.py  --data CIFAR10 --model vgg11  --method FedLOC --k 0.01 --alpha 1.0 -lr 0.01
python server.py  --data CIFAR10 --model vgg11  --method STC --k 0.01 --alpha 1.0 -lr 0.01
python server.py  --data CIFAR10 --model vgg11  --method STCLOC --k 0.01 --alpha 1.0 -lr 0.01
python server.py  --data CIFAR10 --model vgg11  --method SBC --k 0.01 --alpha 1.0 -lr 0.01
python server.py  --data CIFAR10 --model vgg11  --method SBCLOC --k 0.01 --alpha 1.0 -lr 0.01
python server.py  --data CIFAR10 --model vgg11  --method FedCAMS --k 0.01 --alpha 1.0 -lr 0.01
python server.py  --data CIFAR10 --model vgg11  --method FedCAMSLOC --k 0.01 --alpha 1.0 -lr 0.01
```
