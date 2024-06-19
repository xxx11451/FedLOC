import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch import optim
from torch import nn
from lib.model import CNN,vgg11_CIFAR100,vgg11_CIFAR10
from lib.fedtrain import Fedtrainer,Client_message
from math import sqrt
import torchvision
from torchvision import datasets, transforms
from fedlab.utils.dataset.partition import CIFAR10Partitioner,MNISTPartitioner,FMNISTPartitioner,CIFAR100Partitioner
import csv
import time

database = 'mnist'

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")



parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, description="FedAvg")
parser.add_argument('-g', '--gpu', type=str, default='0')
parser.add_argument('-nc', '--num_of_clients', type=int, default=100)
parser.add_argument('-pf', '--participants_fraction', type=float, default=0.1,
                    help='the proportion of participants in all of cliants')
parser.add_argument('-e', '--local_epoch', type=int, default=3, help='local epoch round of one participant')
parser.add_argument('-b', '--batch_size', type=int, default=32, help='local train batch size')
parser.add_argument('-lr', "--learning_rate", type=float, default=0.001)
parser.add_argument('-dc', "--learning_rate_decay", type=float, default=0.995)
parser.add_argument('-comm', '--num_of_communication_rounds', type=int, default=600)
parser.add_argument('-alpha', '--alpha', type=float, default=1.0)
parser.add_argument('--data', type=str, default="MNIST")
parser.add_argument('--model', type=str,default='CNN')
parser.add_argument('--method', type=str, default='Topk')
parser.add_argument('--change_size', type=str2bool, default=False)
parser.add_argument('--k', type=float, default=0.05)


def get_args(args):
    gpu_num = args['gpu']
    total_clients = args['num_of_clients']
    participants_fraction = args['participants_fraction']
    local_epoch = args['local_epoch']
    batch_size = args['batch_size']
    learning_rate = args['learning_rate']
    learning_rate_decay = args['learning_rate_decay']
    total_round = args['num_of_communication_rounds']
    alpha = args['alpha']
    data = args['data']
    model = args['model']
    method = args['method']
    change_size=args['change_size']
    top_k = args['k']
    return gpu_num, total_clients, participants_fraction, local_epoch, batch_size, learning_rate, learning_rate_decay\
        , total_round, alpha,data,method,top_k,model,change_size


def test_mkdir(path):
    if not os.path.isdir(path):
        os.mkdir(path)


def create_excel_file(path, alpha):
    record_time = time.strftime("_%Y-%m-%d-%H-%M-%S", time.localtime())
    file_name = path + "/" + data + '_method='+ str(method) +'_model=' + str(model_name) +  '_k=' + str(k) \
          + '_alpha='+ str(alpha) +'_batch_size=' + str(batch_size) + '_client=' + str(total_clients) + 'in' + str(participants_num) \
            + "_lr=" + str(learning_rate)+'.csv'
    print(file_name)
    with open(file_name, "a", newline='') as csvfile:
        csvfile.truncate(0)
        writer = csv.writer(csvfile)
        writer.writerow(['time', 'accuracy','loss'])
        csvfile.close()
    return file_name


def update_excel_file(path, round_accuracy, loss,i):
    record_list = [i, format(round_accuracy),format(loss)]
    with open(path, "a", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(record_list)
        csvfile.close()


def activate_gpu(model, gpu_number):
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_number
    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.device_count() > 1:
        print("Use", torch.cuda.device_count(), "GPUs")
        model = torch.nn.DataParallel(model)
    return model.to(dev), dev

def create_model(model_name):
    model = None
    if model_name == 'CNN':
        model = CNN()
    if model_name == 'vgg11':
        if data == "CIFAR10":
            model = vgg11_CIFAR10
        if data == "CIFAR100":
            model = vgg11_CIFAR100           
    return model

def initial_model(model):
    parameters = []
    layer_name = []
    for model_key, value in model.state_dict().items():
        layer_name.append(model_key)
        parameters.append(value.clone().float())
    return parameters,layer_name


def top_k_update_aggregated_parameters(client:Client_message,aggregated_parameters:dict,aggregated_sample:dict):
    if aggregated_parameters is None:
        aggregated_parameters = []
        sum_t = 0
        for index,value in enumerate(client.update_parameter):
                aggregated_parameters.append(value*client.sample_num)
    else:
        for index,value in enumerate(client.update_parameter):
            aggregated_parameters[index] = aggregated_parameters[index] + value*client.sample_num
    if aggregated_sample is None:
        aggregated_sample = []
        for index,value in enumerate(client.update_parameter):
            aggregated_sample.append(torch.ones_like(value))
            aggregated_sample[index] = aggregated_sample[index] + (value != 0)*client.sample_num 
    else:
        for index,value in enumerate(client.update_parameter):
            aggregated_sample[index] = aggregated_sample[index] + (value != 0)*client.sample_num
    return aggregated_parameters,aggregated_sample



def update_aggregated_parameters(clients_message:list[Client_message]):
    aggregated_parameters=None
    for client in clients_message:
        if aggregated_parameters is None:
            aggregated_parameters = {}
            for key,value in client.update_parameter.items():
                aggregated_parameters[key] = value*client.sample_num
        else:
            for key,value in client.update_parameter.items():
                aggregated_parameters[key] = aggregated_parameters[key] + value*client.sample_num
    aggregated_sample=None
    for client in clients_message:
        if aggregated_sample is None:
            aggregated_sample = {}
            for key,value in client.mask.items():
                aggregated_sample[key] = torch.ones_like(value)*client.sample_num
        else:
            for key,value in client.mask.items():
                aggregated_sample[key] = aggregated_sample[key] + torch.ones_like(value)*client.sample_num
    for key,value in aggregated_parameters.items():
        aggregated_parameters[key] = value / aggregated_sample[key]
    return aggregated_parameters


def update_global_parameters(gobal_parameters, aggregated_parameters):
    for i in range(len(global_parameters)):
        gobal_parameters[i] = gobal_parameters[i] + aggregated_parameters[i]
    return gobal_parameters


def get_round_result(model, global_parameters,layer_name,loss_function,testDataLoader):
    with torch.no_grad():
        model.load_state_dict(dict(zip(layer_name,global_parameters)), strict=True)
        model.eval()
        model.to(dev)
        accuracy = 0
        num = 0
        sum_loss = 0
        for data, label in testDataLoader:
            data, label = data.to(dev), label.to(dev)
            prediction = model(data)
            loss = loss_function(prediction,label)
            prediction = torch.argmax(prediction, dim=1)
            accuracy += (prediction == label).float().sum()
            num += label.shape[0]
            sum_loss += loss.item()
        accuracy = accuracy / num
        sum_loss = sum_loss / num
    return accuracy,sum_loss

def load_data():
    if data == "FMNIST":
        apply_transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.1307, ), (0.3081, ))])
        train = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=apply_transform)
        test  = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=apply_transform)
        train_parti = FMNISTPartitioner(train.targets,
                                        total_clients,
                                        partition="noniid-labeldir",
                                        dir_alpha=alpha,
                                        seed=2024)
        return train,test,train_parti
    if data == "MNIST":
        apply_transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.1307, ), (0.3081, ))])
        train = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=apply_transform)
        test  = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=apply_transform)
        train_parti = MNISTPartitioner(train.targets,
                                        total_clients,
                                        partition="noniid-labeldir",
                                        dir_alpha=alpha,
                                        seed=2024)
        return train,test,train_parti
    if data == 'CIFAR10':
        if model_name == 'vgg16' or model_name == 'vgg11':
            train = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transforms.Compose([
                        transforms.ToTensor(),
                            transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225))
                        ]))
            test = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transforms.Compose([
                        transforms.ToTensor(),
                            transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225))
                        ]))
        train_parti = CIFAR10Partitioner(train.targets,
                                        total_clients,
                                        balance=None,
                                        partition="dirichlet",
                                        dir_alpha=alpha,
                                        seed=2024)
        return train,test,train_parti
    if data == "CIFAR100":
        mean = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
        std = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)
        transform_train = transforms.Compose([
        #transforms.ToPILImage(),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
        transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
        train = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
        test = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
        train_parti = CIFAR100Partitioner(train.targets,
                                        total_clients,
                                        balance=None,
                                        partition="dirichlet",
                                        dir_alpha=alpha,
                                        seed=2024)        
        return train,test,train_parti

def FedAMS_update_global_parameters(global_parameters, aggregated_parameters,model_m,model_v,model_v_hat):
    for index in range(len(model_m)):
        model_m[index] = beta_1*model_m[index] + (1-beta_1)*aggregated_parameters[index].cpu()
        model_v[index] = beta_2*model_v[index] + (1-beta_2)*aggregated_parameters[index].cpu()*aggregated_parameters[index].cpu()
        model_v_hat[index] = torch.max(model_v_hat[index],model_v[index])
        denom = torch.sqrt(model_v_hat[index]) / sqrt(bias_corrections_2) + eta
        global_parameters[index] = global_parameters[index] + step_size*(model_m[index] / denom ).cuda()
    return global_parameters,model_m,model_v,model_v_hat





args = parser.parse_args()
args = args.__dict__

gpu_num, total_clients, participants_fraction, local_epoch, batch_size, learning_rate, learning_rate_decay \
        , total_round, alpha,data,method,k,model_name,change_size = get_args(args)
print(method)
total_clients = 100
participants_num = 10
path = './result'+"/" + str(data) +"_" + str(model_name)
beta_1 = 0.9
beta_2 = 0.99
eta = 0.001
bias_corrections_1 = 1
bias_corrections_2 = 1
step_size = 1





if __name__ == "__main__":
    test_mkdir(path)
    excel_file_name = create_excel_file(path, alpha)
    train,test,data_slice = load_data()
    testload =  torch.utils.data.DataLoader(dataset=test,
                                           batch_size=128,
                                           shuffle=True)
    loss_function = F.cross_entropy
    model = create_model(model_name)
    print(model_name)
    model, dev = activate_gpu(model, gpu_num)
    clients_group = [Fedtrainer(trainDataset= train,dev=dev,id =i,model_parameters=model.parameters(),learning_rate=learning_rate) for i in range(total_clients)]
    global_parameters,layer_name = initial_model(model)
    client_parameters=[]
    model_m = []
    model_v = []
    model_v_hat = []
    if(method == 'FedCAMS' or method == "OFedCAMS"):
        for index,value in enumerate(global_parameters):
            model_m.append(torch.zeros_like(value).cpu())
            model_v.append(torch.zeros_like(value).cpu())
            model_v_hat.append(torch.zeros_like(value).cpu())
    opti = optim.SGD(model.parameters(), lr=learning_rate)
    lr = learning_rate
    for round in range(total_round):
        order = np.random.permutation(total_clients)
        participants = [i for i in order[0:participants_num]]
        begin_time = time.time()
        aggregated_parameters=None
        aggregated_sample = None
        bias_corrections_1 = 1 - beta_1 ** (round+1)
        bias_corrections_2 = 1 - beta_2 ** (round+1)
        step_size = lr / bias_corrections_1
        for participant in participants:
            client_message = clients_group[participant].localUpdate(round,local_epoch, batch_size, model,model_name,
                                                                    loss_function, lr, global_parameters,layer_name,data_slice[participant],method,k)
            aggregated_parameters,aggregated_sample = top_k_update_aggregated_parameters(client_message,aggregated_parameters,aggregated_sample)
        for index in range(len(aggregated_parameters)):
            aggregated_parameters[index] = aggregated_parameters[index] / aggregated_sample[index]
        if(method == 'FedCAMS' or method == "OFedCAMS"):
            global_parameters,model_m,model_v,model_v_hat = FedAMS_update_global_parameters(global_parameters,\
                                                            aggregated_parameters,model_m,model_v,model_v_hat)
        else:
            global_parameters = update_global_parameters(global_parameters, aggregated_parameters)
        round_accuracy,loss = get_round_result(model, global_parameters,layer_name, loss_function,testload)
        update_excel_file(excel_file_name, round_accuracy, loss,round)
        lr *= 0.996
