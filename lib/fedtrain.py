import numpy as np
import torch
from torch.utils.data import DataLoader
from fedlab.utils.dataset.sampler import SubsetSampler
import time
import lib.compressor
from lib.compressor import FedLOC,Topk,SBC,SBCLOC

class Client_message():
    def __init__(self,update_parameter:dict,sample_num:int) -> None:
        self.update_parameter = update_parameter
        self.sample_num = sample_num
        pass


class Fedtrainer(object):
    def __init__(self, trainDataset, dev,id,model_parameters,learning_rate):
        self.train_dataset = trainDataset
        self.dev = dev
        self.id =id
        self.train_dataloader = None
        self.local_parameters = None
        self.learning_rate = learning_rate

    def localUpdate(self,round,localEpoch, localBatchSize, model,model_name, lossFun, learning_rate, global_parameters,layer_name,data_slices,channel_flag,k):
        start_time = time.time()
        model.load_state_dict(dict(zip(layer_name,global_parameters)), strict=True)
        self.train_dataloader = DataLoader(self.train_dataset, sampler=SubsetSampler(indices=data_slices,shuffle=True
                                      ),batch_size=localBatchSize)
        sample_num = len(data_slices)
        torch.set_num_threads(9)
        model.train()
        if channel_flag == "FedCAMS" or channel_flag == 'OFedCAMS':
            opti = torch.optim.SGD(model.parameters(),lr=learning_rate,momentum=0)
        else:
            opti = torch.optim.SGD(model.parameters(),lr=learning_rate,momentum=0.9)
        for epoch in range(localEpoch):
            for data, label in self.train_dataloader:
                opti.zero_grad()
                data, label = data.to(self.dev), label.to(self.dev)
                predictions = model(data)
                loss = lossFun(predictions, label)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),max_norm=20)
                opti.step()
        new_parameter = []
        for key,val in model.state_dict().items():
            new_parameter.append(val)
        for i in range(len(new_parameter)):
            new_parameter[i] = new_parameter[i] - global_parameters[i]
        if channel_flag == 'FedLOC' or channel_flag == 'STCLOC' or channel_flag == "FedCAMSLOC":
            new_parameter = FedLOC(model,model_name,self.dev,self.train_dataloader,new_parameter,layer_name,sample_num,k=k)
        if channel_flag == 'Topk' or channel_flag == 'STC' or channel_flag == "FedCAMS":
            new_parameter = Topk(new_parameter,k)
        if channel_flag == "SBC":
            new_parameter = SBC(new_parameter,k)
        if channel_flag == "SBCLOC":
            new_parameter = SBCLOC(model,model_name,self.dev,self.train_dataloader,new_parameter,layer_name,sample_num,k=k)
        if channel_flag == 'None':
            return Client_message(new_parameter,sample_num)
        if 'STC' in channel_flag:
            for index,value in enumerate(new_parameter):
                sum_value = torch.sum(torch.abs(value))
                number_element = ( (torch.nonzero(value).size()[0]) if (torch.nonzero(value).size()[0]) > 0 else 1 )
                mu = sum_value / number_element
                torch.cuda.empty_cache()
                new_parameter[index] = torch.sign(value) * mu
                torch.cuda.empty_cache()
            return Client_message(new_parameter,sample_num)
        return Client_message(new_parameter,sample_num)