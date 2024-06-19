import numpy as np
import torch


def compute_TF(out):
    for key,value in out.items():
        out[key] = value.sum(dim=0)
    return out

def TF_add(old_TF,new_TF):
    if old_TF is None:
        old_TF = {}
        for key,value in new_TF.items():
            old_TF[key] = value
    else:
        for key,value in new_TF.items():
            old_TF[key] = (old_TF[key] + value)
    return old_TF
        
def none_top_k(update_parameters,k)->dict:
    mask={}
    for key,value in update_parameters.items():
        mask[key] = torch.ones_like(value)
    return mask



def Topk(update_parameters,k)->dict:
    for i,value in enumerate(update_parameters):
        size = value.shape
        temp = value.reshape(-1)
        sparse_size = max(int(k*temp.shape[0]),1)
        z = torch.abs(temp)
        index = torch.argsort(z,descending=True)[sparse_size:]
        del z
        temp[index] = 0
        update_parameters[i] = temp.reshape(size)
    torch.cuda.empty_cache()
    return update_parameters

def STC(update_parameters,k)->dict:
    mask={}
    for key,value in update_parameters.items():
        size = value.shape
        temp = value.reshape(-1)
        _,index = torch.topk(torch.abs(temp),int(k*temp.shape[0]))
        mask[key] = torch.zeros_like(temp,dtype = torch.bool)
        mask[key][index] = 1
        mask[key] = mask[key].reshape(size)
    torch.cuda.empty_cache()
    return mask



def compute_linear(parameter:torch.tensor,input:torch.tensor,k):
    size = parameter.shape
    parameter = parameter.view(-1)
    sparse_size = max(int(k*parameter.shape[0]),1)
    parameter = parameter.view(size)
    for i in range(parameter.shape[0]):
        parameter[i,:][:] = input*parameter[i,:]
    parameter = parameter.view(-1)
    z = torch.abs(parameter)
    index = torch.argsort(z,descending=True)[0:sparse_size]
    del z
    result = torch.zeros_like(parameter,dtype = torch.bool)
    result[index] = 1
    result = result.view(size)
    torch.cuda.empty_cache()
    return result

def SBCLOC_compute_linear(parameter:torch.tensor,input:torch.tensor,k):
    size = parameter.shape
    parameter = parameter.reshape(-1)
    _,index = torch.topk(parameter,int(3*k*parameter.shape[0]))
    result = torch.zeros_like(parameter,dtype = torch.bool)
    torch.cuda.empty_cache()
    result[index] = 1
    parameter = parameter*result
    parameter = parameter.reshape(size)
    for i in range(parameter.shape[0]):
        parameter[i,:] = input*parameter[i,:]
    parameter = parameter.reshape(-1)
    value,index = torch.topk(parameter,int(k*parameter.shape[0]))
    torch.cuda.empty_cache()
    return index

def get_inputresult(model,dev,dataloader)->dict:
    sum_TF=None
    model.eval()
    features_in_hook={}
    def getinput(name):
        def hook(module, fea_in, fea_out):
            features_in_hook[name] = fea_in       
        return hook
    for (name, module) in model.named_modules():
        if 'Linear' in str(type(module)):
            module.register_forward_hook(hook=getinput(name))
    with torch.no_grad():
        for data,label in dataloader:
            data, label = data.to(dev), label.to(dev)
            out = model(data)
            result={}
            for key,value in features_in_hook.items():
                result[key] = value[0]
            TF=compute_TF(result)
            sum_TF = TF_add(sum_TF,TF) 
    return sum_TF


def single_SBC(parameter:torch.tensor,k)->dict:
    size = parameter.size()
    temp = parameter.reshape(-1)
    _,index_positive = torch.topk(temp,k=int(k*temp.shape[0]))
    _,index_passitive = torch.topk(-1*temp,k=int(k*temp.shape[0]))
    torch.cuda.empty_cache()
    mask = torch.zeros_like(temp)
    mask[index_positive] = 1
    mask[index_passitive] = 1
    temp = mask*temp
    mu_positive = temp[temp>0].sum() / ((temp>0).sum() + 1)
    mu_passitive = temp[temp<0].sum() / ((temp<0).sum() + 1)
    torch.cuda.empty_cache()
    if mu_positive > abs(mu_passitive):
        return (mu_positive * (temp > 0) ).reshape(size)
    else:
        return (mu_passitive * (temp < 0) ).reshape(size)

def single_SBCLOC(parameter:torch.tensor, input:torch.tensor, k)->dict:
    index_positive = SBCLOC_compute_linear(parameter,input,k)
    index_passitive = SBCLOC_compute_linear(-1*parameter,input,k)
    size = parameter.size()
    temp = parameter.reshape(-1)
    mask = torch.zeros_like(temp)
    mask[index_positive] = 1
    mask[index_passitive] = 1
    temp = mask*temp
    mu_positive = temp[temp>0].sum() / ((temp>0).sum() + 1)
    mu_passitive = temp[temp<0].sum() / ((temp<0).sum() + 1)
    torch.cuda.empty_cache()
    if mu_positive > abs(mu_passitive):
        return (mu_positive * (temp > 0) ).reshape(size)
    else:
        return (mu_passitive * (temp < 0) ).reshape(size)




def FedLOC(model,model_name,dev,dataloader,update_parameters,layer_name,sample_num,k)->dict:
    sum_TF = get_inputresult(model,dev,dataloader)
    for key,value in sum_TF.items():
        sum_TF[key] = sum_TF[key] / sample_num
    torch.cuda.empty_cache()
    mask={}
    name_set = []
    for key,_ in sum_TF.items():
        name_set.append(key)
    with torch.no_grad():
        for i,value in enumerate(update_parameters):
            key = layer_name[i]
            if key[0:-7] in name_set:
                if 'weight' in key:
                    mask = compute_linear(value,sum_TF[key[0:-7]],k)
                    update_parameters[i] = value * mask
                if 'bias' in key:
                    temp = value.reshape(-1)
                    sparse_size = max(int(k*temp.shape[0]),1)
                    _,index = torch.topk(torch.abs(temp),sparse_size)
                    mask = torch.zeros_like(temp,dtype=torch.bool)
                    mask[index] = 1
                    mask = mask.reshape(value.shape)
                    update_parameters[i] = value * mask
            else:
                temp = value.reshape(-1)
                sparse_size = max(int(k*temp.shape[0]),1)
                _,index = torch.topk(torch.abs(temp),sparse_size)
                mask = torch.zeros_like(temp,dtype=torch.bool)
                mask[index] = 1
                mask = mask.reshape(value.shape)
                update_parameters[i] = value * mask
    return update_parameters

def SBC(update_parameters,k)->dict:
    for index,value in enumerate(update_parameters):
        update_parameters[index] = single_SBC(value,k)
        torch.cuda.empty_cache()
    return update_parameters

def SBCLOC(model,model_name,dev,dataloader,update_parameters,layer_name,sample_num,k):
    sum_TF = get_inputresult(model,dev,dataloader)
    for key,value in sum_TF.items():
        sum_TF[key] = sum_TF[key] / sample_num
    mask={}
    name_set = []
    for key,_ in sum_TF.items():
        name_set.append(key)
    for index,value in enumerate(update_parameters):
        key = layer_name[index]
        if key[0:-7] in name_set:
            if 'weight' in key:
                update_parameters[index] = single_SBCLOC(value,sum_TF[key[0:-7]],k)
            if 'bias' in key:
                update_parameters[index] = single_SBC(value,k)
        else:
            update_parameters[index] = single_SBC(value,k)
    return update_parameters