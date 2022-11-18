import torch.nn as nn
import numpy as np
from torch.optim import SGD, Adadelta, Adagrad, Adam, RMSprop
from model.models import ResNeSt_parallel, Efficient_parallel, Efficient, ResNeSt, Dense, Dense_parallel
from resnest.torch import resnest50, resnest101, resnest200, resnest269
from efficientnet_pytorch import EfficientNet
from torchvision.models import densenet121, densenet161, densenet169, densenet201


def get_optimizer(params, cfg):
    if cfg.optimizer == 'SGD':
        return SGD(params, lr=cfg.lr, momentum=cfg.momentum,
                   weight_decay=cfg.weight_decay)
    elif cfg.optimizer == 'Adadelta':
        return Adadelta(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    elif cfg.optimizer == 'Adagrad':
        return Adagrad(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    elif cfg.optimizer == 'Adam':
        return Adam(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    elif cfg.optimizer == 'RMSprop':
        return RMSprop(params, lr=cfg.lr, momentum=cfg.momentum,
                       weight_decay=cfg.weight_decay)
    else:
        raise Exception('Unknown optimizer : {}'.format(cfg.optimizer))

def get_model(cfg):
    if cfg.backbone == 'resnest':
        childs_cut = 9
        if cfg.id == '50':
            pre_name = resnest50
        elif cfg.id == '101':
            pre_name = resnest101
        elif cfg.id == '200':
            pre_name = resnest200
        else:
            pre_name = resnest269
        pre_model = pre_name(pretrained=cfg.pretrained)
        for param in pre_model.parameters():
            param.requires_grad = True
        if cfg.split_output:
            model = ResNeSt(pre_model, cfg)
        else:
            model = ResNeSt_parallel(pre_model, 5)
    elif cfg.backbone == 'efficient' or cfg.backbone == 'efficientnet':
        childs_cut = 6
        pre_name = 'efficientnet-'+cfg.id
        if cfg.pretrained:
            pre_model = EfficientNet.from_pretrained(pre_name)
        else:
            pre_model = EfficientNet.from_name(pre_name)
        for param in pre_model.parameters():
            param.requires_grad = True
        if cfg.split_output:
            model = Efficient(pre_model, cfg)
        else:
            model = Efficient_parallel(pre_model, 5)
    elif cfg.backbone == 'dense' or cfg.backbone == 'densenet':
        childs_cut = 2
        if cfg.id == '121':
            pre_name = densenet121
        elif cfg.id == '161':
            pre_name = densenet161
        elif cfg.id == '169':
            pre_name = densenet169
        else:
            pre_name = densenet201
        pre_model = pre_name(pretrained=cfg.pretrained)
        for param in pre_model.parameters():
            param.requires_grad = True
        if cfg.split_output:
            model = Dense(pre_model, cfg)
        else:
            model = Dense_parallel(pre_model, 5)
    else:
        raise Exception("Not support this model!!!!")
    print(f"Model Architecture: {model}")
    print(f"Childs_cut is : {childs_cut}")
    return model, childs_cut

def get_str(metrics, mode, s):
    for key in list(metrics.keys()):
        if key == 'loss':
            s += "{}_{} {:.3f} - ".format(mode, key, metrics[key])
        else:
            x= list(map(lambda x :str(x),metrics[key].round(5)))
            # print(x)
            # metric_str = ' '.join(list(map(lambda x: '{:.5f}'.format(str(x)), metrics[key])))
            metric_str = ' '.join(x)
            # print(f"Metric_str: {metric_str}")
            s += "{}_{} {} - ".format(mode, key, metric_str)
    s = s[:-2] + '\n'
    return s


def tensor2numpy(input_tensor):
  x =[]
  for k in input_tensor:
    x.append(k.cpu().detach().numpy())
  return np.asarray(x)
    # device cuda Tensor to host numpy
    # x =[]
    # t =[]
    # for i in input_tensor:
    #   x.append(i.cuda())
    # print(x)
    # for i in x:
    #   t.append(i.cpu().detach().numpy())  

    # # y = tuple(t.cpu() for t in x)
    # # print(f"Prasun: {input_tensor.get_shape()}")
    # # x = input_tensor.cuda()
    # # x.cpu()
    # # x = (torch.randn([1, 3, 244, 244]).cuda(),
    # #     torch.randn([1, 244, 244]).cuda(),
    # #     torch.randn([1, 244, 244]).cuda())

    # # x.cpu() # error

    # # y = tuple(t.cpu() for t in x)
    # # print(y)
    # print(t)
    # return t
