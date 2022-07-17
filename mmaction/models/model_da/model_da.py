



import torch.nn as nn
from torch.nn.init import *
from torch.autograd import Function
import numpy as np
import random

def init_layer( layer, std = 0.001):
    normal_(layer.weight, 0, std)
    constant_(layer.bias, 0)
    return layer

class LinearWeightedAvg(nn.Module):
    def __init__(self, n_inputs):
        super(LinearWeightedAvg, self).__init__()
        self.weights = nn.ParameterList( [nn.Parameter(torch.randn(1))  for i in range(n_inputs)  ] )
    def forward(self, input):
        res = 0
        for emb_idx, emb in enumerate(input):
            res += emb * self.weights[emb_idx]
        return res



class Cls_Head(nn.Module):
    def __init__(self, embed_dim, mlp_dim, n_class ):
        super(Cls_Head, self).__init__()
        self.fc1 = nn.Linear( embed_dim, mlp_dim )
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(mlp_dim, n_class)
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# definition of Gradient Reversal Layer
class GradReverse(Function):
    @staticmethod
    def forward(ctx, x, beta):
        ctx.beta = beta
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.neg() * ctx.beta
        return grad_input, None


class DA_clsfr(nn.Module):
    def __init__(self, embed_dim,  mlp_dim):
        super(DA_clsfr, self).__init__()

        self.fc1 = nn.Linear(embed_dim, mlp_dim)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(mlp_dim, 2)

    def forward(self, feat, beta):
        feat = GradReverse.apply( feat,  beta) # apply gradient reversal layer
        feat = self.fc1(feat)
        feat = self.relu(feat)
        pred_d = self.fc2(feat)
        return pred_d


def adapt_weight( iter_now, iter_max, weight_value=10.0, high_value=1.0, low_value=0.0):
    """
    compute adaptive weights for
                        iter_max_input       weight_loss
    adaptive beta 0          65000                   -2      iter_max = iter_max_input   frame-level adversarial loss
    adaptive beta 1          50000                  -2     iter_max = iter_max_input      segment-level adversarial loss
    adaptive gamma         65000                -2      iter_max = iter_max_input   weighting for semantic loss
    adaptive nu         16000000              -2       iter_max = iter_max_input   weighting for the discrepancy loss

    :param iter_now: 	   iteration id in the entire training process
    :param iter_max_default:   n_epochs * n_iterations per epoch
    :param iter_max_input:	   iter_max_beta_user = [iter_max_0, iter_max_1]
                            50salads [20000 25000] breakfast [65000 50000] gtea [2000 1400]
                            50salads 40x50=2000    breakfast 1400x50=70000  gtea  21x50 = 1050
    :param weight_loss:	   beta = -2 or gamma = -2
    :param weight_value:       gamma = 10 in page 6 of http://proceedings.mlr.press/v37/ganin15.pdf
    :param high_value:         1.0
    :param low_value:          0.0
    :return:
    """
    # affect adaptive weight value
    # adpative beta values are the weights for different loss terms
    # page 6 in http://proceedings.mlr.press/v37/ganin15.pdf
    # in order to suppress noisy signal from the domain classifier at the early stages instead of fixing the weight value
    # we gradually change it from  low_value (0.0) to high_value (1.0)

    # if weight_loss < -1:
    #     # for all the 3 datasaets, weight_loss (beta) for both frame-wise and segment-wise domain prediction is set to -2
    #     iter_max = iter_max_input   # the given iter_max_beta_user
    # else:
    #     iter_max = iter_max_default  # number of epochs * number of iterations per epoch


    high = high_value
    low = low_value
    weight = weight_value
    p = float(iter_now) / iter_max
    adaptive_weight = (2. / (1. + np.exp(-weight * p)) - 1) * (high-low) + low
    return adaptive_weight


def prepare_domain_labels( pred_domain_f_source,pred_domain_f_target ):
    source_domain_label_f = torch.zeros(pred_domain_f_source.size(0)).long()  # (batch_s*n_frames, 2)
    target_domain_label_f = torch.ones(pred_domain_f_target.size(0)).long()  # (batch_t*n_frames, 2)
    domain_label_f = torch.cat((source_domain_label_f, target_domain_label_f), 0).cuda( non_blocking=True)  # (batch_s*n_frames + batch_t*n_frames, )
    return domain_label_f


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].contiguous().view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res



def model_analysis(model, logger):
    print("Model Structure")
    print(model)

    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    logger.debug('#################################################')
    logger.debug(f'Number of trainable parameters: {params}')
    logger.debug('#################################################')


class MemoryBank(object):
    def __init__(self, n_classes, source_list_file = None, buffer_size = None):
        self.n_classes = n_classes
        self.buffer_size = buffer_size  #  the capacity of the buffer
        self._parse_list(source_list_file)
        self._init_buffer()

    def _parse_list(self, list_file):
        self.sample_list = []  # todo a list of img_path and label
        for line in open(list_file):
            items = line.strip('\n').split(' ')
            img_path, label = items[0], int(items[1])
            # vidname, imgname = img_path.split('/')[-2], img_path.split('/')[-1].split('.')[0]
            # self.sample_list.append( (f'{vidname}_{imgname}', label) )
            self.sample_list.append( (img_path, label) )


    def _init_buffer(self):
        # the buffer that collects all the features in the source domain
        self.buffer_dict = dict()
        for idx in range(self.n_classes):
            self.buffer_dict.update( {idx: dict()}) # a dictionary that contains all the classes

    def update_buffer(self, batch_index, batch_feat):
        pass
        # todo  for each source video, add  [num_clips]  clip features

    def random_sample(self, class_id):
        pass
        # todo  for each target video,  we randomly sample  [num_clips]  source clip feature of the same class id
        #   the contrastive loss is computed on each target clip,  each target video has  [num_clips]  target clips with the same label