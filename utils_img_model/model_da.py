
import torch.nn as nn
from torch.nn.init import *
from torch.autograd import Function
import numpy as np
# from utils_img_cls import freeze_resnet
# from torchvision.models.feature_extraction import create_feature_extractor
import time
import copy
from sklearn.metrics import classification_report
from utils_img_model.MyImageFolder import generate_tsn_frames, ImageFilelist, generate_target_frame_list
from utils_img_model.memory_bank import MemoryBank
import os
import random

def process_pred_dict(  pred_scores_dict,  logger, n_class = None,  result_dir =None, log_time = None, epoch = None ):
    """

    :param pred_scores_dict:  # each video has a tuple of  gt_label (n_frames, ),   pred_label (n_frames, ) ,  max_pred_score (n_frames, ),  all_class_scores  (n_frames, n_class )
    :param logger:
    :param n_class:
    :param result_dir:
    :param log_time:
    :return:
    """
    log_time = log_time if epoch is None else f'{log_time}_epoch{epoch}'

    np.save( os.path.join( result_dir, f'{log_time}_frame_ps.npy' ),  pred_scores_dict  )

    final_results_write = open(os.path.join(result_dir, f'{log_time}_ps_quality.txt'), 'w+')
    n_correct_total = 0
    n_correct_class_total = np.zeros((n_class,))
    n_samples_class_total = np.zeros((n_class,))
    acc_class_total = np.zeros((n_class,))
    for vidname, items in pred_scores_dict.items():
        # vidname : ( gt label, pred label,  max_pred_score, predicted scores for all classes )
        gt_label_seq, pred_label_seq, max_pred_score_seq = items[0], items[1], items[2]
        n_frames = len(gt_label_seq)
        assert len(set(gt_label_seq)) == 1  # check if all the gt labels in a video are identical
        gt_label = int(gt_label_seq[0])

        n_samples_class_total[gt_label] += n_frames
        n_correct_frames = np.sum(  gt_label_seq == pred_label_seq )
        n_correct_total += n_correct_frames
        n_correct_class_total[gt_label] += n_correct_frames
    n_frames_total = np.sum( n_samples_class_total)
    acc_total = float(n_correct_total) / n_frames_total

    for class_idx in range(n_class):
        acc_class_total[class_idx] = float(n_correct_class_total[class_idx]) / n_samples_class_total[class_idx]
        final_results_write.write(
            f'class {class_idx }: #frames {n_samples_class_total[class_idx]}, acc {acc_class_total[class_idx]:.3f}\n')
    final_results_write.write('Total:\n')
    final_results_write.write(f'Total : #frames { n_frames_total }/100% , acc {acc_total:.3f}\n')
    # compute accuracy of pseudo labels w.r.t thresholding
    final_results_write.write('\n')


    n_vid_correct = 0

    # todo  video-level confidence score is the average of frame-level confidence scores
    for vidname, items in pred_scores_dict.items():
        vid_confidence = np.mean( items[3], axis= 0 ) # (n_class, )
        gt_label = items[0][0]
        vid_ps_label = np.argmax(vid_confidence )
        if gt_label == vid_ps_label:
            n_vid_correct += 1
    vid_ps_acc = float(n_vid_correct) / len( pred_scores_dict)
    to_print1 = 'Video-level pseudo labels, derived from the average of frame-level confidence scores'
    to_print2 = f'# correct vids {n_vid_correct} / #vids total {len(pred_scores_dict)} : {vid_ps_acc:.3f}'
    final_results_write.write(f'{to_print1}\n')
    final_results_write.write(f'{to_print2}\n')
    logger.debug(to_print1)
    logger.debug(to_print2)

    final_results_write.write('\n')



    confidence_list = []
    for vidname, items in pred_scores_dict.items():
        confidence_list.extend( items[2].tolist()  )
    # confidence_list = [ np.mean(items[2])  for vidname, items in pred_scores_dict.items()]
    confidence_list = sorted(confidence_list, reverse=True)
    for ps_thresh_percent_ in np.arange(0.9, 0, -0.1):
        pos_ = min(len(confidence_list) - 1, int(len(confidence_list) * float(ps_thresh_percent_)))
        thresh = confidence_list[pos_]

        n_correct_frames = 0
        n_frames_above_thresh = 0

        # n_correct_vids = 0
        # n_vids_above_thresh = 0
        # n_vids_total = len(pred_scores_dict)

        for vidname, items in pred_scores_dict.items():
            gt_label_seq, pred_label_seq, max_pred_score_seq = items[0], items[1], items[2]
            n_frames_above_thresh += np.sum(  max_pred_score_seq >= thresh  )
            n_correct_frames += np.sum( np.logical_and(max_pred_score_seq >= thresh, gt_label_seq == pred_label_seq )  )



        percent_frames_above_thresh = float(n_frames_above_thresh) / n_frames_total * 100.0
        acc_frame = np.NaN if n_frames_above_thresh == 0 else float(n_correct_frames) / n_frames_above_thresh
        final_results_write.write(
            f'Thresh {thresh:.2f} ({ps_thresh_percent_ * 100.0:.1f}%) : #frames {n_frames_above_thresh}/{percent_frames_above_thresh:.2f}% , acc frame {acc_frame:.3f}\n')

    final_results_write.write('\n')

    final_results_write.close()

def normal_init(module, mean=0, std=1, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.normal_(module.weight, mean, std)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val  # current value
        self.sum += val * n   # total sum
        self.count += n   # total count
        self.avg = self.sum / self.count  # average among all

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

def init_layer( layer, std = 0.001):
    normal_(layer.weight, 0, std)
    constant_(layer.bias, 0)
    return layer

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


def train_model_da(model, dataloaders, criterion, criterion_domain, optimizer,
                   num_epochs=25, is_inception=False, device = None, logger = None,
                   dataload_iter = None, writer = None, beta_high = None, val_frequency = None,
                   model_name = None,
                   adv_da = True,
                   target_train_vid_info_dict = None, mapping_dict = None, target_train_dir = None, n_frames_per_vid = None,
                   transform_target_train = None, batch_size = None, num_workers = None,
                   sample_first = None, source_only_pseudo_dict = None,

                   w_ps = None,
                   weight_cls_s = None, weight_cls_t_ps = None,
                   ps_thresh_percent = None, target_train_vid_level_pseudo_dict = None, ps_main_dir = None, tsn_sampling = None,

                   use_ps = 'supervised', n_classes = None, source_list_file = None, buffer_size = None,
                   cos_sim = None, temp_param = None,

                   return_model = 'last', img_format = '.png', compute_pseudo_labels = None, result_dir = None, log_time = None
                   ):
    since = time.time()

    # total_loss_history = [None] * num_epochs
    # source_train_loss_history = [None] * num_epochs
    # target_train_loss_history = [None] * num_epochs
    # adv_loss_history = [None] * num_epochs
    # val_loss_history = [None] * num_epochs
    #
    # source_train_acc_history = [None] * num_epochs
    # target_train_acc_history = [None] * num_epochs
    # val_acc_history = [None] * num_epochs

    total_loss = AverageMeter()
    source_train_loss = AverageMeter()
    target_train_loss = AverageMeter()
    if w_ps and use_ps == 'supervised':
        target_train_loss_ps = AverageMeter()
    elif w_ps and use_ps == 'contrast':
        contrast_loss = AverageMeter()
    if adv_da:
        adv_loss = AverageMeter()
    val_loss = AverageMeter()

    source_train_acc = AverageMeter()
    target_train_acc = AverageMeter()
    # val_acc = AverageMeter()

    if w_ps and use_ps == 'contrast':
        source_memory_bank = MemoryBank(n_classes= n_classes, source_list_file= source_list_file, buffer_size=buffer_size  )


    best_model_wts = copy.deepcopy(model.state_dict())
    best_val_acc = 0.0
    best_val_vid_ps_acc = 0.0
    epoch_best_val_acc = 0
    epoch_best_val_vid_ps_acc = 0

    source_train_loader = dataloaders['source_train']
    target_train_loader = dataloaders['target_train']
    dataloader_list = [source_train_loader, target_train_loader]
    max_dataload_idx = 0 if len(source_train_loader) >= len(target_train_loader) else 1

    dataload_main_idx = 1-max_dataload_idx if dataload_iter == 'min' else max_dataload_idx
    dataloader_main = dataloader_list[dataload_main_idx]
    dataloader_sec = dataloader_list[1-dataload_main_idx]

    max_iters = num_epochs * len(dataloader_main)

    global_iter = 0
    for epoch in range(num_epochs):
        logger.debug('Epoch {}/{}'.format(epoch, num_epochs - 1))
        logger.debug('-' * 10)
        # todo Each epoch has a training and validation phase
        # todo ################################
        for phase in ['train', 'val']:
            # todo ################################
            if phase == 'val' and (epoch +1) % val_frequency != 0:
                continue
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            # running_val_loss = 0.0
            running_val_corrects = 0
            # running_total_loss = 0.0
            # running_train_cls_s_loss = 0.0
            # running_train_cls_t_loss = 0.0
            # running_adv_loss = 0.0
            # running_s_corrects = 0
            # running_t_corrects = 0


            if phase == 'train':
                if model_name in ['resnet_grl_tsn', 'resnet50_grl_tsn', 'resnet_grl_tsn_weight' ,'resnet_ps_target_tsn', 'resnet_s_and_t_tsn', 'resnet_s_and_t_grl_tsn', 'resnet_contrast_tsn','resnet50_contrast_tsn']:
                    # re-intialize the target train dataset and dataloader
                    logger.debug(f'Re-initialize target train dataset, sample {n_frames_per_vid} frames per video! ')
                    if model_name in ['resnet_grl_tsn','resnet50_grl_tsn']:
                        target_train_frame_list = generate_tsn_frames(vid_info_dict=target_train_vid_info_dict,
                                                                      mapping_dict=mapping_dict,
                                                                      n_segments=n_frames_per_vid,
                                                                      img_dir=target_train_dir, img_format=img_format )
                    elif model_name == 'resnet_grl_tsn_weight':
                        target_train_frame_list = generate_tsn_frames(vid_info_dict=target_train_vid_info_dict,mapping_dict=mapping_dict,n_segments=n_frames_per_vid,img_dir=target_train_dir,
                                                                      prob_sampling=True, sample_first=sample_first,   source_only_pseudo_dict=source_only_pseudo_dict, img_format=img_format)
                    elif model_name in ['resnet_ps_target_tsn', 'resnet_s_and_t_tsn', 'resnet_s_and_t_grl_tsn','resnet_contrast_tsn','resnet50_contrast_tsn' ]:
                        target_train_frame_list = generate_target_frame_list(vid_info_dict=target_train_vid_info_dict,
                                                         mapping_dict=mapping_dict,
                                                         ps_thresh_percent=ps_thresh_percent, img_dir=target_train_dir,
                                                         target_train_vid_level_pseudo_dict=target_train_vid_level_pseudo_dict,
                                                         ps_main_dir=ps_main_dir,
                                                         tsn_sampling=tsn_sampling, n_segments= n_frames_per_vid, img_format=img_format, logger=logger)
                    target_train_datasets = ImageFilelist(imlist=target_train_frame_list,
                                                          transform=transform_target_train, w_ps= w_ps)
                    target_train_loader = torch.utils.data.DataLoader(target_train_datasets, batch_size=batch_size,
                                                                      shuffle=True, num_workers=num_workers)
                    dataloader_list = [source_train_loader, target_train_loader]
                    max_dataload_idx = 0 if len(source_train_loader) >= len(target_train_loader) else 1

                    dataload_main_idx = 1 - max_dataload_idx if dataload_iter == 'min' else max_dataload_idx
                    dataloader_main = dataloader_list[dataload_main_idx]
                    dataloader_sec = dataloader_list[1 - dataload_main_idx]



                logger.debug(f'In the epoch beginning, the secondary dataloader of length {len(dataloader_sec)} is reset for a new iterator!')
                dataloader_sec_iterator = iter(dataloader_sec)

                for i, data_tuple_main in enumerate(dataloader_main):
                    try:
                        data_tuple_sec = next( dataloader_sec_iterator)
                    except StopIteration:
                        logger.debug(f'The secondary dataloader of length {len(dataloader_sec)} is reset for a new iterator!')
                        dataloader_sec_iterator = iter(dataloader_sec)
                        data_tuple_sec = next(dataloader_sec_iterator)
                    if dataload_main_idx == 0:
                        data_tuple_s, data_tuple_t = data_tuple_main, data_tuple_sec
                    elif dataload_main_idx == 1:
                        data_tuple_s, data_tuple_t = data_tuple_sec, data_tuple_main
                    if use_ps == 'contrast':
                        inputs_s, labels_s, source_frame_indices = data_tuple_s
                    else:
                        inputs_s, labels_s = data_tuple_s

                    if w_ps:
                        inputs_t,labels_t, labels_t_ps = data_tuple_t
                    else:
                        inputs_t, labels_t = data_tuple_t

                    inputs_s, inputs_t = inputs_s.to(device), inputs_t.to(device)  # (batch, 3, 224, 224)
                    labels_s, labels_t = labels_s.to(device), labels_t.to(device)
                    if w_ps:
                        labels_t_ps = labels_t_ps.to(device)

                    beta = adapt_weight(global_iter, max_iters, weight_value=10.0, high_value=beta_high, low_value=0.0) if adv_da else 0.0

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(True):

                        # Get model outputs and calculate loss
                        # Special case for inception because in training it has an auxiliary output. In train
                        #   mode we calculate the loss by summing the final output and the auxiliary output
                        #   but in testing we only consider the final output.
                        pred_cls_s, pred_d_s, feat_s = model(inputs_s, beta= beta) # (batch, n_classes),  (batch, 2), (batch, n_ftrs)
                        pred_cls_t, pred_d_t, feat_t = model(inputs_t, beta = beta)

                        if use_ps == 'contrast':
                            # update the memory bank of source samples
                            source_memory_bank.update_buffer(batch_index= source_frame_indices, batch_feat= feat_s)

                        # loss computation
                        loss = torch.tensor(0).float().cuda()
                        # source classification loss
                        loss_cls_s = criterion(pred_cls_s, labels_s)
                        loss += loss_cls_s * weight_cls_s

                        if w_ps and use_ps== 'supervised':
                            # use pseudo labels for supervised training
                            loss_cls_t_ps = criterion(pred_cls_t, labels_t_ps )
                            loss += loss_cls_t_ps * weight_cls_t_ps

                        if w_ps and use_ps == 'contrast':
                            assert labels_t_ps.size(0) == feat_t.size(0)
                            # use pseudo labels for contrastive learning
                            n_samples_contra = 0  #  number of target samples (that have positive and negative samples) in this batch
                            l_contra_frame = torch.tensor(0).float().cuda()
                            pos_samples = []
                            neg_samples = []
                            target_features = []
                            for idx_ in range( labels_t_ps.size(0)):  # a batch of pseudo labels
                                target_ps_label_ = labels_t_ps[idx_]
                                # randomly sample a positive sample and a negative sample, given a pseudo label
                                pos_sample,  neg_sample = source_memory_bank.random_sample( target_ps_label_)
                                if pos_sample is not None:
                                    n_samples_contra += 1
                                    pos_samples.append( pos_sample )
                                    neg_samples.append( neg_sample)
                                    target_features.append( feat_t[idx_] )
                            if n_samples_contra > 0:
                                pos_samples = torch.stack(pos_samples, dim=0) # a list of tensors
                                neg_samples = torch.stack(neg_samples, dim=0)
                                target_features = torch.stack(target_features, dim=0)
                                pos_component = torch.exp(cos_sim(target_features, pos_samples) / temp_param)
                                neg_component = torch.exp(cos_sim(target_features, neg_samples) / temp_param)
                                l_contra_frame = -torch.log(pos_component / (pos_component + neg_component))
                                l_contra_frame = torch.sum(l_contra_frame, dim=0)

                                l_contra_frame /= n_samples_contra
                                loss += l_contra_frame

                        if adv_da:
                            s_domain_label = torch.zeros( pred_d_s.size(0) ).long() # (batch_s, )
                            t_domain_label = torch.zeros( pred_d_t.size(0) ).long()
                            domain_label = torch.cat((s_domain_label, t_domain_label ), 0).cuda(non_blocking=True)
                            pred_d = torch.cat(( pred_d_s, pred_d_t), 0 )
                            loss_adv = criterion_domain(pred_d, domain_label)
                            loss += loss_adv

                        loss_cls_t = criterion(pred_cls_t, labels_t) # todo only for logging purpose

                        # _, preds_s = torch.max(pred_cls_s, 1)
                        # _, preds_t = torch.max(pred_cls_t, 1)

                        loss.backward()
                        optimizer.step()

                    # statistics
                    #  update after one iteration
                    total_loss.update(loss.item(), inputs_s.size(0))
                    source_train_loss.update(loss_cls_s.item(), inputs_s.size(0))
                    target_train_loss.update(loss_cls_t.item(), inputs_t.size(0))
                    if w_ps and use_ps == 'supervised':
                        target_train_loss_ps.update(loss_cls_t_ps.item(), inputs_t.size(0) )
                    if w_ps and use_ps == 'contrast':
                        if n_samples_contra > 0:
                            contrast_loss.update( l_contra_frame.item(), n_samples_contra  )
                    if adv_da:
                        adv_loss.update( loss_adv.item(),  pred_d.size(0) )

                    prec1_source = accuracy(pred_cls_s, labels_s)[0]
                    prec1_target = accuracy(pred_cls_t, labels_t)[0]

                    source_train_acc.update(prec1_source.item(), pred_cls_s.size(0))
                    target_train_acc.update(prec1_target.item(), pred_cls_t.size(0))

                    global_iter += 1

            elif phase == 'val':

                pred_concat = []
                gt_concat = []
                # # todo ###################### to delete #######################
                # for inputs, labels in dataloaders[phase]:
                #     inputs = inputs.to(device)  # (batch, 3, 224, 224)
                #     labels = labels.to(device)
                #     # zero the parameter gradients
                #     optimizer.zero_grad()
                #     # forward
                #     # track history if only in train
                #     with torch.set_grad_enabled(False):
                #         # Get model outputs and calculate loss
                #         # Special case for inception because in training it has an auxiliary output. In train
                #         #   mode we calculate the loss by summing the final output and the auxiliary output
                #         #   but in testing we only consider the final output.
                #         outputs, _, _ = model(inputs, beta= 0)   #  (batch, n_classes )
                #         loss = criterion(outputs, labels)
                #         _, preds = torch.max(outputs, 1)
                #     # statistics
                #     # running_val_loss += loss.item() * inputs.size(0)
                #     running_val_corrects += torch.sum(preds == labels.data)
                #     val_loss.update( loss.item(), inputs.size(0) )
                #
                #     # pred_concat = np.concatenate( [pred_concat, preds.detach().cpu().numpy() ] )
                #     # gt_concat = np.concatenate( [gt_concat, labels.detach().cpu().numpy() ] )
                # # todo ###################### to delete #######################
                pred_dict = dict()
                for paths, inputs, labels in dataloaders[phase]:
                    inputs = inputs.to(device)  # (batch, 3, 224, 224)
                    labels = labels.to(device)
                    # zero the parameter gradients
                    optimizer.zero_grad()
                    with torch.set_grad_enabled(False):
                        # Get model outputs and calculate loss
                        # Special case for inception because in training it has an auxiliary output. In train
                        #   mode we calculate the loss by summing the final output and the auxiliary output
                        #   but in testing we only consider the final output.
                        outputs, _, _ = model(inputs, beta= 0)   #  (batch, n_classes )
                        loss = criterion(outputs, labels)
                        _, preds = torch.max(outputs, 1)  # (batch, )
                        outputs = torch.nn.functional.softmax( outputs , dim=1).detach().cpu().numpy()  # (batch, n_classes )
                    n_classes_ = outputs.shape[1]
                    running_val_corrects += torch.sum(preds == labels.data)
                    val_loss.update(loss.item(), inputs.size(0))

                    pred_concat = np.concatenate( [pred_concat, preds.detach().cpu().numpy()] )
                    gt_concat = np.concatenate( [gt_concat, labels.detach().cpu().numpy()] )

                    labels = labels.detach().cpu().numpy()  # (batch, )
                    for sample_idx, path in enumerate(paths):
                        vidname = path.split('/')[-2]
                        if compute_pseudo_labels == 'per_epoch':
                            # todo here compute_pseudo_labels could be True or False
                            if vidname not in pred_dict:
                                # each video has a tuple of  gt_label (n_frames, ),   pred_label (n_frames, ) ,  max_pred_score (n_frames, ),  all_class_scores  (n_frames, n_class )
                                pred_dict.update({ vidname: [  np.empty( (0, )),np.empty( (0, )), np.empty( (0, )), np.empty( (0, n_classes_ ))  ]   })
                            pred_label = np.argmax(outputs[sample_idx])
                            max_pred_score = outputs[sample_idx][pred_label]
                            pred_dict[vidname][0] = np.concatenate( [pred_dict[vidname][0], np.array([labels[sample_idx]])])
                            pred_dict[vidname][1] = np.concatenate( [pred_dict[vidname][1], np.array([pred_label])])
                            pred_dict[vidname][2] = np.concatenate( [pred_dict[vidname][2], np.array([max_pred_score])])
                            pred_dict[vidname][3] = np.concatenate( [pred_dict[vidname][3], np.expand_dims(outputs[sample_idx], axis=0)], axis=0)
                        else:
                            if vidname not in pred_dict:  # each video has a list  [ gt_label,     all_class_scores in shape (n_frames, n_class ) ]
                                pred_dict.update( {vidname: [0, np.empty((0, n_classes_ )) ]}  )
                            pred_dict[vidname][0] = labels[sample_idx].item()
                            pred_dict[vidname][1] = np.concatenate( [pred_dict[vidname][1],   np.expand_dims( outputs[sample_idx], axis=0)  ], axis=0 )


            # End of one epoch

            if phase == 'train':
                info_to_print = f'{phase} total loss: {total_loss.avg:.4f}, cls_s_loss {source_train_loss.avg:.4f}, cls_t_loss {target_train_loss.avg:.4f}'
                if w_ps and use_ps == 'supervised':
                    info_to_print += f', cls_t_loss_ps {target_train_loss_ps.avg:.4f}'
                if w_ps and use_ps == 'contrast':
                    info_to_print += f', contrast_loss {contrast_loss.avg:.4f}'
                if adv_da:
                    info_to_print += f', adv_loss {adv_loss.avg:.4f}'

                logger.debug(f'source acc {source_train_acc.avg:.4f},  target acc {target_train_acc.avg:.4f}')
                writer.add_scalars( 'loss', {'total': total_loss.avg, 'cls_s_loss': source_train_loss.avg, 'cls_t_loss':target_train_loss.avg,   }, global_step = global_iter )
                if w_ps and use_ps == 'supervised':
                    writer.add_scalars('loss', {'cls_t_loss_ps': target_train_loss_ps.avg}, global_step = global_iter )
                if w_ps and use_ps == 'contrast':
                    writer.add_scalars('loss', {'contrast_loss': contrast_loss.avg}, global_step = global_iter )
                if adv_da:
                    writer.add_scalar( 'adv_loss', adv_loss.avg, global_step=global_iter )
                writer.add_scalars( 'acc', {'source_acc': source_train_acc.avg, 'target_acc': target_train_acc.avg,}, global_step = global_iter )
                writer.add_scalar('beta', beta, global_step=global_iter )
            elif phase == 'val':
                # epoch_val_loss = running_val_loss / len(dataloaders[phase].dataset)
                epoch_val_acc = running_val_corrects.double() / len(dataloaders[phase].dataset) * 100.0
                # todo compute vid-level accuracy, video-level confidence score is the average of frame-level confidence scores
                n_vid_correct = 0
                for vidname, items in pred_dict.items():
                    if compute_pseudo_labels == 'per_epoch':
                        gt_label = items[0][0]
                        vid_confidence = np.mean(items[3], axis=0)  # (n_class, )
                    else: # todo compute_pseudo_labels can be True or False
                        gt_label = items[0]
                        vid_confidence =  np.mean( items[1], axis=0 )   # (n_class, )
                    vid_ps_label = np.argmax( vid_confidence)
                    if gt_label == vid_ps_label:
                        n_vid_correct += 1
                vid_ps_acc = float(n_vid_correct) / len(pred_dict) * 100.0
                logger.debug(f'{phase} Loss: {val_loss.avg:.4f} Acc: {epoch_val_acc:.4f} Best acc: {best_val_acc:.4f} (Epoch {epoch_best_val_acc})')
                logger.debug(f'# correct vids {n_vid_correct} / #vids total {len(pred_dict)} : {vid_ps_acc:.3f} Best vid ps acc {best_val_vid_ps_acc:.4f} (Epoch {epoch_best_val_vid_ps_acc})')
                logger.debug(classification_report(pred_concat, gt_concat))
                writer.add_scalars('loss', {'val_loss': val_loss.avg  }, global_step= global_iter)
                writer.add_scalars( 'acc', { 'val_acc': epoch_val_acc, 'vid_ps_acc' : vid_ps_acc }, global_step= global_iter )
                if compute_pseudo_labels == 'per_epoch':
                    process_pred_dict(pred_dict, logger, n_class=n_classes_, result_dir= result_dir  , log_time= log_time, epoch=epoch+1)

                # writer.add_scalar( 'val_loss', val_loss.avg, global_step= global_iter)
                # writer.add_scalar( 'val_acc', epoch_val_acc, global_step=global_iter)

            # deep copy the model
            if phase == 'val':
                if epoch_val_acc > best_val_acc:  # model with best val ps frame accuracy
                    best_val_acc = epoch_val_acc
                    epoch_best_val_acc = epoch
                    best_model_wts = copy.deepcopy(model.state_dict())
                if vid_ps_acc > best_val_vid_ps_acc: # model with best val ps vid accuracy
                    best_val_vid_ps_acc = vid_ps_acc
                    epoch_best_val_vid_ps_acc = epoch


    # end of training
    time_elapsed = time.time() - since
    logger.debug('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    logger.debug('Best val Acc: {:4f} (Epoch {})'.format(best_val_acc, epoch_best_val_acc))

    if return_model == 'best':
        logger.debug(f'Returning model at best val ps frame acc Epoch {epoch_best_val_acc}')
        # load best model weights
        model.load_state_dict(best_model_wts)
    return model

def train_model_da_v2(model, dataloaders, criterion, criterion_domain, optimizer,
                      num_epochs=25, device = None, logger = None,
                      writer = None, beta_high = None, val_frequency = None,
                      model_name = None,
                      adv_da = True,
                      target_train_vid_info_dict = None, source_train_vid_info_dict = None,
                      mapping_dict = None,
                      target_train_vid_frame_dir = None, source_train_vid_frame_dir = None,
                      n_frames_per_vid = None,
                      data_transforms = None,
                      batch_size = None, num_workers = None,
                      sample_first = None, source_only_pseudo_dict = None,

                      w_ps = None,
                      weight_cls_s = None, weight_cls_t_ps = None,
                      ps_thresh_percent = None, target_train_vid_level_pseudo_dict = None, ps_main_dir = None, tsn_sampling = None,

                      use_ps = 'supervised', n_classes = None, source_img_list_file = None, source_train_sampled_frame_list = None,
                      buffer_size = None,
                      cos_sim = None, temp_param = None,

                      return_model = 'last', img_format = '.png', compute_pseudo_labels = None, result_dir = None, log_time = None
                      ):
    since = time.time()
    total_loss = AverageMeter()
    source_train_loss = AverageMeter()
    target_train_loss = AverageMeter()
    if w_ps and use_ps == 'supervised':
        target_train_loss_ps = AverageMeter()
    elif w_ps and use_ps == 'contrast':
        contrast_loss = AverageMeter()
    if adv_da:
        adv_loss = AverageMeter()
    val_loss = AverageMeter()

    source_train_acc = AverageMeter()
    target_train_acc = AverageMeter()
    # val_acc = AverageMeter()

    if w_ps and use_ps == 'contrast':
        #  todo the source_train_sampled_frame_list  is updated in each epoch, but the 5 frames (from the same video) always have the same 5 indices,  the corresponding labels will be the same
        #                          e.g.  target train video 1, correspond to indices of  0,1,2,3,4  in the source_train_sampled_frame_list,
        #                          even there are different 5 frames sampled from video1 in each epoch,  their indices will always be 0,1,2,3,4,  the corresponding labels will be the same
        source_img_memory_bank = MemoryBank(n_classes= n_classes, source_list_file= source_img_list_file, buffer_size=buffer_size)
        source_frame_memory_bank = MemoryBank(n_classes= n_classes, source_list_file= source_train_sampled_frame_list, buffer_size=buffer_size) # source_train_sampled_frame_list


    best_model_wts = copy.deepcopy(model.state_dict())
    best_val_acc = 0.0
    best_val_vid_ps_acc = 0.0
    epoch_best_val_acc = 0
    epoch_best_val_vid_ps_acc = 0

    source_train_img_loader = dataloaders['source_train_img']
    target_train_loader = dataloaders['target_train']
    # dataloader_list = [source_train_img_loader, target_train_loader]
    # max_dataload_idx = 0 if len(source_train_img_loader) >= len(target_train_loader) else 1

    # dataload_main_idx = 1-max_dataload_idx if dataload_iter == 'min' else max_dataload_idx
    # dataloader_main = dataloader_list[dataload_main_idx]
    # dataloader_sec = dataloader_list[1-dataload_main_idx]
    # max_iters = num_epochs * len(dataloader_main)

    # todo set the target train loader as the main loader
    max_iters = num_epochs * len(target_train_loader)


    half_batch_size = int(batch_size / 2)

    global_iter = 0
    for epoch in range(num_epochs):
        logger.debug('Epoch {}/{}'.format(epoch, num_epochs - 1))
        logger.debug('-' * 10)
        # todo Each epoch has a training and validation phase
        # todo ################################
        for phase in ['train', 'val']:
            # todo ################################
            if phase == 'val' and (epoch +1) % val_frequency != 0:
                continue
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode
            running_val_corrects = 0
            if phase == 'train':
                if model_name in [ 'resnet_grl_tsn', 'resnet_grl_tsn_weight' ,'resnet_ps_target_tsn', 'resnet_s_and_t_tsn', 'resnet_s_and_t_grl_tsn', 'resnet_contrast_tsn',]:
                    # re-intialize the source video frame dataset and dataloader
                    logger.debug(f'Re-initialize source video frame dataset, sample {n_frames_per_vid} frames per video!')
                    source_train_frame_list = generate_tsn_frames(vid_info_dict= source_train_vid_info_dict,
                                                                  mapping_dict=mapping_dict,
                                                                  n_segments=n_frames_per_vid,
                                                                  img_dir= source_train_vid_frame_dir,
                                                                  img_format=img_format)
                    if 'contrast' in model_name:
                        #  todo the source_train_sampled_frame_list  is updated in each epoch, but the 5 frames (from the same video) always have the same 5 indices,
                        #                          e.g.  target train video 1, correspond to indices of  0,1,2,3,4  in the source_train_sampled_frame_list,
                        #                          even there are different 5 frames sampled from video1 in each epoch,  their indices will always be 0,1,2,3,4
                        source_train_vid_frame_datasets = ImageFilelist(imlist=source_train_frame_list, transform=data_transforms['source_train'], w_ps=False, return_index=True)
                    else:
                        source_train_vid_frame_datasets = ImageFilelist(imlist=source_train_frame_list,transform=data_transforms['source_train'], w_ps=False)
                    source_train_vid_frame_loader = torch.utils.data.DataLoader(source_train_vid_frame_datasets, batch_size=half_batch_size, shuffle=True, num_workers=num_workers)

                    # re-intialize the target train dataset and dataloader
                    logger.debug(f'Re-initialize target train dataset, sample {n_frames_per_vid} frames per video! ')
                    if model_name == 'resnet_grl_tsn':
                        target_train_frame_list = generate_tsn_frames(vid_info_dict=target_train_vid_info_dict,
                                                                      mapping_dict=mapping_dict,
                                                                      n_segments=n_frames_per_vid,
                                                                      img_dir=target_train_vid_frame_dir, img_format=img_format)
                    elif model_name == 'resnet_grl_tsn_weight':
                        target_train_frame_list = generate_tsn_frames(vid_info_dict=target_train_vid_info_dict, mapping_dict=mapping_dict, n_segments=n_frames_per_vid, img_dir=target_train_vid_frame_dir,
                                                                      prob_sampling=True, sample_first=sample_first, source_only_pseudo_dict=source_only_pseudo_dict, img_format=img_format)
                    elif model_name in ['resnet_ps_target_tsn', 'resnet_s_and_t_tsn', 'resnet_s_and_t_grl_tsn','resnet_contrast_tsn', ]:
                        target_train_frame_list = generate_target_frame_list(vid_info_dict=target_train_vid_info_dict,
                                                                             mapping_dict=mapping_dict,
                                                                             ps_thresh_percent=ps_thresh_percent, img_dir=target_train_vid_frame_dir,
                                                                             target_train_vid_level_pseudo_dict=target_train_vid_level_pseudo_dict,
                                                                             ps_main_dir=ps_main_dir,
                                                                             tsn_sampling=tsn_sampling, n_segments= n_frames_per_vid, img_format=img_format, logger=logger)
                    target_train_datasets = ImageFilelist(imlist=target_train_frame_list,  transform=data_transforms['target_train'], w_ps= w_ps)
                    target_train_loader = torch.utils.data.DataLoader(target_train_datasets, batch_size=batch_size, shuffle=True, num_workers=num_workers)

                    # todo set the target loader as the main loader

                    # dataloader_list = [source_train_img_loader, target_train_loader]
                    # max_dataload_idx = 0 if len(source_train_img_loader) >= len(target_train_loader) else 1
                    #
                    # dataload_main_idx = 1 - max_dataload_idx if dataload_iter == 'min' else max_dataload_idx
                    # dataloader_main = dataloader_list[dataload_main_idx]
                    # dataloader_sec = dataloader_list[1 - dataload_main_idx]


                logger.debug(f'In the epoch beginning, source train img loader (len {len(source_train_img_loader)}) and  source train vid frame loader (len {len(source_train_vid_frame_loader)})   reset for a new iterator!')
                source_train_img_iterator = iter(source_train_img_loader)
                source_train_vid_frame_iterator = iter(source_train_vid_frame_loader)

                for i, data_tuple_t in enumerate( target_train_loader):  # enumerate set a new iterator for the dataloader
                    try:
                        data_tuple_s_img = next(source_train_img_iterator)
                    except StopIteration:
                        logger.debug(f'Source train img loader (len {len(source_train_img_loader)}) reset for a new iterator!')
                        source_train_img_iterator = iter(source_train_img_loader)
                        data_tuple_s_img = next(source_train_img_iterator)

                    try:
                        data_tuple_s_frame = next(source_train_vid_frame_iterator)
                    except StopIteration:
                        logger.debug(f'Source train vid frame loader (len {len( source_train_vid_frame_loader)}) reset for a new iterator!')
                        source_train_vid_frame_iterator = iter(source_train_vid_frame_loader)
                        data_tuple_s_frame = next(source_train_vid_frame_iterator)

                    if use_ps == 'contrast':
                        # inputs_s, labels_s, source_frame_indices = data_tuple_s
                        inputs_s_img, labels_s_img, indices_s_img = data_tuple_s_img
                        inputs_s_frame, labels_s_frame, indices_s_frame = data_tuple_s_frame
                    else:
                        # inputs_s, labels_s = data_tuple_s
                        inputs_s_img, labels_s_img = data_tuple_s_img
                        inputs_s_frame, labels_s_frame = data_tuple_s_frame

                    if w_ps:
                        inputs_t,labels_t, labels_t_ps = data_tuple_t
                    else:
                        inputs_t, labels_t = data_tuple_t

                    # inputs_s, inputs_t = inputs_s.to(device), inputs_t.to(device)  # (batch, 3, 224, 224)

                    inputs_s = torch.cat((inputs_s_img, inputs_s_frame), 0 )  # todo the first half is input img,  second half is input frame
                    labels_s = torch.cat((labels_s_img, labels_s_frame), 0)

                    inputs_s, inputs_t = inputs_s.to(device), inputs_t.to(device)
                    labels_s, labels_t = labels_s.to(device), labels_t.to(device)
                    if w_ps:
                        labels_t_ps = labels_t_ps.to(device)

                    beta = adapt_weight(global_iter, max_iters, weight_value=10.0, high_value=beta_high, low_value=0.0) if adv_da else 0.0

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(True):

                        # Get model outputs and calculate loss
                        # Special case for inception because in training it has an auxiliary output. In train
                        #   mode we calculate the loss by summing the final output and the auxiliary output
                        #   but in testing we only consider the final output.
                        pred_cls_s, pred_d_s, feat_s = model(inputs_s, beta= beta) # (batch, n_classes),  (batch, 2), (batch, n_ftrs)
                        pred_cls_t, pred_d_t, feat_t = model(inputs_t, beta = beta)

                        if use_ps == 'contrast':
                            # update the memory bank of source samples
                            source_img_memory_bank.update_buffer(batch_index= indices_s_img, batch_feat= feat_s[:indices_s_img.shape[0], :]) # todo the first half is input img,  second half is input frame
                            source_frame_memory_bank.update_buffer(batch_index=indices_s_frame, batch_feat= feat_s[indices_s_img.shape[0]:, :] )

                        # loss computation
                        loss = torch.tensor(0).float().cuda()
                        # source classification loss
                        loss_cls_s = criterion(pred_cls_s, labels_s)
                        loss += loss_cls_s * weight_cls_s

                        if w_ps and use_ps== 'supervised':
                            # use pseudo labels for supervised training
                            loss_cls_t_ps = criterion(pred_cls_t, labels_t_ps )
                            loss += loss_cls_t_ps * weight_cls_t_ps

                        if w_ps and use_ps == 'contrast':
                            assert labels_t_ps.size(0) == feat_t.size(0)
                            # use pseudo labels for contrastive learning
                            n_samples_contra = 0  #  number of target samples (that have positive and negative samples) in this batch
                            l_contra_frame = torch.tensor(0).float().cuda()
                            pos_samples = []
                            neg_samples = []
                            target_features = []
                            for idx_ in range( labels_t_ps.size(0)):  # a batch of pseudo labels
                                target_ps_label_ = labels_t_ps[idx_]

                                # todo randomly choose from source img memory bank and source vid frame memory bank
                                #  then randomly sample a positive sample and a negative sample, given a pseudo label
                                source_memory_bank = [source_img_memory_bank, source_frame_memory_bank][ random.randint(0, 1)]
                                pos_sample,  neg_sample = source_memory_bank.random_sample( target_ps_label_)

                                if pos_sample is not None:
                                    n_samples_contra += 1
                                    pos_samples.append( pos_sample )
                                    neg_samples.append( neg_sample)
                                    target_features.append( feat_t[idx_] )
                            if n_samples_contra > 0:
                                pos_samples = torch.stack(pos_samples, dim=0)
                                neg_samples = torch.stack(neg_samples, dim=0)
                                target_features = torch.stack(target_features, dim=0)
                                pos_component = torch.exp(cos_sim(target_features, pos_samples) / temp_param)
                                neg_component = torch.exp(cos_sim(target_features, neg_samples) / temp_param)
                                l_contra_frame = -torch.log(pos_component / (pos_component + neg_component))
                                l_contra_frame = torch.sum(l_contra_frame, dim=0)

                                l_contra_frame /= n_samples_contra
                                loss += l_contra_frame

                        if adv_da:
                            s_domain_label = torch.zeros( pred_d_s.size(0) ).long() # (batch_s, )
                            t_domain_label = torch.zeros( pred_d_t.size(0) ).long()
                            domain_label = torch.cat((s_domain_label, t_domain_label ), 0).cuda(non_blocking=True)
                            pred_d = torch.cat(( pred_d_s, pred_d_t), 0 )
                            loss_adv = criterion_domain(pred_d, domain_label)
                            loss += loss_adv

                        loss_cls_t = criterion(pred_cls_t, labels_t) # todo only for logging purpose

                        # _, preds_s = torch.max(pred_cls_s, 1)
                        # _, preds_t = torch.max(pred_cls_t, 1)

                        loss.backward()
                        optimizer.step()

                    # statistics
                    #  update after one iteration
                    total_loss.update(loss.item(), inputs_s.size(0))
                    source_train_loss.update(loss_cls_s.item(), inputs_s.size(0))
                    target_train_loss.update(loss_cls_t.item(), inputs_t.size(0))
                    if w_ps and use_ps == 'supervised':
                        target_train_loss_ps.update(loss_cls_t_ps.item(), inputs_t.size(0) )
                    if w_ps and use_ps == 'contrast':
                        if n_samples_contra > 0:
                            contrast_loss.update( l_contra_frame.item(), n_samples_contra  )
                    if adv_da:
                        adv_loss.update( loss_adv.item(),  pred_d.size(0) )

                    prec1_source = accuracy(pred_cls_s, labels_s)[0]
                    prec1_target = accuracy(pred_cls_t, labels_t)[0]

                    source_train_acc.update(prec1_source.item(), pred_cls_s.size(0))
                    target_train_acc.update(prec1_target.item(), pred_cls_t.size(0))

                    global_iter += 1

            elif phase == 'val':
                pred_dict = dict()
                for paths, inputs, labels in dataloaders[phase]:
                    inputs = inputs.to(device)  # (batch, 3, 224, 224)
                    labels = labels.to(device)
                    # zero the parameter gradients
                    optimizer.zero_grad()
                    with torch.set_grad_enabled(False):
                        # Get model outputs and calculate loss
                        # Special case for inception because in training it has an auxiliary output. In train
                        #   mode we calculate the loss by summing the final output and the auxiliary output
                        #   but in testing we only consider the final output.
                        outputs, _, _ = model(inputs, beta= 0)   #  (batch, n_classes )
                        loss = criterion(outputs, labels)
                        _, preds = torch.max(outputs, 1)  # (batch, )
                        outputs = torch.nn.functional.softmax( outputs , dim=1).detach().cpu().numpy()  # (batch, n_classes )
                    n_classes_ = outputs.shape[1]
                    running_val_corrects += torch.sum(preds == labels.data)
                    val_loss.update(loss.item(), inputs.size(0))

                    labels = labels.detach().cpu().numpy()  # (batch, )
                    for sample_idx, path in enumerate(paths):
                        vidname = path.split('/')[-2]
                        if compute_pseudo_labels == 'per_epoch':
                            # todo here compute_pseudo_labels could be True or False
                            if vidname not in pred_dict:
                                # each video has a tuple of  gt_label (n_frames, ),   pred_label (n_frames, ) ,  max_pred_score (n_frames, ),  all_class_scores  (n_frames, n_class )
                                pred_dict.update({ vidname: [  np.empty( (0, )),np.empty( (0, )), np.empty( (0, )), np.empty( (0, n_classes_ ))  ]   })
                            pred_label = np.argmax(outputs[sample_idx])
                            max_pred_score = outputs[sample_idx][pred_label]
                            pred_dict[vidname][0] = np.concatenate( [pred_dict[vidname][0], np.array([labels[sample_idx]])])
                            pred_dict[vidname][1] = np.concatenate( [pred_dict[vidname][1], np.array([pred_label])])
                            pred_dict[vidname][2] = np.concatenate( [pred_dict[vidname][2], np.array([max_pred_score])])
                            pred_dict[vidname][3] = np.concatenate( [pred_dict[vidname][3], np.expand_dims(outputs[sample_idx], axis=0)], axis=0)
                        else:
                            if vidname not in pred_dict:  # each video has a list  [ gt_label,     all_class_scores in shape (n_frames, n_class ) ]
                                pred_dict.update( {vidname: [0, np.empty((0, n_classes_ )) ]}  )
                            pred_dict[vidname][0] = labels[sample_idx].item()
                            pred_dict[vidname][1] = np.concatenate( [pred_dict[vidname][1],   np.expand_dims( outputs[sample_idx], axis=0)  ], axis=0 )


            # End of one epoch

            if phase == 'train':
                info_to_print = f'{phase} total loss: {total_loss.avg:.4f}, cls_s_loss {source_train_loss.avg:.4f}, cls_t_loss {target_train_loss.avg:.4f}'
                if w_ps and use_ps == 'supervised':
                    info_to_print += f', cls_t_loss_ps {target_train_loss_ps.avg:.4f}'
                if w_ps and use_ps == 'contrast':
                    info_to_print += f', contrast_loss {contrast_loss.avg:.4f}'
                if adv_da:
                    info_to_print += f', adv_loss {adv_loss.avg:.4f}'

                logger.debug(f'source acc {source_train_acc.avg:.4f},  target acc {target_train_acc.avg:.4f}')
                writer.add_scalars( 'loss', {'total': total_loss.avg, 'cls_s_loss': source_train_loss.avg, 'cls_t_loss':target_train_loss.avg,   }, global_step = global_iter )
                if w_ps and use_ps == 'supervised':
                    writer.add_scalars('loss', {'cls_t_loss_ps': target_train_loss_ps.avg}, global_step = global_iter )
                if w_ps and use_ps == 'contrast':
                    writer.add_scalars('loss', {'contrast_loss': contrast_loss.avg}, global_step = global_iter )
                if adv_da:
                    writer.add_scalar( 'adv_loss', adv_loss.avg, global_step=global_iter )
                writer.add_scalars( 'acc', {'source_acc': source_train_acc.avg, 'target_acc': target_train_acc.avg,}, global_step = global_iter )
                writer.add_scalar('beta', beta, global_step=global_iter )
            elif phase == 'val':
                # epoch_val_loss = running_val_loss / len(dataloaders[phase].dataset)
                epoch_val_acc = running_val_corrects.double() / len(dataloaders[phase].dataset) * 100.0
                # todo compute vid-level accuracy, video-level confidence score is the average of frame-level confidence scores
                n_vid_correct = 0
                for vidname, items in pred_dict.items():
                    if compute_pseudo_labels == 'per_epoch':
                        gt_label = items[0][0]
                        vid_confidence = np.mean(items[3], axis=0)  # (n_class, )
                    else: # todo compute_pseudo_labels can be True or False
                        gt_label = items[0]
                        vid_confidence =  np.mean( items[1], axis=0 )   # (n_class, )
                    vid_ps_label = np.argmax( vid_confidence)
                    if gt_label == vid_ps_label:
                        n_vid_correct += 1
                vid_ps_acc = float(n_vid_correct) / len(pred_dict) * 100.0
                logger.debug(f'{phase} Loss: {val_loss.avg:.4f} Acc: {epoch_val_acc:.4f} Best acc: {best_val_acc:.4f} (Epoch {epoch_best_val_acc})')
                logger.debug(f'# correct vids {n_vid_correct} / #vids total {len(pred_dict)} : {vid_ps_acc:.3f} Best vid ps acc {best_val_vid_ps_acc:.4f} (Epoch {epoch_best_val_vid_ps_acc})')
                # logger.debug(classification_report(pred_concat, gt_concat))
                writer.add_scalars('loss', {'val_loss': val_loss.avg  }, global_step= global_iter)
                writer.add_scalars( 'acc', { 'val_acc': epoch_val_acc, 'vid_ps_acc' : vid_ps_acc }, global_step= global_iter )
                if compute_pseudo_labels == 'per_epoch':
                    process_pred_dict(pred_dict, logger, n_class=n_classes_, result_dir= result_dir  , log_time= log_time, epoch=epoch+1)

                # writer.add_scalar( 'val_loss', val_loss.avg, global_step= global_iter)
                # writer.add_scalar( 'val_acc', epoch_val_acc, global_step=global_iter)

            # deep copy the model
            if phase == 'val':
                if epoch_val_acc > best_val_acc:  # model with best val ps frame accuracy
                    best_val_acc = epoch_val_acc
                    epoch_best_val_acc = epoch
                    best_model_wts = copy.deepcopy(model.state_dict())
                if vid_ps_acc > best_val_vid_ps_acc: # model with best val ps vid accuracy
                    best_val_vid_ps_acc = vid_ps_acc
                    epoch_best_val_vid_ps_acc = epoch


    # end of training
    time_elapsed = time.time() - since
    logger.debug('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    logger.debug('Best val Acc: {:4f} (Epoch {})'.format(best_val_acc, epoch_best_val_acc))

    if return_model == 'best':
        logger.debug(f'Returning model at best val ps frame acc Epoch {epoch_best_val_acc}')
        # load best model weights
        model.load_state_dict(best_model_wts)
    return model