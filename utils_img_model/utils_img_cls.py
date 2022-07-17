
from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import numpy as np
from torchvision import models, transforms
# from torchvision.models.feature_extraction import get_graph_node_names
import matplotlib.pyplot as plt
import time
import os
import os.path as osp
import copy
import logging
from sklearn.metrics import classification_report
from utils_img_model.model_da import normal_init, Cls_Head, DA_clsfr, process_pred_dict
# import scipy.special.so
import getpass

def get_env_id():
    if getpass.getuser() == 'eicg':
        env_id = 0
    elif getpass.getuser() == 'lin':
        env_id = 1
    elif getpass.getuser() == 'ivanl':
        env_id = 2
    else:
        raise Exception("Unknown username!")
    return env_id





def compute_frame_pseudo_label(model, dataloader, logger, device, result_dir =None, log_time = None  ):
    # called at the end of training
    model.eval()
    pred_scores_dict = dict()
    for paths, inputs, labels in dataloader:
        inputs = inputs.to(device)
        # outputs = model(inputs).detach().cpu().numpy()  # (batch, n_classes )
        outputs, _, _ = model(inputs, beta= 0.0)   #  pred_cls, pred_d, feat
        outputs = torch.nn.functional.softmax( outputs , dim=1).detach().cpu().numpy()  # (batch, n_classes )

        labels = labels.detach().cpu().numpy() # (batch, )
        n_class = outputs.shape[1]
        for sample_idx, path in enumerate( paths):
            vidname = path.split('/')[-2]
            if vidname not in pred_scores_dict:
                # each video has a tuple of  gt_label (n_frames, ),   pred_label (n_frames, ) ,  max_pred_score (n_frames, ),  all_class_scores  (n_frames, n_class )
                pred_scores_dict.update({ vidname : [  np.empty( (0, )),np.empty( (0, )), np.empty( (0, )), np.empty( (0, n_class ))  ]     })
            pred_label = np.argmax( outputs[sample_idx]  )
            max_pred_score = outputs[sample_idx][pred_label]
            pred_scores_dict[vidname][0] = np.concatenate( [pred_scores_dict[vidname][0],  np.array([labels[sample_idx] ] )   ] )
            pred_scores_dict[vidname][1] = np.concatenate( [pred_scores_dict[vidname][1],  np.array([pred_label])   ] )
            pred_scores_dict[vidname][2] = np.concatenate( [pred_scores_dict[vidname][2],  np.array([max_pred_score])  ] )
            pred_scores_dict[vidname][3] = np.concatenate( [pred_scores_dict[vidname][3],  np.expand_dims( outputs[sample_idx], axis= 0 )    ], axis= 0 )

    process_pred_dict(pred_scores_dict, logger, n_class=n_class, result_dir=result_dir, log_time=log_time)

def compute_features( model, dataloader,  device,  feat_dim = None, ):



    model.eval()
    pred_scores_dict = dict()
    feat_dict = dict()

    feat_all = np.empty((0, feat_dim))
    labels_all = np.empty((0, ))
    pred_labels_all = np.empty((0, ))
    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        # outputs = model(inputs).detach().cpu().numpy()  # (batch, n_classes )
        outputs, _, feat = model(inputs, beta=0.0)  # pred_cls, pred_d,    feat (batch, feat_dim)
        outputs = torch.nn.functional.softmax(outputs, dim=1).detach().cpu().numpy()  # (batch, n_classes )
        pred_labels = np.argmax( outputs, axis= 1)
        feat = feat.detach().cpu().numpy() # (batch, feat_dim)
        labels = labels.detach().cpu().numpy()  # (batch, )

        feat_all = np.concatenate( [feat_all, feat], axis= 0 )  # (*, feat_dim )
        labels_all = np.concatenate( [labels_all, labels] )
        pred_labels_all = np.concatenate( [pred_labels_all, pred_labels ] )
    feat_dict.update({'feat': feat_all, 'gt':labels_all, 'pred_label': pred_labels })
    return feat_dict


    #     n_class = outputs.shape[1]
    #     for sample_idx, path in enumerate(paths):
    #         vidname = path.split('/')[-2]
    #         if vidname not in pred_scores_dict:
    #             # each video has a tuple of  gt_label (n_frames, ),   pred_label (n_frames, ) ,  max_pred_score (n_frames, ),  all_class_scores  (n_frames, n_class )
    #             pred_scores_dict.update(  {vidname: [np.empty((0,)), np.empty((0,)), np.empty((0,)), np.empty((0, n_class))]})
    #         pred_label = np.argmax(outputs[sample_idx])
    #         max_pred_score = outputs[sample_idx][pred_label]
    #         pred_scores_dict[vidname][0] = np.concatenate(
    #             [pred_scores_dict[vidname][0], np.array([labels[sample_idx]])])
    #         pred_scores_dict[vidname][1] = np.concatenate([pred_scores_dict[vidname][1], np.array([pred_label])])
    #         pred_scores_dict[vidname][2] = np.concatenate([pred_scores_dict[vidname][2], np.array([max_pred_score])])
    #         pred_scores_dict[vidname][3] = np.concatenate(
    #             [pred_scores_dict[vidname][3], np.expand_dims(outputs[sample_idx], axis=0)], axis=0)
    #
    # process_pred_dict(pred_scores_dict, logger, n_class=n_class, result_dir=result_dir, log_time=log_time)

def train_model(model, dataloaders, criterion, optimizer, num_epochs=25, is_inception=False,device = None, logger = None,  ):
    since = time.time()

    train_acc_history = [None] * num_epochs
    train_loss_history = [None] * num_epochs
    val_acc_history = [None] * num_epochs
    val_loss_history = [None] * num_epochs


    best_model_wts = copy.deepcopy(model.state_dict())
    best_val_acc = 0.0
    epoch_best_val_acc = 0

    for epoch in range(num_epochs):
        logger.debug('Epoch {}/{}'.format(epoch, num_epochs - 1))
        logger.debug('-' * 10)



        # todo Each epoch has a training and validation phase
        # todo ################################
        for phase in ['train', 'val']:
        # for phase in ['train', ]:
            # todo ################################

            if phase == 'val' and epoch % 5 != 0:
                continue

            if phase == 'train':
                model.train()  # Set model to training mode
                # model_analysis(model, logger)
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            if phase == 'val':
                pred_concat = []
                gt_concat = []

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)  # (batch, 3, 224, 224)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    # Special case for inception because in training it has an auxiliary output. In train
                    #   mode we calculate the loss by summing the final output and the auxiliary output
                    #   but in testing we only consider the final output.
                    if is_inception and phase == 'train':
                        # From https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958
                        outputs, aux_outputs = model(inputs)
                        loss1 = criterion(outputs, labels)
                        loss2 = criterion(aux_outputs, labels)
                        loss = loss1 + 0.4*loss2
                    else:
                        outputs = model(inputs)   #  (batch, n_classes )
                        loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                if phase == 'val':
                    pred_concat = np.concatenate( [pred_concat, preds.detach().cpu().numpy() ] )
                    gt_concat = np.concatenate( [gt_concat, labels.detach().cpu().numpy() ] )

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
            if phase == 'train':
                logger.debug('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
            elif phase == 'val':
                logger.debug('{} Loss: {:.4f} Acc: {:.4f} Best acc: {:.4f} (Epoch {})'.format(phase, epoch_loss, epoch_acc, best_val_acc, epoch_best_val_acc))

            if phase == 'train':
                # train_acc_history.append( epoch_acc)
                # train_loss_history.append( epoch_loss)
                train_acc_history[epoch] = epoch_acc
                train_loss_history[epoch] = epoch_loss
            # deep copy the model
            if phase == 'val' and epoch_acc > best_val_acc:
                best_val_acc = epoch_acc
                epoch_best_val_acc = epoch
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                # val_acc_history.append(epoch_acc)
                # val_loss_history.append(epoch_loss)
                val_acc_history[epoch] = epoch_acc
                val_loss_history[epoch] = epoch_loss
                logger.debug( classification_report(pred_concat, gt_concat) )

        print()

    time_elapsed = time.time() - since
    logger.debug('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    logger.debug('Best val Acc: {:4f} (Epoch {})'.format(best_val_acc, epoch_best_val_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, train_acc_history, val_acc_history, train_loss_history, val_loss_history

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


def freeze_resnet( model, freeze_child_nr):

    # todo freeze the first few layers of the backbone
    assert freeze_child_nr >= 3
    # assert freeze_child_nr == 6
    for idx, child in enumerate(model.children()):
        if idx <= freeze_child_nr:
            print(f'Freezing child {idx}')
            for param in child.parameters():
                param.requires_grad = False
        # if idx == 7:
        #     # assert len(child.children()) == 2
        #     for sub_idx, sub_child in enumerate( child.children()):
        #         if sub_idx == 0:
        #             for param in sub_child.parameters():
        #                 param.requires_grad = False

def freeze_resnet_6andhalf(model, freeze_child_nr):
    assert freeze_child_nr == 6.5
    for idx, child in enumerate(model.children()):
        if idx <= 6:
            print(f'Freezing child {idx}')
            for param in child.parameters():
                param.requires_grad = False
        if idx == 7:
            print(f'Freezing first half of child {idx}')
            # assert len(child.children()) == 2
            for sub_idx, sub_child in enumerate( child.children()):
                if sub_idx == 0:
                    for param in sub_child.parameters():
                        param.requires_grad = False


class ResNet_GRL(nn.Module):
    def __init__(self, num_classes, use_pretrained=True, freeze_child_nr=None,
                 init_std=0.01,
                 clsfr_mlp_dim=None, d_clsfr_mlp_dim=None,
                 if_dim_reduct = False, new_dim = None, model_name = None):
        super(ResNet_GRL, self).__init__()
        if 'resnet50' in model_name:
            self.model_ft = models.resnet50(pretrained=use_pretrained)
        else:
            self.model_ft = models.resnet18(pretrained=use_pretrained)
        if freeze_child_nr == 6.5:
            freeze_resnet_6andhalf(self.model_ft, freeze_child_nr)
        elif freeze_child_nr == -1:
            pass
        else:
            freeze_resnet(self.model_ft, freeze_child_nr)

        num_ftrs = self.model_ft.fc.in_features
        self.init_std = init_std

        modules = list(self.model_ft.children())[:-1]  # remove the last FC layer
        self.model_ft = nn.Sequential(*modules)
        self.if_dim_reduct = if_dim_reduct and ( num_ftrs != new_dim )
        if self.if_dim_reduct:
            self.dim_reduct = nn.Linear(num_ftrs, new_dim)
            num_ftrs = new_dim
        self.clsfr = Cls_Head(embed_dim=num_ftrs, mlp_dim=clsfr_mlp_dim, n_class=num_classes)
        self.d_clsfr = DA_clsfr(embed_dim=num_ftrs, mlp_dim=d_clsfr_mlp_dim, )
        normal_init(self.clsfr, std=self.init_std)
        normal_init(self.d_clsfr, std=self.init_std)

    def forward(self, x, beta ):
        feat = self.model_ft(x).squeeze()  # (batch, 3, 224, 224) ->  (batch, num_ftrs ),   image representation from the backbone
        if self.if_dim_reduct:
            feat = self.dim_reduct(feat) # (batch, num_ftrs ) -> (batch, new_dim)
        if len(feat.size()) == 1:
            feat = feat.unsqueeze(0)
        pred_cls = self.clsfr(feat)
        pred_d = self.d_clsfr(feat, beta)
        return pred_cls, pred_d, feat

def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True, freeze_child_nr = None,
                     clsfr_mlp_dim = None,  d_clsfr_mlp_dim = None,
                      if_dim_reduct = False, new_dim = None, ):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None
    input_size = 0

    # if model_name == "resnet":
    #     """ Resnet18
    #     """
    #     model_ft = models.resnet18(pretrained=use_pretrained)
    #
    #     # todo if feature_extract is False,  freeze the entire model
    #     # set_parameter_requires_grad(model_ft, feature_extract)
    #     freeze_resnet(model_ft, freeze_child_nr)
    #     num_ftrs = model_ft.fc.in_features  # num_ftrs = 512
    #
    #     # todo overwrite the classifier with a new linear layer
    #     model_ft.fc = nn.Linear(num_ftrs, num_classes)
    #
    #     # intermediate_ftrs = int(num_ftrs / 2)
    #     # model_ft.fc = nn.Linear(num_ftrs,  intermediate_ftrs )
    #     # model_ft = nn.Sequential( model_ft,
    #     #                           nn.ReLU(),
    #     #                           nn.Dropout(0.5),
    #     #                           nn.Linear( intermediate_ftrs, num_classes ))
    #     input_size = 224
    if model_name in [ "resnet", 'resnet50', 'resnet_ps_target', 'resnet_ps_target_tsn', 'resnet_s_and_t_tsn', 'resnet_s_and_t_grl_tsn', 'resnet_contrast_tsn','resnet50_contrast_tsn',
                       'resnet_grl', "resnet_grl_tsn", 'resnet50_grl_tsn', 'resnet_grl_tsn_weight' ]:
        # todo dimension reduction will be performed if if_dim_reduct is True and  new_dim is not equal to num_ftrs
        model_ft = ResNet_GRL(num_classes, use_pretrained=use_pretrained, freeze_child_nr=freeze_child_nr,
                              init_std=0.01, clsfr_mlp_dim=clsfr_mlp_dim, d_clsfr_mlp_dim=d_clsfr_mlp_dim,
                              if_dim_reduct=if_dim_reduct, new_dim=new_dim, model_name = model_name)
        input_size = 224

    elif model_name == "alexnet":
        """ Alexnet
        """
        model_ft = models.alexnet(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224

    elif model_name == "vgg":
        """ VGG11_bn
        """
        model_ft = models.vgg11_bn(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224

    elif model_name == "squeezenet":
        """ Squeezenet
        """
        model_ft = models.squeezenet1_0(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
        model_ft.num_classes = num_classes
        input_size = 224

    elif model_name == "densenet":
        """ Densenet
        """
        model_ft = models.densenet121(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "inception":
        """ Inception v3
        Be careful, expects (299,299) sized images and has auxiliary output
        """
        model_ft = models.inception_v3(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        # Handle the auxilary net
        num_ftrs = model_ft.AuxLogits.fc.in_features
        model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
        # Handle the primary net
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs,num_classes)
        input_size = 299

    else:
        print("Invalid model name, exiting...")
        exit()

    return model_ft, input_size


def plot_history(train_history, result_dir, log_time = None):
    def plot_item(keyword):
        all_item_names = []
        for key in train_history:
            if keyword in key:
                plt.plot(train_history[key])
                all_item_names.append(key)
        plt.xlabel('epoch')
        plt.ylabel(keyword)
        plt.legend(all_item_names, loc= 'upper right')
        plt.title(f'train {keyword}')
        plt.grid(True)


    fig = plt.figure( figsize=(25,10))
    plt.subplot( 121)
    plot_item(keyword='loss')
    plt.subplot(122)
    plot_item(keyword='acc')
    fig.savefig( os.path.join( result_dir, f'{log_time}.png'))

def make_dir(_dir):
    if not os.path.exists(_dir):
        os.makedirs(_dir)

def path_logger(result_dir, log_time):


    streamHandler = logging.StreamHandler()
    streamHandler.setLevel(logging.DEBUG)
    global logger

    logger = logging.getLogger('basic')
    logger.setLevel(logging.DEBUG)

    path_logging = os.path.join( result_dir, f'{log_time}' )

    fileHandler = logging.FileHandler(path_logging, mode='w')
    fileHandler.setLevel(logging.DEBUG)

    formatter = logging.Formatter('%(asctime)s - %(levelno)s - %(filename)s - %(funcName)s - %(message)s')
    streamHandler.setFormatter(formatter)
    fileHandler.setFormatter(formatter)
    logger.addHandler(streamHandler)
    logger.addHandler(fileHandler)
    return logger

def model_analysis(model, logger):
    # print("Model Structure")
    # print(model)

    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    logger.debug('#################################################')
    logger.debug(f'Number of trainable parameters: {params}')
    logger.debug('#################################################')

def get_data_transforms(input_size = None, target_dataset = None, ):
    # Data augmentation and normalization for training
    # Just normalization for validation



    data_transforms = {
        'source_train': transforms.Compose([
            transforms.RandomResizedCrop(input_size),  # random resize crop
            # A crop of random size (default: of 0.08 to 1.0) of the original size and a random
            #     aspect ratio (default: of 3/4 to 4/3) of the original aspect ratio is made. This crop
            #     is finally resized to given size.

            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),


        'target_train': transforms.Compose([
            transforms.RandomResizedCrop(input_size),  # random resize crop
            # A crop of random size (default: of 0.08 to 1.0) of the original size and a random
            #     aspect ratio (default: of 3/4 to 4/3) of the original aspect ratio is made. This crop
            #     is finally resized to given size.

            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),

        'val': transforms.Compose([
            transforms.Resize(input_size),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    if target_dataset == 'NEC':
        print('Setting center crop for target train dataset')
        data_transforms['target_train'] = transforms.Compose([
            transforms.Resize(input_size),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    return data_transforms