from torchvision import datasets, transforms
from utils_img_model.MyImageFolder import MyImageFolder, load_vid_info, load_mapping, generate_tsn_frames, ImageFilelist, ImageFilelist_w_path, \
    generate_target_frame_list, load_frame_list, uniform_sample_frames
from utils_img_model.utils_img_cls import initialize_model, make_dir, path_logger, model_analysis, compute_frame_pseudo_label, get_env_id, get_data_transforms, compute_features
from utils_img_model.model_da import train_model_da, train_model_da_v2
from tensorboardX import SummaryWriter
import torch
import torch.nn as nn
import torch.optim as optim
import os.path as osp
import time
import numpy as np
def load_model(source_dataset = None, target_dataset = None,
        target_train_vidlist = None, target_train_dir=None, source_train_img_list = None, val_img_list =None,
        mapping_dict = None, main_dir =None,
        model_name = None, n_frames_per_vid = None, num_classes = None,
        batch_size = None, num_workers = None,
        if_dim_reduct = None, reduced_dim =None, clsfr_mlp_dim=None, d_clsfr_mlp_dim =None, freeze_child_nr =None,
        if_compute_features = None,
        model_path = None,
               result_dir = None, img_format = None, ):
    log_time = time.strftime("%Y%m%d_%H%M%S")
    # result_dir = osp.join(main_dir, 'ucf_hmdb_img_train', f'{source_dataset}_train_{target_dataset}_val_{model_name}')
    make_dir(result_dir)
    logger = path_logger(result_dir, log_time)
    # writer = SummaryWriter(log_dir=osp.join(result_dir, f'{log_time}_tb'))

    target_train_vid_info_dict = load_vid_info(datalist=target_train_vidlist, img_dir=target_train_dir,
                                               mapping_dict=mapping_dict, )

    target_train_sampled_frame_list = uniform_sample_frames( vid_info_dict = target_train_vid_info_dict, mapping_dict = mapping_dict, n_segments = n_frames_per_vid,img_dir = target_train_dir , img_format=img_format)
    logger.debug(model_name)
    logger.debug(
        f'source_train_list: {source_train_img_list}, target train list {target_train_vidlist}, val_data list: {val_img_list}')
    # logger.debug(f'target train list: {target_train_vidlist}')


    logger.debug(f'classifier mlp dim {clsfr_mlp_dim}, domain discriminator mlp dim {d_clsfr_mlp_dim}')
    # logger.debug(f'freeze all {feature_extract}')
    logger.debug(f'freeze children til and including {freeze_child_nr}')

    # Initialize the model for this run
    # model_ft, input_size = initialize_model(model_name, num_classes, feature_extract, use_pretrained=True)
    # assert model_name == 'resnet'
    model_ft, input_size = initialize_model(model_name, num_classes, feature_extract=None, use_pretrained=True,
                                            freeze_child_nr=freeze_child_nr,
                                            clsfr_mlp_dim=clsfr_mlp_dim, d_clsfr_mlp_dim=d_clsfr_mlp_dim,
                                            if_dim_reduct=if_dim_reduct, new_dim=reduced_dim)

    # Print the model we just instantiated
    print(model_ft)
    model_analysis(model_ft, logger)  # calculate number of trainable parameters

    # todo load model
    model_ft.load_state_dict( torch.load( model_path)   )
    logger.debug(f'loaded model {model_path}')

    data_transforms = get_data_transforms(input_size=input_size, target_dataset=target_dataset)

    print("Initializing Datasets and Dataloaders...")

    # Create training and validation datasets

    # todo ImageFolder
    #  each class is a subfolder

    val_data_transform = data_transforms['val']
    source_train_list = load_frame_list(source_train_img_list)  # 3014 images,  910 videos * 5 frames
    source_dataset = ImageFilelist(imlist=source_train_list, transform=val_data_transform,  w_ps=False)
    target_dataset = ImageFilelist(imlist=target_train_sampled_frame_list, transform= val_data_transform, w_ps=False)


    dataloader_source = torch.utils.data.DataLoader(source_dataset, batch_size=batch_size, shuffle=False,  num_workers=num_workers, drop_last=False)
    dataloader_target = torch.utils.data.DataLoader(target_dataset, batch_size=batch_size, shuffle=False,  num_workers=num_workers, drop_last=False)


    # Detect if we have a GPU available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Send the model to GPU
    model_ft = model_ft.to(device)

    if if_compute_features:
        feat_dict_s = compute_features(  model_ft, dataloader_source,  device,  feat_dim = reduced_dim,  )
        np.save(osp.join(result_dir, f'{log_time}_source_feat.npy'), feat_dict_s)

        feat_dict_t = compute_features(model_ft, dataloader_target, device, feat_dim=reduced_dim, )
        np.save(osp.join(result_dir, f'{log_time}_target_feat.npy'), feat_dict_t)



def run(source_dataset = None, target_dataset = None,
        target_train_vidlist = None, target_train_dir=None, source_train_img_list = None, val_img_list =None, data_prefix = None,
        target_train_img_list = None, target_train_vid_level_pseudo_dict = None,
        mapping_dict = None, main_dir =None,
        model_name = None, n_frames_per_vid = None, num_classes = None,
        batch_size = None, num_epochs=None, lr= None, dataload_iter =None, beta_high = None, val_frequency = None, num_workers = None,
        if_dim_reduct = None, reduced_dim =None, clsfr_mlp_dim=None, d_clsfr_mlp_dim =None, freeze_child_nr =None,
        return_model = None,
        compute_pseudo_labels = None, if_compute_features = None, if_save_model = False,

        ps_main_dir = None, ps_thresh_percent = None,
        img_format ='.png'):
    log_time = time.strftime("%Y%m%d_%H%M%S")
    result_dir = osp.join(main_dir, 'ucf_hmdb_img_train', f'{source_dataset}_train_{target_dataset}_val_{model_name}')
    make_dir(result_dir)
    logger = path_logger(result_dir, log_time)
    writer = SummaryWriter(log_dir=osp.join(result_dir, f'{log_time}_tb'))

    target_train_vid_info_dict = load_vid_info(datalist=target_train_vidlist, img_dir=target_train_dir,
                                               mapping_dict=mapping_dict, )
    if model_name in ['resnet', 'resnet50', 'resnet_grl']:
        w_ps = False
        weight_cls_s = 1
    if model_name in ['resnet_grl_tsn', 'resnet50_grl_tsn', 'resnet_grl_tsn_weight']:
        if model_name in ['resnet_grl_tsn', 'resnet50_grl_tsn']:
            target_train_sampled_frame_list = generate_tsn_frames(vid_info_dict=target_train_vid_info_dict,
                                                                  mapping_dict=mapping_dict,
                                                                  n_segments=n_frames_per_vid,
                                                                  img_dir=target_train_dir,
                                                                  img_format=img_format)
        elif model_name == 'resnet_grl_tsn_weight':
            source_only_pseudo_dict = osp.join(main_dir,
                                               'ucf_hmdb_img_train/ucf_train_hmdb_val/20211215_232454_frame_ps.npy')
            sample_first = 'low'  # frames with low confidence score have high probablity of sampling
            target_train_sampled_frame_list = generate_tsn_frames(vid_info_dict=target_train_vid_info_dict,
                                                                  mapping_dict=mapping_dict,
                                                                  n_segments=n_frames_per_vid, img_dir=target_train_dir,
                                                                  prob_sampling=True, sample_first=sample_first,
                                                                  source_only_pseudo_dict=source_only_pseudo_dict,
                                                                  img_format=img_format)
        w_ps = False
        weight_cls_s = 1
    elif model_name in ['resnet_ps_target', 'resnet_ps_target_tsn', 'resnet_s_and_t_tsn', 'resnet_s_and_t_grl_tsn', 'resnet_contrast_tsn', 'resnet50_contrast_tsn' ]:
        # target_train_vid_level_pseudo_dict = ['/home/eicg/action_recognition_codes/data_lin_on_krios/UCF-HMDB/work_dirs/ResNet3d_tmp_transformer_uh_DA_clip4x16_w_ps_filterbyframe_img0.3_targetonly__/ps_labels/best_top1_acc_epoch_50_vid_ps.npy',
        #                                       '/data/lin/UCF-HMDB/work_dirs/ResNet3d_tmp_transformer_uh_DA_clip4x16_w_ps_filterbyframe_img0.3_targetonly__/ps_labels/best_top1_acc_epoch_50_vid_ps.npy'  ][env_id]
        # todo ################################
        # target_train_vid_level_pseudo_dict = [
        #     '/media/data_8T/UCF-HMDB/ablation_study/ucf_1frame_to_hmdb/step1_grl_tsn_img_model_64d/step2_video_model/ps_epoch_50_vid_ps.npy',
        #     None, ][env_id]
        # todo ################################

        # ps_thresh_percent = 1.0  #  using all the pseudo labeled target training videos
        assert ps_thresh_percent is not None
        logger.debug(f'ps thresh {ps_thresh_percent}')
        assert ps_main_dir is not None
        w_ps = True
        if model_name == 'resnet_ps_target':
            #  todo only cls loss of pseudo labeled target
            tsn_sampling = False
            weight_cls_s = 0
            weight_cls_t_ps = 1
        elif model_name == 'resnet_ps_target_tsn':
            tsn_sampling = True
            weight_cls_s = 0
            weight_cls_t_ps = 1
        elif model_name in ['resnet_s_and_t_tsn', 'resnet_s_and_t_grl_tsn']:
            tsn_sampling = True
            weight_cls_s = 1  # todo labeled source + pseudo labeled target
            weight_cls_t_ps = 1
        elif model_name in ['resnet_contrast_tsn', 'resnet50_contrast_tsn']:
            use_ps = 'contrast'
            tsn_sampling = True
            weight_cls_s = 1
            weight_cls_t_ps = 0
            buffer_size = 'inf'
            cos_sim = nn.CosineSimilarity(dim=1)
            temp_param = 0.05
        # todo if a video has confidence above thresh, add all frames in this video (either TSN style or all frames )
        target_train_sampled_frame_list = generate_target_frame_list(vid_info_dict=target_train_vid_info_dict,
                                                                     mapping_dict=mapping_dict,
                                                                     ps_thresh_percent=ps_thresh_percent,
                                                                     img_dir=target_train_dir,
                                                                     target_train_vid_level_pseudo_dict=target_train_vid_level_pseudo_dict,
                                                                     ps_main_dir=ps_main_dir,
                                                                     tsn_sampling=tsn_sampling,
                                                                     n_segments=n_frames_per_vid,img_format=img_format,
                                                                     logger = logger)

    logger.debug(model_name)
    logger.debug(f'source_train_list: {source_train_img_list}, target train list {target_train_vidlist}, val_data list: {val_img_list}')
    # logger.debug(f'target train list: {target_train_vidlist}')
    logger.debug(f'batch size {batch_size}, num_epochs {num_epochs}, lr {lr}, dataload_iter {dataload_iter}, beta_high {beta_high}, val_frequency {val_frequency}')
    if model_name in ['resnet_grl_tsn', 'resnet50_grl_tsn', 'resnet_grl_tsn_weight' ]:
        logger.debug(f'#frames per target vid {n_frames_per_vid}')
        if model_name == 'resnet_grl_tsn_weight':
            logger.debug(f'source only pseudo {source_only_pseudo_dict}')
            logger.debug(f'Sample frames with {sample_first} confidence')
    elif model_name in ['resnet_ps_target', 'resnet_ps_target_tsn', 'resnet_s_and_t_tsn', 'resnet_s_and_t_grl_tsn',  'resnet_contrast_tsn', 'resnet50_contrast_tsn']:
        logger.debug( f'ps thresh percent {ps_thresh_percent}' )
        logger.debug(f'weight_cls_s {weight_cls_s}, weight_cls_t_ps {weight_cls_t_ps}')
        if 'tsn' in model_name:
            logger.debug(f'TSN style sampling  #frames per target vid {n_frames_per_vid}')
        if 'contrast' in model_name:
            logger.debug(f'buffer size {buffer_size}, temp param {temp_param}' )
            logger.debug(f'reduced dim {reduced_dim}')

    logger.debug(f'classifier mlp dim {clsfr_mlp_dim}, domain discriminator mlp dim {d_clsfr_mlp_dim}')
    # logger.debug(f'freeze all {feature_extract}')
    logger.debug(f'freeze children til and including {freeze_child_nr}')

    # Initialize the model for this run
    # model_ft, input_size = initialize_model(model_name, num_classes, feature_extract, use_pretrained=True)
    # assert model_name == 'resnet'
    model_ft, input_size = initialize_model(model_name, num_classes, feature_extract=None, use_pretrained=True, freeze_child_nr=freeze_child_nr,
                                                clsfr_mlp_dim=clsfr_mlp_dim, d_clsfr_mlp_dim=d_clsfr_mlp_dim, if_dim_reduct=if_dim_reduct, new_dim=reduced_dim)

    # Print the model we just instantiated
    print(model_ft)
    model_analysis(model_ft, logger) # calculate number of trainable parameters


    data_transforms = get_data_transforms(input_size=input_size, target_dataset=target_dataset)


    print("Initializing Datasets and Dataloaders...")

    # Create training and validation datasets

    # todo ImageFolder
    #  each class is a subfolder

    source_train_list = load_frame_list(source_train_img_list)
    if 'contrast' in model_name:
        source_train_datasets = ImageFilelist(imlist=source_train_list, transform=data_transforms['source_train'], w_ps= False, return_index=True  )
    else:
        source_train_datasets = ImageFilelist(imlist= source_train_list, transform=data_transforms['source_train'], w_ps= False )
    if model_name in [ "resnet", 'resnet50', 'resnet_grl']:
        try:
            target_train_list = load_frame_list(target_train_img_list)
            target_train_datasets = ImageFilelist(imlist= target_train_list, transform= data_transforms['target_train'] )
        except NameError:
            # load the target train data  using ImageFolder,  from data directory
            target_train_datasets = datasets.ImageFolder(target_train_dir, data_transforms['target_train'])
    elif model_name in ['resnet_grl_tsn', 'resnet50_grl_tsn', 'resnet_ps_target', 'resnet_ps_target_tsn', 'resnet_s_and_t_tsn','resnet_s_and_t_grl_tsn', 'resnet_contrast_tsn', 'resnet50_contrast_tsn']:
        target_train_datasets = ImageFilelist(imlist=target_train_sampled_frame_list, transform=data_transforms['target_train'], w_ps= w_ps)
    val_list = load_frame_list(val_img_list, data_prefix=data_prefix)
    # val_datasets = ImageFilelist(imlist= val_list, transform= data_transforms['val'], w_ps= False)
    val_datasets = ImageFilelist_w_path(imlist= val_list, transform= data_transforms['val'], w_ps= False)


    image_datasets = {  'source_train' : source_train_datasets,
                        'target_train' : target_train_datasets,
                        'val': val_datasets}

    # Create training and validation dataloaders
    # dataloaders_dict = {x: torch.utils_img_model.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=4) for x in ['train', 'val']}
    # todo use the same batch size for source train and target train ???
    dataloaders_dict = {
        'source_train': torch.utils.data.DataLoader(image_datasets['source_train'], batch_size=batch_size, shuffle=True, num_workers=num_workers),
        'target_train': torch.utils.data.DataLoader(image_datasets['target_train'], batch_size=batch_size, shuffle=True, num_workers=num_workers),
        'val': torch.utils.data.DataLoader(image_datasets['val'], batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=False ),
    }



    # Detect if we have a GPU available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    # Send the model to GPU
    model_ft = model_ft.to(device)

    # Gather the parameters to be optimized/updated in this run. If we are
    #  finetuning we will be updating all parameters. However, if we are
    #  doing feature extract method, we will only update the parameters
    #  that we have just initialized, i.e. the parameters with requires_grad
    #  is True.
    params_to_update = model_ft.parameters()
    print("Params to learn:")

    # if feature_extract:
    # todo only update the parameters whose requires_grad is True
    #  pass only the paramters whose requires_grad is True  to the optimizer
    params_to_update = []
    for name,param in model_ft.named_parameters():
        if param.requires_grad == True:
            params_to_update.append(param)
            print("\t",name)
    # else:
    #     # todo  if feature_extract is False, update the entire model
    #     #  pass all the parameters of the model to optimizer
    #     for name,param in model_ft.named_parameters():
    #         if param.requires_grad == True:
    #             print("\t",name)

    # Observe that all parameters are being optimized
    # optimizer_ft = optim.SGD(params_to_update, lr=0.001, momentum=0.9)
    optimizer_ft = optim.SGD(params_to_update, lr=lr, momentum=0.9)


    # Setup the loss fxn
    criterion = nn.CrossEntropyLoss() #  automatic averaging
    criterion_domain = nn.CrossEntropyLoss()




    # Train and evaluate
    if model_name in ['resnet', 'resnet50']: # todo source only
        adv_da = False
        model_ft = train_model_da(model_ft, dataloaders_dict, criterion, criterion_domain,   optimizer_ft,
                                     num_epochs=num_epochs, is_inception=(model_name=="inception"), device= device, logger= logger,
                                    dataload_iter=dataload_iter, writer=writer, beta_high=beta_high, val_frequency= val_frequency,
                                   adv_da= adv_da,
                                  w_ps= w_ps, weight_cls_s= weight_cls_s, img_format=img_format, compute_pseudo_labels=compute_pseudo_labels, result_dir= result_dir, log_time=log_time)
    elif model_name == 'resnet_grl':
        model_ft  = train_model_da(model_ft, dataloaders_dict, criterion, criterion_domain,   optimizer_ft,
                                     num_epochs=num_epochs, is_inception=(model_name=="inception"), device= device, logger= logger,
                                    dataload_iter=dataload_iter, writer=writer, beta_high=beta_high, val_frequency= val_frequency,

                                   w_ps= w_ps, weight_cls_s= weight_cls_s, img_format=img_format, compute_pseudo_labels=compute_pseudo_labels, result_dir= result_dir, log_time=log_time)
    elif model_name in ['resnet_grl_tsn', 'resnet50_grl_tsn']:
        model_ft = train_model_da(model_ft, dataloaders_dict, criterion, criterion_domain, optimizer_ft,
                                  num_epochs=num_epochs, is_inception=(model_name == "inception"), device=device,
                                  logger=logger,
                                  dataload_iter=dataload_iter, writer=writer, beta_high=beta_high, val_frequency=val_frequency,
                                  w_ps= w_ps, weight_cls_s= weight_cls_s,
                                  model_name=model_name,  target_train_vid_info_dict=target_train_vid_info_dict, mapping_dict=mapping_dict,
                                  target_train_dir=target_train_dir, n_frames_per_vid=n_frames_per_vid, transform_target_train=data_transforms['target_train'],
                                  batch_size=batch_size, num_workers=num_workers, img_format=img_format, compute_pseudo_labels=compute_pseudo_labels, result_dir= result_dir, log_time=log_time)
    elif model_name == 'resnet_grl_tsn_weight':
        model_ft = train_model_da(model_ft, dataloaders_dict, criterion, criterion_domain, optimizer_ft,
                                  num_epochs=num_epochs, is_inception=(model_name == "inception"), device=device,
                                  logger=logger,
                                  dataload_iter=dataload_iter, writer=writer, beta_high=beta_high, val_frequency=val_frequency,
                                  w_ps=w_ps, weight_cls_s=weight_cls_s,
                                  model_name=model_name,  target_train_vid_info_dict=target_train_vid_info_dict, mapping_dict=mapping_dict,
                                  target_train_dir=target_train_dir, n_frames_per_vid=n_frames_per_vid, transform_target_train=data_transforms['target_train'],
                                  batch_size=batch_size, num_workers=num_workers,
                                  sample_first=sample_first, source_only_pseudo_dict= source_only_pseudo_dict, img_format=img_format, compute_pseudo_labels=compute_pseudo_labels, result_dir= result_dir, log_time=log_time)
    elif model_name == 'resnet_ps_target':
        assert dataload_iter == 'max'  #  use all the target frames in training,  pseudo labels for supervised training
        adv_da = False
        model_ft = train_model_da(model_ft, dataloaders_dict, criterion, criterion_domain,   optimizer_ft,
                                     num_epochs=num_epochs, is_inception=(model_name=="inception"), device= device, logger= logger,
                                    dataload_iter=dataload_iter, writer=writer, beta_high=beta_high, val_frequency= val_frequency,
                                   model_name=model_name, adv_da= adv_da,

                                  w_ps=w_ps, weight_cls_s= weight_cls_s, weight_cls_t_ps=weight_cls_t_ps, img_format=img_format, compute_pseudo_labels=compute_pseudo_labels, result_dir= result_dir, log_time=log_time )
    elif model_name in ['resnet_ps_target_tsn', 'resnet_s_and_t_tsn', 'resnet_s_and_t_grl_tsn']:
        adv_da = True if model_name == 'resnet_s_and_t_grl_tsn' else False
        model_ft = train_model_da(model_ft, dataloaders_dict, criterion, criterion_domain, optimizer_ft,
                                  num_epochs=num_epochs, is_inception=(model_name == "inception"), device=device,
                                  logger=logger,
                                  dataload_iter=dataload_iter, writer=writer, beta_high=beta_high,
                                  val_frequency=val_frequency,
                                  model_name=model_name, adv_da=adv_da, target_train_vid_info_dict=target_train_vid_info_dict, mapping_dict=mapping_dict,
                                 target_train_dir=target_train_dir, n_frames_per_vid=n_frames_per_vid,  transform_target_train=data_transforms['target_train'],
                                  batch_size=batch_size, num_workers=num_workers,

                                  w_ps=w_ps, weight_cls_s=weight_cls_s, weight_cls_t_ps=weight_cls_t_ps,
                                  ps_thresh_percent=ps_thresh_percent, target_train_vid_level_pseudo_dict=target_train_vid_level_pseudo_dict, ps_main_dir=ps_main_dir, tsn_sampling=tsn_sampling, img_format=img_format, compute_pseudo_labels=compute_pseudo_labels, result_dir= result_dir, log_time=log_time)

    elif model_name in ['resnet_contrast_tsn', 'resnet50_contrast_tsn']:
        adv_da = False
        model_ft = train_model_da(model_ft, dataloaders_dict, criterion, criterion_domain, optimizer_ft,
                                  num_epochs=num_epochs, is_inception=(model_name == "inception"), device=device,
                                  logger=logger,
                                  dataload_iter=dataload_iter, writer=writer, beta_high=beta_high,
                                  val_frequency=val_frequency,
                                  model_name=model_name, adv_da=adv_da, target_train_vid_info_dict=target_train_vid_info_dict, mapping_dict=mapping_dict,
                                  target_train_dir=target_train_dir, n_frames_per_vid=n_frames_per_vid, transform_target_train=data_transforms['target_train'],
                                  batch_size=batch_size, num_workers=num_workers,

                                  w_ps=w_ps, weight_cls_s=weight_cls_s, weight_cls_t_ps=weight_cls_t_ps,
                                  ps_thresh_percent=ps_thresh_percent, target_train_vid_level_pseudo_dict=target_train_vid_level_pseudo_dict, ps_main_dir=ps_main_dir, tsn_sampling=tsn_sampling,

                                  use_ps=use_ps, n_classes=num_classes, source_list_file=source_train_img_list, buffer_size=buffer_size, cos_sim=cos_sim, temp_param=temp_param,
                                  return_model=return_model, img_format=img_format, compute_pseudo_labels=compute_pseudo_labels, result_dir= result_dir, log_time=log_time)



    if compute_pseudo_labels:
        if compute_pseudo_labels != 'per_epoch':
            val_datasets_for_ps = ImageFilelist_w_path(imlist=val_list, transform=data_transforms['val'], w_ps=False)
            dataloader_val_for_ps = torch.utils.data.DataLoader(val_datasets_for_ps, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=False)
            compute_frame_pseudo_label(model_ft, dataloader_val_for_ps, logger, device, result_dir= result_dir, log_time=log_time)

    if if_save_model:
        torch.save(model_ft.state_dict(),  osp.join(result_dir, f'{log_time}.model'))




