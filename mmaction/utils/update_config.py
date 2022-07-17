

import os.path as osp


def get_file_list( experiment_type, use_target, source_train_file, target_train_file, source_val_file, target_val_file, test_domain_label ):
    if experiment_type == 'source_only':
        assert use_target == 'none'   # todo source_only  train_unlabeled part will not be needed
        assert test_domain_label == 0
        train_labeled_list = source_train_file  # source train is domain 0
        train_unlabeled_list = target_train_file  # target train is domain 1
        val_list = target_val_file    # val is of different domain as source train,       in validation, domain_label = 0   # todo when use_target is none,  domain_label in validation should be 0 ; when use_target is not none, domain_label in validation should be 1
    elif experiment_type == 'source_only_source_val':
        assert use_target == 'none'
        assert test_domain_label == 0
        train_labeled_list = source_train_file
        train_unlabeled_list = target_train_file
        val_list = source_val_file  # val is of same domain as train source,              in validation, domain_label = 0
    elif experiment_type == 'target_only':
        assert use_target == 'none'     # todo target_only  train_unlabeled part will not be needed
        assert test_domain_label == 0
        train_labeled_list = target_train_file
        train_unlabeled_list = target_train_file
        val_list = target_val_file  # val is of same domain as train source,                 in validation, domain_label = 0
    elif experiment_type == 'source_and_target':
        # labeled source and labeled target
        assert use_target == 'Sv'
        assert test_domain_label == 1
        raise Exception('Source and labeled target: not yet implemented! ')
        train_labeled_list = source_train_file
        train_unlabeled_list = target_train_file
        val_list = target_val_file  #  val is of different domain as train source,           in validation, domain_label = 1
    elif experiment_type == 'DA':
        assert use_target == 'uSv'
        assert test_domain_label == 1
        train_labeled_list = source_train_file  # todo DA, if weight_vid_clspred, weight_clip_clspred are 0,  train_labeled will not be needed
        train_unlabeled_list = target_train_file
        val_list = target_val_file  # val is of different domain as train source,           in validation, domain_label = 1
    elif experiment_type == 'compute_pseudo_labels':
        # todo test_domain_label could be 0 or 1, depends on how the model is trained
        #  when running test_da.py, we perform inference with the test pipeline

        train_labeled_list = None
        train_unlabeled_list = None
        val_list = target_train_file  # val is of same domain as train source,              only inference, no training, no validation
    return train_labeled_list, train_unlabeled_list, val_list

def update_config(cfg):
    data_dir = cfg.data_dir
    DA_config = cfg.DA_config
    source_to_target = DA_config.source_to_target
    log_str = cfg.log_str

    if source_to_target == 'uh':
        source_dataset, target_dataset = 'ucf101', 'hmdb51'
        target_train_vid_info = data_dir + '/UCF-HMDB/datalist_new/hmdb_train_vid_info.npy'
    elif source_to_target == 'hu':
        source_dataset, target_dataset = 'hmdb51', 'ucf101'
        target_train_vid_info = data_dir + '/UCF-HMDB/datalist_new/ucf_train_vid_info.npy'
    elif source_to_target == '2h':  #  my own crawled web images to hmdb  12 classes
        source_dataset, target_dataset = 'hmdb51', 'hmdb51'
    elif source_to_target == '2u':  #  my own crawled web images to ucf  12 classes
        source_dataset, target_dataset = 'ucf101', 'ucf101'
    elif source_to_target == '2hmdb_all':  # my own crawed web image to  hmdb 51 classes
        source_dataset, target_dataset = 'hmdb_all', 'hmdb_all'
    elif source_to_target == '2ucf_all':  # my own crawled web image to ucf 101 classes
        source_dataset, target_dataset =  'ucf_all', 'ucf_all'
        target_train_vid_info = data_dir + '/UCF-HMDB/BU2UCF/ucf_all_vid_info.npy'
    elif source_to_target == 'e2h':
        target_train_vid_info = data_dir + '/UCF-HMDB/E2H/hmdb_all_vid_info.npy'
    elif source_to_target == 's2u':
        target_train_vid_info = data_dir + '/UCF-HMDB/S2U/ucf_all_vid_info.npy'
    elif source_to_target == 'bu2ucf':  # bu101 to ucf101  todo       !!!!!!!!!!!!!!!!!!!!!!! here  bu2ucf  and  2ucf_all, these two settings are the same,  the same files are returned
        target_train_vid_info = data_dir + '/UCF-HMDB/BU2UCF/ucf_all_vid_info.npy'
    elif source_to_target == 'bu_hmdb_2u':  # 12 classes    bu + hmdb -> ucf
        # source_dataset, target_dataset = 'hmdb_portion', 'ucf101'
        target_train_vid_info = data_dir + '/UCF-HMDB/datalist_new/ucf_train_vid_info.npy'
    elif source_to_target == 'bu_ucf_2h':  # 12 classes, bu + ucf -> hmdb
        target_train_vid_info = data_dir + '/UCF-HMDB/datalist_new/hmdb_train_vid_info.npy'
    elif source_to_target == 's2n':
        target_train_vid_info = data_dir + '/UCF-HMDB/Stanford_Ki_2_NEC/NEC-Drone-7/vid_info_dict.npy'

    # cfg.DA_config.update(cfg.DA_config, target_train_vid_info =target_train_vid_info)


    experiment_type = DA_config.experiment_type

    cfg.DA_config.target_train_vid_info = target_train_vid_info if experiment_type == 'compute_pseudo_labels' else None
    use_target = DA_config.use_target
    test_domain_label = DA_config.test_domain_label


    exp_DA_name = 'baseline' if use_target == 'none' else 'DA'

    if source_to_target in ['uh', 'hu', '2h', '2u']:  # video to video DA
        if cfg.debug:
            source_train_file = osp.join( data_dir,   f'UCF-HMDB/datalist_new/list_{source_dataset}_train_hmdb_ucf-feature_dummy.txt')
            source_val_file = osp.join( data_dir,   f'UCF-HMDB/datalist_new/list_{source_dataset}_val_hmdb_ucf-feature_dummy.txt')
            target_train_file = osp.join( data_dir,   f'UCF-HMDB/datalist_new/list_{target_dataset}_train_hmdb_ucf-feature_dummy.txt')
            target_val_file = osp.join( data_dir,   f'UCF-HMDB/datalist_new/list_{target_dataset}_val_hmdb_ucf-feature_dummy.txt')
        else:
            source_train_file = osp.join( data_dir,   f'UCF-HMDB/datalist_new/list_{source_dataset}_train_hmdb_ucf-feature.txt')
            source_val_file = osp.join( data_dir,   f'UCF-HMDB/datalist_new/list_{source_dataset}_val_hmdb_ucf-feature.txt')
            target_train_file = osp.join( data_dir,   f'UCF-HMDB/datalist_new/list_{target_dataset}_train_hmdb_ucf-feature.txt')
            target_val_file =osp.join( data_dir,   f'UCF-HMDB/datalist_new/list_{target_dataset}_val_hmdb_ucf-feature.txt')
    elif source_to_target in  ['e2h', 's2u', 'bu2ucf', 's2n']:  #  image to video DA
        split_nr = cfg.split_nr
        if source_to_target == 'e2h':
            target_train_file, target_val_file = osp.join( data_dir, f'UCF-HMDB/E2H/list_hmdb_train_split{split_nr}.txt' ), osp.join( data_dir, f'UCF-HMDB/E2H/list_hmdb_val_split{split_nr}.txt' )
        elif source_to_target == 's2u':
            target_train_file, target_val_file = osp.join(data_dir, f'UCF-HMDB/S2U/list_ucf_train_split{split_nr}.txt'), osp.join(data_dir, f'UCF-HMDB/S2U/list_ucf_val_split{split_nr}.txt')
        elif source_to_target == 'bu2ucf':
            target_train_file, target_val_file = osp.join(data_dir,f'UCF-HMDB/BU2UCF/list_ucf_all_train_split{split_nr}.txt'), osp.join(data_dir,f'UCF-HMDB/BU2UCF/list_ucf_all_val_split{split_nr}.txt')
        elif source_to_target == 's2n':
            if cfg.debug:
                target_train_file, target_val_file = osp.join(data_dir, 'UCF-HMDB/Stanford_Ki_2_NEC/NEC-Drone-7/train_vidlist_dummy.txt'), osp.join(data_dir, 'UCF-HMDB/Stanford_Ki_2_NEC/NEC-Drone-7/test_vidlist_dummy.txt')
            else:
                target_train_file, target_val_file = osp.join(data_dir, 'UCF-HMDB/Stanford_Ki_2_NEC/NEC-Drone-7/train_vidlist.txt'  ), osp.join(data_dir, 'UCF-HMDB/Stanford_Ki_2_NEC/NEC-Drone-7/test_vidlist.txt'  )
        assert experiment_type in ['target_only', 'DA', 'compute_pseudo_labels']
        source_train_file = target_train_file
        source_val_file = target_val_file
    elif source_to_target in ['2hmdb_all', '2ucf_all']:  #  image to video DA
        split_nr = cfg.split_nr
        target_train_file = osp.join( data_dir, f'UCF-HMDB/UCF-HMDB_all/splits/list_{target_dataset}_train_split{split_nr}.txt' )
        target_val_file = osp.join( data_dir, f'UCF-HMDB/UCF-HMDB_all/splits/list_{target_dataset}_val_split{split_nr}.txt' )
        assert experiment_type in [ 'target_only', 'DA', 'compute_pseudo_labels']
        source_train_file =target_train_file # source train
        source_val_file = target_val_file
    elif source_to_target in ['bu_hmdb_2u']:  # 12 classes    bu + hmdb -> ucf,  image + video to video DA
        source_percent = cfg.source_percent
        source_train_file = osp.join(data_dir,  f'UCF-HMDB/BU_HMDB_2_UCF/list_hmdb_train_vid.txt') if source_percent == 1 else osp.join(data_dir,  f'UCF-HMDB/BU_HMDB_2_UCF/list_hmdb_train_vid_percent{source_percent}.txt')
        source_val_file = osp.join(data_dir, f'UCF-HMDB/datalist_new/list_hmdb51_val_hmdb_ucf-feature.txt')
        target_train_file = osp.join(data_dir, f'UCF-HMDB/datalist_new/list_ucf101_train_hmdb_ucf-feature.txt')
        target_val_file = osp.join(data_dir, f'UCF-HMDB/datalist_new/list_ucf101_val_hmdb_ucf-feature.txt')
    elif source_to_target in ['bu_ucf_2h'  ]:
        source_percent = cfg.source_percent
        source_train_file = osp.join(data_dir,
                                     f'UCF-HMDB/BU_HMDB_2_UCF/list_ucf_train_vid.txt') if source_percent == 1 else osp.join(data_dir, f'UCF-HMDB/BU_HMDB_2_UCF/list_ucf_train_vid_percent{source_percent}.txt')
        source_val_file = osp.join(data_dir, f'UCF-HMDB/datalist_new/list_ucf101_val_hmdb_ucf-feature.txt')
        target_train_file = osp.join(data_dir, f'UCF-HMDB/datalist_new/list_hmdb51_train_hmdb_ucf-feature.txt')
        target_val_file = osp.join(data_dir, f'UCF-HMDB/datalist_new/list_hmdb51_val_hmdb_ucf-feature.txt')


    # target_test_file = f'/media/data_8T/UCF-HMDB/datalist_new/list_{target_dataset}_val_hmdb_ucf-feature.txt'
    target_test_file = target_val_file

    train_labeled_list, train_unlabeled_list, val_list \
        = get_file_list( experiment_type, use_target, source_train_file, target_train_file, source_val_file, target_val_file, test_domain_label )

    test_list = val_list

    if source_to_target in ['s2n']:
        data_prefix = osp.join(data_dir, 'UCF-HMDB/Stanford_Ki_2_NEC')
        target_start_index = 1
        if_rawframe = True
        filename_tmpl = '{:05}.jpg'
    else:
        data_prefix = None
        target_start_index = 0
        if_rawframe = False
        filename_tmpl = None

    if cfg.model.type == 'RecognizerI3dDA':

        cfg.data = dict(
            videos_per_gpu=cfg.videos_per_gpu,  # number of videos on each gpu,  batch size of each gpu
            # todo only one batch size is allowed

            workers_per_gpu=cfg.workers_per_gpu,  # number of subprocesses to use for data loading for each gpu
            val_dataloader=dict(videos_per_gpu=cfg.videos_per_gpu_val_dataloader),
            test_dataloader=dict(videos_per_gpu=cfg.videos_per_gpu_test_dataloader),
            source_train=dict(    #   todo  config dataset,  set start_index
                type=cfg.dataset_type,  #  MyVideoDataset
                ann_file=train_labeled_list,
                # modality='Flow',
                # filename_tmpl='{}_{:05d}.jpg',
                pipeline=cfg.train_pipeline,
                data_prefix=data_prefix,
                if_rawframe = if_rawframe,
                filename_tmpl = filename_tmpl),
            target_train=dict(
                type=cfg.dataset_type,
                ann_file=train_unlabeled_list,
                # modality='Flow',
                # filename_tmpl='{}_{:05d}.jpg',
                pipeline=cfg.train_pipeline,
                data_prefix=data_prefix,
                start_index=target_start_index,
                if_rawframe=if_rawframe,
                filename_tmpl=filename_tmpl,
                w_pseudo=DA_config.w_pseudo,
                ps_filter_by=DA_config.ps_filter_by,
                pseudo_gt_dict=DA_config.pseudo_gt_dict,
                ps_thresh=DA_config.ps_thresh,
                data_dir=cfg.data_dir,

            ),
            val=dict(
                type=cfg.dataset_type,
                ann_file=val_list,
                pipeline=cfg.val_pipeline,
                data_prefix=data_prefix,
                start_index=target_start_index,
                if_rawframe = if_rawframe,
                filename_tmpl=filename_tmpl),
            test=dict(
                type=cfg.dataset_type,
                ann_file=test_list,
                pipeline=cfg.test_pipeline,
                data_prefix=data_prefix,
                start_index=target_start_index ,
                if_rawframe = if_rawframe,
                filename_tmpl=filename_tmpl))

        if DA_config.w_pseudo:
            cfg.data.update(cfg.data,
                            target_train_for_inference=dict(
                                type=cfg.dataset_type,
                                ann_file=train_unlabeled_list,
                                # modality='Flow',
                                # filename_tmpl='{}_{:05d}.jpg',
                                pipeline=cfg.val_pipeline,
                                data_prefix=data_prefix,
                                start_index=target_start_index,
                                if_rawframe=if_rawframe
                                # todo  target train for inference, us the validation processing pipeline for prediction
                            )
                            )
        else:
            cfg.data.update(cfg.data, target_train_for_inference=None)

        # update the work_dir
        folder_name = f'{cfg.model.backbone.type}_{source_to_target}_{cfg.DA_config.experiment_type}_{log_str}'
    elif cfg.model.type == 'RecognizerTempAgg':
        cfg.data =  dict(
                            videos_per_gpu= cfg.videos_per_gpu,  # number of videos on each gpu,  batch size of each gpu
                            # todo only one batch size is allowed


                            workers_per_gpu= cfg.workers_per_gpu,  # number of subprocesses to use for data loading for each gpu
                            val_dataloader=dict(videos_per_gpu=cfg.videos_per_gpu_val_dataloader),
                            test_dataloader=dict(videos_per_gpu=cfg.videos_per_gpu_test_dataloader),
                            source_train=dict(
                                type=cfg.dataset_type,
                                ann_file=train_labeled_list,
                                data_prefix= None,
                                # modality='Flow',
                                # filename_tmpl='{}_{:05d}.jpg',
                                pipeline=cfg.train_pipeline),
                            target_train=dict(
                                type=cfg.dataset_type,
                                ann_file=train_unlabeled_list,
                                data_prefix= None,
                                # modality='Flow',
                                # filename_tmpl='{}_{:05d}.jpg',
                                pipeline=cfg.train_pipeline,
                                w_pseudo = DA_config.w_pseudo,
                                ps_filter_by = DA_config.ps_filter_by,
                                pseudo_gt_dict = DA_config.pseudo_gt_dict,
                                ps_thresh = DA_config.ps_thresh,
                                data_dir = cfg.data_dir,

                                ),
                            val=dict(
                                type=cfg.dataset_type,
                                ann_file=val_list,
                                data_prefix=None,
                                pipeline=cfg.val_pipeline),
                            test=dict(
                                type=cfg.dataset_type,
                                ann_file=test_list,
                                data_prefix=None,
                                pipeline=cfg.test_pipeline))

        if DA_config.w_pseudo:
            cfg.data.update( cfg.data,
                             target_train_for_inference=dict(
                                 type=cfg.dataset_type,
                                 ann_file=train_unlabeled_list,
                                 data_prefix=None,
                                 # modality='Flow',
                                 # filename_tmpl='{}_{:05d}.jpg',
                                 pipeline=cfg.val_pipeline, #  todo  target train for inference, us the validation processing pipeline for prediction
                                 )
                             )
        else:
            cfg.data.update(cfg.data, target_train_for_inference = None)

        # update the work_dir
        folder_name = f'{cfg.model.backbone.type}_{cfg.tmp_agg}_{source_to_target}_{cfg.DA_config.experiment_type}_{log_str}'

    cfg.work_dir = osp.join( cfg.work_main_dir, folder_name )


    cfg.DA_config.update({
        'exp_DA_name': exp_DA_name,
        'source_train_list': train_labeled_list,
        'target_train_list': train_unlabeled_list,
        'val_list': val_list,
        'test_list': test_list,
    })


    # cfg.DA_config.exp_DA_name = exp_DA_name
    # cfg.DA_config.source_train_list = train_labeled_list
    # cfg.DA_config.target_train_list = train_unlabeled_list
    # cfg.DA_config.val_list = val_list
    # cfg.DA_config.test_list = test_list



    return cfg