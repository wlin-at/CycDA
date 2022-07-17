model = dict(
    type='RecognizerI3dDA',
    backbone=dict(
        type='InceptionI3d',
        pretrained2d=True,
        pretrained='torchvision://resnet50',
        depth=50,
        conv1_kernel=(5, 7, 7),
        conv1_stride_t=2,
        pool1_stride_t=2,
        conv_cfg=dict(type='Conv3d'),
        norm_eval=False,
        inflate=((1, 1, 1), (1, 0, 1, 0), (1, 0, 1, 0, 1, 0), (0, 1, 0)),
        zero_init_residual=False,
        pretrained_model_path=
        '/media/data_8T/mmaction2_checkpoints/InceptionI3d/rgb_imagenet.pt',
        freeze_child_nr=12),
    cls_head=None,
    train_cfg=None,
    test_cfg=dict(average_clips='prob'),
    neck=None,
    spatial_type='avg',
    spatial_dropout_ratio=0.5,
    feat_dim=1024,
    n_class=13,
    init_std=0.01,
    embed_dim_t=64,
    d_clsfr_mlp_dim=64,
    clsfr_mlp_dim=64)
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=40, norm_type=2))
lr_config = dict(policy='step', step=[40, 80])
total_epochs = 10
checkpoint_config = dict(interval=-1, save_last=True)
log_config = dict(
    interval=20,
    hooks=[dict(type='TextLoggerHook'),
           dict(type='TensorboardLoggerHook')])
dist_params = dict(backend='nccl')
log_level = 'DEBUG'
load_from = None
resume_from = None
workflow = [('train', 1)]
env_id = 0
val_frequency = 2
data_dir = '/media/data_8T'
debug = True
source_to_target = 'e2h'
split_nr = 1
n_class = 13
optimizer_cfg = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=1e-07)
paramwise_cfg = dict(
    custom_keys=dict(backbone=dict(lr_mult=0.1, decay_mult=1.0)))
num_clips = 1
num_clips_val = 3
num_clips_test = 3
clip_len = 64
frame_interval = 1
save_last = True
save_best = False
videos_per_gpu = 1
workers_per_gpu = 0
videos_per_gpu_val_dataloader = 2
videos_per_gpu_test_dataloader = 1
custom_hooks = [dict(type='PseudoTargetHook', priority='NORMAL')]
beta_high = 0.1
ps_thresh = 0.7
ps_filter_by = 'frame_v2'
pseudo_gt_dict = '/media/data_8T/UCF-HMDB/ablation_incepi3d/img2vid/e2h/split1/step1_grl_tsn_img_model_64d/v4/step2_vid_model/0.7/epoch_10_test3clip_vid_ps.npy'
log_str = 'clip1x64_w_ps0.7_strid1_testclip3'
pretrained_model_path = '/media/data_8T/mmaction2_checkpoints/InceptionI3d/rgb_imagenet.pt'
embed_dim = 64
freeze_child_nr = 12
DA_config = dict(
    batch_size=1,
    model_type='RecognizerI3dDA',
    source_to_target='e2h',
    experiment_type='DA',
    dataload_iter='min',
    use_target='uSv',
    test_domain_label=1,
    weight_clip_clspred=0,
    weight_clip_clspred_vid_ps=1,
    n_clips=1,
    adv_DA=False,
    w_pseudo='vid',
    ps_filter_by='frame_v2',
    pseudo_gt_dict=
    '/media/data_8T/UCF-HMDB/ablation_incepi3d/img2vid/e2h/split1/step1_grl_tsn_img_model_64d/v4/step2_vid_model/0.7/epoch_10_test3clip_vid_ps.npy',
    ps_thresh=0.7,
    update_ps_start_epoch=-1,
    update_ps_freq=-1,
    target_train_vid_info=None,
    exp_DA_name='DA',
    source_train_list='/media/data_8T/UCF-HMDB/E2H/list_hmdb_train_split1.txt',
    target_train_list='/media/data_8T/UCF-HMDB/E2H/list_hmdb_train_split1.txt',
    val_list='/media/data_8T/UCF-HMDB/E2H/list_hmdb_val_split1.txt',
    test_list='/media/data_8T/UCF-HMDB/E2H/list_hmdb_val_split1.txt')
dataset_type = 'MyVideoDataset'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_bgr=False)
train_pipeline = [
    dict(type='DecordInit'),
    dict(type='SampleFrames', clip_len=64, frame_interval=1, num_clips=1),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(
        type='MultiScaleCrop',
        input_size=224,
        scales=(1, 0.8),
        random_crop=True,
        max_wh_scale_gap=1),
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
    dict(type='Flip', flip_ratio=0.5),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_bgr=False),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'label'])
]
val_pipeline = [
    dict(type='DecordInit'),
    dict(
        type='SampleFrames',
        clip_len=64,
        frame_interval=1,
        num_clips=3,
        test_mode=True),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='CenterCrop', crop_size=224),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_bgr=False),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]
test_pipeline = [
    dict(type='DecordInit'),
    dict(
        type='SampleFrames',
        clip_len=64,
        frame_interval=1,
        num_clips=3,
        test_mode=True),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='CenterCrop', crop_size=224),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_bgr=False),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]
evaluation = dict(
    interval=2,
    save_best=None,
    metrics=['top_k_accuracy', 'mean_class_accuracy'])
work_main_dir = './work_dirs/'
find_unused_parameters = True
work_dir = './work_dirs/InceptionI3d_e2h_DA_clip1x64_w_ps0.7_strid1_testclip3'
data = dict(
    videos_per_gpu=1,
    workers_per_gpu=0,
    val_dataloader=dict(videos_per_gpu=2),
    test_dataloader=dict(videos_per_gpu=1),
    source_train=dict(
        type='MyVideoDataset',
        ann_file='/media/data_8T/UCF-HMDB/E2H/list_hmdb_train_split1.txt',
        data_prefix=None,
        pipeline=[
            dict(type='DecordInit'),
            dict(
                type='SampleFrames',
                clip_len=64,
                frame_interval=1,
                num_clips=1),
            dict(type='DecordDecode'),
            dict(type='Resize', scale=(-1, 256)),
            dict(
                type='MultiScaleCrop',
                input_size=224,
                scales=(1, 0.8),
                random_crop=True,
                max_wh_scale_gap=1),
            dict(type='Resize', scale=(224, 224), keep_ratio=False),
            dict(type='Flip', flip_ratio=0.5),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_bgr=False),
            dict(type='FormatShape', input_format='NCTHW'),
            dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
            dict(type='ToTensor', keys=['imgs', 'label'])
        ]),
    target_train=dict(
        type='MyVideoDataset',
        ann_file='/media/data_8T/UCF-HMDB/E2H/list_hmdb_train_split1.txt',
        data_prefix=None,
        pipeline=[
            dict(type='DecordInit'),
            dict(
                type='SampleFrames',
                clip_len=64,
                frame_interval=1,
                num_clips=1),
            dict(type='DecordDecode'),
            dict(type='Resize', scale=(-1, 256)),
            dict(
                type='MultiScaleCrop',
                input_size=224,
                scales=(1, 0.8),
                random_crop=True,
                max_wh_scale_gap=1),
            dict(type='Resize', scale=(224, 224), keep_ratio=False),
            dict(type='Flip', flip_ratio=0.5),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_bgr=False),
            dict(type='FormatShape', input_format='NCTHW'),
            dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
            dict(type='ToTensor', keys=['imgs', 'label'])
        ],
        w_pseudo='vid',
        ps_filter_by='frame_v2',
        pseudo_gt_dict=
        '/media/data_8T/UCF-HMDB/ablation_incepi3d/img2vid/e2h/split1/step1_grl_tsn_img_model_64d/v4/step2_vid_model/0.7/epoch_10_test3clip_vid_ps.npy',
        ps_thresh=0.7,
        data_dir='/media/data_8T'),
    val=dict(
        type='MyVideoDataset',
        ann_file='/media/data_8T/UCF-HMDB/E2H/list_hmdb_val_split1.txt',
        data_prefix=None,
        pipeline=[
            dict(type='DecordInit'),
            dict(
                type='SampleFrames',
                clip_len=64,
                frame_interval=1,
                num_clips=3,
                test_mode=True),
            dict(type='DecordDecode'),
            dict(type='Resize', scale=(-1, 256)),
            dict(type='CenterCrop', crop_size=224),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_bgr=False),
            dict(type='FormatShape', input_format='NCTHW'),
            dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
            dict(type='ToTensor', keys=['imgs'])
        ]),
    test=dict(
        type='MyVideoDataset',
        ann_file='/media/data_8T/UCF-HMDB/E2H/list_hmdb_val_split1.txt',
        data_prefix=None,
        pipeline=[
            dict(type='DecordInit'),
            dict(
                type='SampleFrames',
                clip_len=64,
                frame_interval=1,
                num_clips=3,
                test_mode=True),
            dict(type='DecordDecode'),
            dict(type='Resize', scale=(-1, 256)),
            dict(type='CenterCrop', crop_size=224),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_bgr=False),
            dict(type='FormatShape', input_format='NCTHW'),
            dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
            dict(type='ToTensor', keys=['imgs'])
        ]),
    target_train_for_inference=dict(
        type='MyVideoDataset',
        ann_file='/media/data_8T/UCF-HMDB/E2H/list_hmdb_train_split1.txt',
        data_prefix=None,
        pipeline=[
            dict(type='DecordInit'),
            dict(
                type='SampleFrames',
                clip_len=64,
                frame_interval=1,
                num_clips=3,
                test_mode=True),
            dict(type='DecordDecode'),
            dict(type='Resize', scale=(-1, 256)),
            dict(type='CenterCrop', crop_size=224),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_bgr=False),
            dict(type='FormatShape', input_format='NCTHW'),
            dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
            dict(type='ToTensor', keys=['imgs'])
        ]))
gpu_ids = [0]
omnisource = False
module_hooks = []
