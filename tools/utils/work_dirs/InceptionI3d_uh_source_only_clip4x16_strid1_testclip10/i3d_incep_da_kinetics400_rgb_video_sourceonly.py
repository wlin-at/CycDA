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
        '/media/data_8T/mmaction2_checkpoints/InceptionI3d/rgb_imagenet.pt'),
    cls_head=None,
    train_cfg=None,
    test_cfg=dict(average_clips='prob'),
    neck=None,
    spatial_type='avg',
    spatial_dropout_ratio=0.5,
    feat_dim=1024,
    n_class=12,
    init_std=0.01,
    embed_dim_t=64,
    d_clsfr_mlp_dim=64,
    clsfr_mlp_dim=64)
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=40, norm_type=2))
lr_config = dict(policy='step', step=[40, 80])
total_epochs = 5
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
val_frequency = 1
data_dir = '/media/data_8T'
debug = True
optimizer_cfg = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=1e-07)
paramwise_cfg = dict(
    custom_keys=dict(backbone=dict(lr_mult=0.1, decay_mult=1.0)))
num_clips = 4
num_clips_val = 10
num_clips_test = 10
clip_len = 16
frame_interval = 1
save_last = True
save_best = False
videos_per_gpu = 2
workers_per_gpu = 0
videos_per_gpu_val_dataloader = 2
videos_per_gpu_test_dataloader = 1
custom_hooks = [dict(type='PseudoTargetHook', priority='NORMAL')]
beta_high = 0.1
log_str = 'clip4x16_strid1_testclip10'
pretrained_model_path = '/media/data_8T/mmaction2_checkpoints/InceptionI3d/rgb_imagenet.pt'
DA_config = dict(
    batch_size=2,
    model_type='RecognizerI3dDA',
    source_to_target='uh',
    experiment_type='source_only',
    dataload_iter='max',
    use_target='none',
    test_domain_label=0,
    weight_clip_clspred=1,
    n_clips=4,
    adv_DA=False,
    w_pseudo=False,
    ps_filter_by=None,
    pseudo_gt_dict=None,
    ps_thresh=None,
    update_ps_start_epoch=-1,
    update_ps_freq=-1,
    exp_DA_name='baseline',
    source_train_list=
    '/media/data_8T/UCF-HMDB/datalist_new/list_ucf101_train_hmdb_ucf-feature_dummy.txt',
    target_train_list=
    '/media/data_8T/UCF-HMDB/datalist_new/list_hmdb51_train_hmdb_ucf-feature_dummy.txt',
    val_list=
    '/media/data_8T/UCF-HMDB/datalist_new/list_hmdb51_val_hmdb_ucf-feature_dummy.txt',
    test_list=
    '/media/data_8T/UCF-HMDB/datalist_new/list_hmdb51_val_hmdb_ucf-feature_dummy.txt'
)
dataset_type = 'MyVideoDataset'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_bgr=False)
train_pipeline = [
    dict(type='DecordInit'),
    dict(type='SampleFrames', clip_len=16, frame_interval=1, num_clips=4),
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
        clip_len=16,
        frame_interval=1,
        num_clips=10,
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
        clip_len=16,
        frame_interval=1,
        num_clips=10,
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
    interval=1,
    save_best=None,
    metrics=['top_k_accuracy', 'mean_class_accuracy'])
work_main_dir = './work_dirs/'
find_unused_parameters = True
work_dir = './work_dirs/InceptionI3d_uh_source_only_clip4x16_strid1_testclip10'
data = dict(
    videos_per_gpu=2,
    workers_per_gpu=0,
    val_dataloader=dict(videos_per_gpu=2),
    test_dataloader=dict(videos_per_gpu=1),
    source_train=dict(
        type='MyVideoDataset',
        ann_file=
        '/media/data_8T/UCF-HMDB/datalist_new/list_ucf101_train_hmdb_ucf-feature_dummy.txt',
        data_prefix=None,
        pipeline=[
            dict(type='DecordInit'),
            dict(
                type='SampleFrames',
                clip_len=16,
                frame_interval=1,
                num_clips=4),
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
        ann_file=
        '/media/data_8T/UCF-HMDB/datalist_new/list_hmdb51_train_hmdb_ucf-feature_dummy.txt',
        data_prefix=None,
        pipeline=[
            dict(type='DecordInit'),
            dict(
                type='SampleFrames',
                clip_len=16,
                frame_interval=1,
                num_clips=4),
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
        w_pseudo=False,
        ps_filter_by=None,
        pseudo_gt_dict=None,
        ps_thresh=None,
        data_dir='/media/data_8T'),
    val=dict(
        type='MyVideoDataset',
        ann_file=
        '/media/data_8T/UCF-HMDB/datalist_new/list_hmdb51_val_hmdb_ucf-feature_dummy.txt',
        data_prefix=None,
        pipeline=[
            dict(type='DecordInit'),
            dict(
                type='SampleFrames',
                clip_len=16,
                frame_interval=1,
                num_clips=10,
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
        ann_file=
        '/media/data_8T/UCF-HMDB/datalist_new/list_hmdb51_val_hmdb_ucf-feature_dummy.txt',
        data_prefix=None,
        pipeline=[
            dict(type='DecordInit'),
            dict(
                type='SampleFrames',
                clip_len=16,
                frame_interval=1,
                num_clips=10,
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
    target_train_for_inference=None)
gpu_ids = [0]
omnisource = False
module_hooks = []
