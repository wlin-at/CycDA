# from tools.uda_action_recog.utils import check_username
# from ../../..tools.uda_action_recog.utils import check_username
# import os.path as osp
# from tools.utils.utils import mkdir
_base_ = [
    '../../../_base_/models/i3d_r50.py', '../../../_base_/schedules/sgd_100e.py',
    '../../../_base_/default_runtime.py'
]

# /home/eicg/action_recognition_codes/domain_adaptation/mmaction2/configs/recognition/i3d/i3d_r50_32x2x1_100e_kinetics400_rgb_video.py --gpu-ids 0 1 2 3 --validate
# /home/eicg/action_recognition_codes/domain_adaptation/mmaction2/configs/recognition/i3d/i3d_r50_32x2x1_100e_kinetics400_rgb_video.py --gpu-ids 0 1 2 3 --validate


#./tools/dist_train_da.sh configs/recognition/i3d/i3d_r50_32x2x1_100e_kinetics400_rgb_video_tmpagg_DA_distri_krios.py 4 --gpu-ids 0 1 2 3 --validate
#python tools/train_da.py configs/recognition/i3d/i3d_r50_32x2x1_100e_kinetics400_rgb_video_tmpagg_single_krios.py --gpu-ids 0 --validate

env_id = 0
total_epochs = 20
val_frequency = 5
data_dir = ['/media/data_8T', '/data/lin'][env_id]
new_data_dir = '/data/lin' # the new_data_dir  will be contained in the pseudo label dict keys of video paths


debug = [False, True][0]

source_to_target = '2ucf_all'
split_nr = 1
n_class = 101

optimizer_cfg = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0000001)
paramwise_cfg = dict(custom_keys={
        'backbone': dict(lr_mult=0.1, decay_mult=1.0)}  ) # todo other parts have the learning of 0.01

num_clips = 1
num_clips_val = 1 # todo at test time, the score of 10 clips will be averaged
num_clips_test= 1
clip_len = 64
frame_interval = 1

save_last = [False, True][1]
save_best = [False, True][0]

videos_per_gpu = 6 # todo only one batch size is allowed
workers_per_gpu=0
videos_per_gpu_val_dataloader = 2
videos_per_gpu_test_dataloader = 1

custom_hooks = [
    dict(type='PseudoTargetHook',
         priority = 'NORMAL')
]

beta_high = 0.1
ps_thresh = 0.7
ps_filter_by = [ 'frame', 'frame_v2','vid', 'cbframe', 'cbvid' ][1]  # todo only used for frame-level pseudo labels
pseudo_gt_dict = data_dir + '/UCF-HMDB/ucf_hmdb_img_train/BU101_train_ucf_all_val_resnet_grl_tsn/20220208_212310_epoch20_frame_ps.npy'
log_str = f'clip{num_clips}x{clip_len}_w_ps_img{ps_thresh}_strid{frame_interval}_testclip{num_clips_val}'

pretrained_model_path = data_dir + '/mmaction2_checkpoints/InceptionI3d/rgb_imagenet.pt'

embed_dim=128
model = dict(
    type='RecognizerI3dDA',
    backbone=dict(
        type='InceptionI3d',  # todo name of the backbone
        pretrained_model_path=pretrained_model_path),
    cls_head=None,
    neck=None,
    train_cfg=None,
    test_cfg=dict(average_clips='prob'),
    spatial_type = 'avg',
    spatial_dropout_ratio = 0.5,
    feat_dim = 1024,
    n_class = n_class,  #  todo number of classes
    init_std = 0.01,
    embed_dim_t = embed_dim,

    d_clsfr_mlp_dim= embed_dim,
    clsfr_mlp_dim= embed_dim,

    # tmp_agg_type = 'avg',
    # # temporal transformer
    # n_transformer_blocks= 2,
    # n_heads= 8,
    # intermediate_dim= 256,
    # n_clips=num_clips,
    # token_pool_type= ['cls', 'mean', 'w_sum'][2], #  how to aggragate all tokens to a final representation
    # head_dim= 64,
    # transformer_dropout= 0.0,
    # emb_dropout= 0.0,
)


# env_id = check_username()
# if env_id == 0:
# main_dir = ['/media/data_8T', '/data/lin' ][0]
# dataset settings
# dataset_type = 'RawframeDataset'


DA_config = dict(
    batch_size = videos_per_gpu,
# todo #########################################
    # todo #########################################
    model_type = model['type'],
    source_to_target = source_to_target,
    split_nr = split_nr,
    experiment_type = ['source_only', 'target_only', 'source_and_target', 'source_and_target_pseudo', 'DA', 'compute_pseudo_labels'][-1],
    dataload_iter = ['min', 'max'][0], # todo  source only,  set the source loader ucf as the main loader
    use_target = ['none', 'Sv', 'uSv'][2] ,
    # todo #########################################
    # todo #########################################
    test_domain_label = 1,
    weight_clip_clspred = 0,  # todo w/o clip clspred
    # weight_clip_domainpred = 1, # todo w/o clip domainpred
    weight_clip_clspred_vid_ps = 1, #  todo with CE loss on pseudo labeled target data
    # temporal transformer
    n_clips = num_clips,

    ## training config
    adv_DA = [False, True][0],

    w_pseudo = [False, 'vid', 'img'][2],
    ps_filter_by= ps_filter_by,
    pseudo_gt_dict= pseudo_gt_dict,
    ps_thresh=ps_thresh,
    update_ps_start_epoch=-1,
    update_ps_freq=-1, #  never update the pseudo labels
)

dataset_type = 'MyVideoDataset'



#  these mean and std are used for both Kinetics400 and HMDB51, ActivityNet, UCF101
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_bgr=False)
train_pipeline = [
    dict(type= 'DecordInit'), # initialize the video reader
    dict(type='SampleFrames', clip_len=clip_len, frame_interval=frame_interval, num_clips=num_clips),  # only 1 temporal window from each video
    dict(type='DecordDecode'),
    # dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(-1, 256)),  #  Resize to a specific size
    dict(
        type='MultiScaleCrop',  # Crop images with a list of randomly selected scales.   this is equal to 'scale jittering'
        input_size=224,
        scales=(1, 0.8),  #  [1, .875, 0.75]  is used in MM-SADA
        random_crop=True,  # if set to True, the cropping bbox will be randomly sampled, otherwise it will be sampler from fixed regions.   if True, this is random crop
        # notice that the scale of w and h are both randomly chosen (there might be different scales chosen for w and h)
        # then the shift along w and h are also randomly chosen

        max_wh_scale_gap=1  # maximum gap of w and h scale levels, default is 1.
                            #  if set to 0, it means that the scale of w and h must be the same!!!
    ),
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
    dict(type='Flip', flip_ratio=0.5), #  horizontal flip
    dict(type='Normalize', **img_norm_cfg),  # normalize each image frame
    dict(type='FormatShape', input_format='NCTHW'), # NCTHW   # (N_crops x N_clips)  x C x L x H x W,   (N,3,32,224,224)
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'label'])
]
val_pipeline = [
    dict(type= 'DecordInit'), # initialize the video reader
    dict(
        type='SampleFrames',
        clip_len=clip_len,
        frame_interval=frame_interval,
        num_clips=num_clips_val,
        test_mode=True),
    dict(type='DecordDecode'),
    # dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='CenterCrop', crop_size=224),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]
test_pipeline = [
    dict(type= 'DecordInit'), # initialize the video reader
    dict(
        type='SampleFrames',
        clip_len=clip_len,
        frame_interval=frame_interval,
        num_clips=num_clips_test,
        test_mode=True),
    dict(type='DecordDecode'),
    # dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(-1, 256)),
    # dict(type='ThreeCrop', crop_size=256), #3 crops of squared patches (left, middle right)  or (top, down, middle )
    dict(type='CenterCrop', crop_size=224),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]



if save_best:
    evaluation = dict(  # config of evaluation during training ,  validation every 5-th epoch,         --validate has to be set in the training command
        interval=val_frequency, metrics=['top_k_accuracy', 'mean_class_accuracy'])
else:
    evaluation = dict(  # config of evaluation during training ,  validation every 5-th epoch,         --validate has to be set in the training command
        interval=val_frequency, save_best=None,  metrics=['top_k_accuracy', 'mean_class_accuracy'])


checkpoint_config = dict(interval=-1, save_last = save_last)
log_config = dict(
    interval=20,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook'),
    ])
log_level = 'DEBUG' # The level of logging
#work_dir = '/data/lin/UCF-HMDB/work_dirs/i3d_r50_32x2x1_100e_kinetics400_rgb/'
work_main_dir = ['./work_dirs/', '/data/lin/UCF-HMDB/work_dirs/'][env_id]



find_unused_parameters = True
