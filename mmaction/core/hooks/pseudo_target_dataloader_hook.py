from mmcv.runner import HOOKS, Hook, get_dist_info
from mmaction.datasets.builder import build_dataloader,  build_dataset
from mmcv.engine import single_gpu_test, multi_gpu_test
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
import numpy as np
import torch
@HOOKS.register_module()
class PseudoTargetHook(Hook):
    # todo
    #  in case of target train dataset ( with or without pseudo labels)
    #  1) in the beginning of training:
    #       if with pseudo labels:
    #           a) call filter pseudo, filter pseudo labels by confidence thresholding, and then update the video_infos  list
    #       b) initialize the target train dataloader with the new target train list and updated pseudo label scores dict
    #  2) in the end of each k-th epoch:
    #       if with pseudo labels:
    #           a)  perform inference with the trained model on target training set,  update pseudo scores dict,
    #           b) call filter_pseudo, filter pseudo labels by confidence thresholding, and then update the video_infos  list
    #           c) re-initialize the target train dataloader with the new target train list and the updated pseudo label scores dict

    def __init__(self, if_rawframe = False, if_distributed = False):
        self.if_rawframe = if_rawframe
        self.if_distributed = if_distributed

    def before_run(self, runner):
        self.cfg_target_train = runner.cfg_target_train
        self.cfg_target_train_for_inference = runner.cfg_target_train_for_inference
        # self.cfg_val = runner.cfg_val
        self.train_dataloader_setting = runner.train_dataloader_setting
        self.val_dataloader_setting = runner.val_dataloader_setting

        self.w_pseudo = runner.DA_config.w_pseudo
        self.ps_filter_by = runner.DA_config.ps_filter_by
        self.ps_thresh = runner.DA_config.ps_thresh
        self.update_ps_start_epoch = runner.DA_config.update_ps_start_epoch
        self.update_ps_freq = runner.DA_config.update_ps_freq #  update the pseudo labels for target train every k-th epoch


        # build target dataset inside the hook
        self.target_train_dataset = build_dataset( self.cfg_target_train)
        self.cfg_target_train.pseudo_gt_dict = None  # remove the initial pseudo gt file to avoid repetitve loading when initalizing dataset




        if self.w_pseudo == 'vid': #  todo video pseudo labels
            # in target_train_dataset_for_inference,  w_pseudo is False (not specified, default is False )
            self.target_train_dataset_for_inference = build_dataset(self.cfg_target_train_for_inference,  dict(test_mode=True))
            self.target_train_dataloader_for_inference = build_dataloader(self.target_train_dataset_for_inference, **self.val_dataloader_setting)

            # filter pseudo labels by confidence thresholding,  then  video_infos list sampled
            self.target_train_dataset.filter_pseudo_vid(ps_thresh_percent= self.ps_thresh, epoch='begin')
        elif self.w_pseudo == 'img':   # todo frame-level pseudo labels from image backbone
            self.target_train_dataset_for_inference = build_dataset(self.cfg_target_train_for_inference,dict(test_mode=True))
            self.target_train_dataloader_for_inference = build_dataloader(self.target_train_dataset_for_inference,  **self.val_dataloader_setting)

            self.target_train_dataset.filter_pseudo_img(ps_thresh_percent=self.ps_thresh, epoch='begin')


        runner.logger.info(f'Before run: target train dataloader initialized! ')

        # pass the new dataloader to runner for training
        runner.target_train_loader = build_dataloader( self.target_train_dataset , **self.train_dataloader_setting)


    def after_run(self, runner):
        pass

    def before_epoch(self, runner):
        pass

    def after_epoch(self, runner):
        pass

    def after_train_epoch(self, runner):
        if self.w_pseudo in ['vid', 'img'] and self.update_ps_freq != -1:
            if (runner.epoch +1)  >= self.update_ps_start_epoch  and  (runner.epoch +1) % self.update_ps_freq == 0:
                # todo  perform inference on target training set using the trained model , update the pseudo scores dict
                runner.logger.info(f'End of Epoch {runner.epoch +1} - Computing pseudo labels on target training set of {len(self.target_train_dataset_for_inference)} samples ...')

                if self.if_distributed:
                    # model = MMDistributedDataParallel(
                    #     runner.model.cuda(),
                    #     device_ids=[torch.cuda.current_device()],
                    #     broadcast_buffers=False)
                    # results = multi_gpu_test(model, self.target_train_dataloader_for_inference,  DA_config=runner.DA_config)

                    results = multi_gpu_test(runner.model, self.target_train_dataloader_for_inference,
                                             DA_config=runner.DA_config)
                else:
                    if hasattr( runner, 'DA_config'):
                        # results is a list of (n_class,)  array as confidence scores
                        results = single_gpu_test(runner.model, self.target_train_dataloader_for_inference, DA_config= runner.DA_config)
                    else:
                        results = single_gpu_test(runner.model, self.target_train_dataloader_for_inference)

                rank, _ = get_dist_info()
                if rank == 0:
                    # new pseudo score dict
                    pred_scores_dict = dict()
                    for vid_idx in range(len(results)):
                        pred_label = np.argmax( results[vid_idx])
                        max_predict_score = results[vid_idx][pred_label]
                        if self.if_rawframe:
                            frame_dir = self.target_train_dataset_for_inference.video_infos[vid_idx]['frame_dir']
                            vid_label = self.target_train_dataset_for_inference.video_infos[vid_idx]['label']
                            pred_scores_dict.update(
                                {frame_dir: (vid_label, pred_label, max_predict_score, results[vid_idx])})
                        else:
                            vid_filename = self.target_train_dataset_for_inference.video_infos[vid_idx]['filename']
                            vid_label = self.target_train_dataset_for_inference.video_infos[vid_idx]['label']
                            pred_scores_dict.update({vid_filename: (vid_label, pred_label, max_predict_score, results[vid_idx]) })

                    # generate new target train dataset
                    self.target_train_dataset = build_dataset( self.cfg_target_train)  # a new & complete video_infos list
                    self.target_train_dataset.pseudo_scores_dict = pred_scores_dict  # new pseudo score dict

                    # filter pseudo labels,  then sample the  video_infos  list  based on confidence threshold
                    self.target_train_dataset.filter_pseudo_vid(ps_thresh_percent=self.ps_thresh, epoch= runner.epoch)
                    if self.w_pseudo == 'img':
                        self.target_train_dataset.vid_ps_from_img = dict()
                        for vid_path, values_ in self.target_train_dataset.pseudo_scores_dict.items():
                            self.target_train_dataset.vid_ps_from_img.update({vid_path: values_[1] })

                    # re-intialize dataloader
                    runner.logger.info(f'Epoch {runner.epoch+1}: target train dataloader RE-initialized! ')
                    runner.target_train_loader = build_dataloader(self.target_train_dataset,  **self.train_dataloader_setting)





    def before_iter(self, runner):
        pass

    def after_iter(self, runner):
        pass

