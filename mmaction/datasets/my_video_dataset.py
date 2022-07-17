# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp

from .base import BaseDataset
from .builder import DATASETS
import torch
# from tools.utils.utils import initialize_logger
from mmaction.utils import utils
import numpy as np
import getpass
import copy
import os
def frame_thresholding(pseudo_scores_dict = None, logger = None, thresh = None, n_frames_total = None, ps_thresh_percent_ = None, ):
    n_correct_frames = 0
    n_frames_above_thresh = 0

    n_vids_correct = 0
    n_vids_above_thresh = 0

    for vidname, items in pseudo_scores_dict.items():
        gt_label_seq, pred_label_seq, max_pred_score_seq, pred_scores_all = items[0], items[1], items[2], items[3]
        mask_frames_above_thresh = max_pred_score_seq >= thresh
        n_frames_above_thresh_this_vid = np.sum(mask_frames_above_thresh)
        n_frames_above_thresh += n_frames_above_thresh_this_vid
        n_correct_frames += np.sum(np.logical_and(mask_frames_above_thresh, gt_label_seq == pred_label_seq))

        # drive the video-level confidence score as the average of confidence scores of frames above threshold
        if n_frames_above_thresh_this_vid > 0:
            n_vids_above_thresh += 1
            pred_scores_all_filtered = pred_scores_all[mask_frames_above_thresh, :]  # (n_frames_above_thresh, 12)
            vid_confidence = np.mean(pred_scores_all_filtered, axis=0)  # (n_class, )
            vid_ps_label = np.argmax(vid_confidence)
            gt_label = gt_label_seq[0]
            if gt_label == vid_ps_label:
                n_vids_correct += 1
    percent_frames_above_thresh = float(n_frames_above_thresh) / n_frames_total * 100.0
    acc_frame = np.NaN if n_frames_above_thresh == 0 else float(n_correct_frames) / n_frames_above_thresh
    percent_vid_above_thresh = float(n_vids_above_thresh) / len(pseudo_scores_dict) * 100.0
    acc_vid_ps = np.NaN if n_vids_above_thresh == 0 else float(n_vids_correct) / n_vids_above_thresh
    logger.debug(
        f'Frame Thresh {thresh:.2f} ({ps_thresh_percent_ * 100.0:.1f}%) : #frames {n_frames_above_thresh}/{percent_frames_above_thresh:.2f}% , acc frame {acc_frame:.3f}'
        f'\t #vids {n_vids_above_thresh}/{percent_vid_above_thresh:.2f}%, acc vid {acc_vid_ps:.3f}')

def get_cls_thresh(n_class = None, confidence_class_dict =None, ps_thresh_percent_ = None):
    # compute threshold in each class
    cls_thresh = np.zeros((n_class,))
    # compute the threshold of vid-level confidence score for each class
    for cls_idx in range(n_class):
        pos_ = min(len(confidence_class_dict[cls_idx]) - 1, int(len(confidence_class_dict[cls_idx]) * float(ps_thresh_percent_)))
        thresh = confidence_class_dict[cls_idx][pos_]
        cls_thresh[cls_idx] = thresh
    return cls_thresh

def get_env_id():
    if getpass.getuser() == 'eicg':
        env_id = 0
    elif getpass.getuser() == 'lin':
        env_id = 1
    else:
        raise Exception("Unknown username!")
    return env_id

@DATASETS.register_module()
class MyVideoDataset(BaseDataset):

    #  todo the extended version of VideoDataset,
    #       1. during self-training:
    #           for target train dataset, support the option of outputting pseudo labels and the confidence score
    #       2. during contrastive learning:
    #           on source dataset, output the video index required in the memory bank

    """Video dataset for action recognition.

    The dataset loads raw videos and apply specified transforms to return a
    dict containing the frame tensors and other information.

    The ann_file is a text file with multiple lines, and each line indicates
    a sample video with the filepath and label, which are split with a
    whitespace. Example of a annotation file:

    .. code-block:: txt

        some/path/000.mp4 1
        some/path/001.mp4 1
        some/path/002.mp4 2
        some/path/003.mp4 2
        some/path/004.mp4 3
        some/path/005.mp4 3


    Args:
        ann_file (str): Path to the annotation file.
        pipeline (list[dict | callable]): A sequence of data transforms.
        start_index (int): Specify a start index for frames in consideration of
            different filename format. However, when taking videos as input,
            it should be set to 0, since frames loaded from videos count
            from 0. Default: 0.
        **kwargs: Keyword arguments for ``BaseDataset``.
    """

    def __init__(self, ann_file, pipeline, start_index=0, if_rawframe = False, with_offset = False, w_pseudo = False, vid_format = '.avi',
                 filename_tmpl='{:05}.jpg',
                 **kwargs):
        self.w_pseudo = w_pseudo
        self.if_rawframe = if_rawframe
        self.with_offset = with_offset
        self.vid_format = vid_format
        self.filename_tmpl = filename_tmpl
        if self.w_pseudo:   # todo the keys of self.pseudo_scores_dict are video names
            self.pseudo_scores_dict = None if kwargs['pseudo_gt_dict'] is None else  np.load(kwargs['pseudo_gt_dict'], allow_pickle=True  ).item()
            self.ps_filter_by = kwargs['ps_filter_by']
            self.ps_thresh = kwargs['ps_thresh']
            self.data_dir = kwargs['data_dir']

            env_id = get_env_id()
            if env_id == 0:
                new_pseudo_scores_dict = dict()
                if self.pseudo_scores_dict is not None:
                    for vid_path,value in self.pseudo_scores_dict.items():
                        if self.data_dir not in vid_path:
                            vid_path = vid_path.replace('/data/lin', self.data_dir)
                        new_pseudo_scores_dict.update( {vid_path : value})
                    self.pseudo_scores_dict = new_pseudo_scores_dict

        for key_ in ['w_pseudo',  'pseudo_gt_dict', 'ps_thresh', 'data_dir', 'ps_filter_by', 'if_rawframe' ]:
            if key_ in kwargs:
                del kwargs[key_]

        super().__init__(ann_file, pipeline, start_index=start_index, **kwargs)
        #  todo  if rawframes,   video_infos is list of  { 'frame_dir', 'total_frames', 'label' }
        # self.video_infos_origin = self.video_infos.copy()
        self.vid_path_prefix_dict = dict()
        self.n_frames_dict = dict()

        # if self.w_pseudo:
        #     # to filter out for debugging !!!!!!!!
        #     if self.pseudo_scores_dict is not None:
        #         vidname_list = [element_['frame_dir'].split('/')[-1] for element_ in self.video_infos]
        #         new_pseudo_scores_dict =dict()
        #         for vidname, data in self.pseudo_scores_dict.items():
        #             if vidname in vidname_list:
        #                 new_pseudo_scores_dict.update({vidname: data})
        #         self.pseudo_scores_dict = new_pseudo_scores_dict


        for video_info in self.video_infos:
            if self.if_rawframe:
                vid_path = video_info['frame_dir']
                vid_path_prefix, vidname = os.path.split(vid_path)
            else:
                vid_path = video_info['filename']
                vid_path_prefix, vidname = os.path.split(vid_path)
                vidname = vidname.split('.')[0]

            self.vid_path_prefix_dict.update({ vidname: vid_path_prefix})
            if self.if_rawframe:
                self.n_frames_dict.update({ vidname:  video_info['total_frames']  })



        # if self.if_rawframe:
        #     self.all_vid_paths = [video_info['frame_dir'] for video_info in self.video_infos]
        # else:
        #     self.all_vid_paths =  [  video_info['filename']   for video_info in self.video_infos ]

        # todo
        #  in case of target train dataset with pseudo labels
        #  1) in the beginning of training: call filter pseudo
        #  2) in the end of each k-th epoch:
        #       a) update pseudo scores dict,  perform inference with the trained model on target training set
        #       b) call filter_pseudo, filter pseudo labels by confidence thresholding, and then update the target train list (video_infos list?? )
        #       c) re-initialize the target train dataloader with the new target train list and the updated pseudo label scores dict

    def frame_thresholding_collect_vid_info(self, logger = None,  ps_thresh=None, n_frames_total = None, ps_thresh_percent= None, ):

        self.video_infos = []
        n_correct_frames = 0
        n_frames_above_thresh = 0
        n_vid_correct = 0
        # n_vids_above_thresh = 0
        self.vid_ps_from_img = dict()

        for vidname, items in self.pseudo_scores_dict.items():
            gt_label_seq, pred_label_seq, max_pred_score_seq, pred_scores_all = items[0], items[1], items[2], items[3]
            mask_frames_above_thresh = max_pred_score_seq >= ps_thresh
            n_frames_above_thresh_this_vid = np.sum(mask_frames_above_thresh)
            n_frames_above_thresh += n_frames_above_thresh_this_vid
            n_correct_frames += np.sum(np.logical_and(mask_frames_above_thresh, gt_label_seq == pred_label_seq))
            if n_frames_above_thresh_this_vid > 0:
                gt_label = int(gt_label_seq[0])
                # # todo avoid case of  vid_1  and vid_10
                # for vidpath_ in self.all_vid_paths:
                #     if vidname + '.avi' in vidpath_:
                #         vidpath = vidpath_
                #         break

                if self.if_rawframe:
                    vidpath = os.path.join(self.vid_path_prefix_dict[vidname], vidname )
                    self.video_infos.append( {  'frame_dir': vidpath, 'total_frames': self.n_frames_dict[vidname], 'label': gt_label } )
                else:
                    vidpath = os.path.join(self.vid_path_prefix_dict[vidname], vidname + self.vid_format)
                    self.video_infos.append({'filename': vidpath, 'label': gt_label})
                pred_scores_all_filtered = pred_scores_all[mask_frames_above_thresh, :]  # (n_frames_above_thresh, 12)
                vid_confidence = np.mean(pred_scores_all_filtered, axis=0)  # (n_class, )
                vid_ps_label = np.argmax(vid_confidence)
                self.vid_ps_from_img.update({vidpath: vid_ps_label})
                if gt_label == vid_ps_label:
                    n_vid_correct += 1

        percent_frames_above_thresh = float(n_frames_above_thresh) / n_frames_total * 100.0
        acc_frame = np.NaN if n_frames_above_thresh == 0 else float(n_correct_frames) / n_frames_above_thresh
        self.num_target_train = len(self.video_infos)  # number of sampled videos
        n_vids_above_thresh = self.num_target_train

        percent_vid_above_thresh = float(n_vids_above_thresh) / len(self.pseudo_scores_dict) * 100.0
        acc_vid_ps = np.NaN if n_vids_above_thresh == 0 else float(n_vid_correct) / n_vids_above_thresh
        logger.debug(
            f'Chosen Frame Thresh {ps_thresh:.2f} ({ps_thresh_percent * 100.0:.1f}%) : #frames {n_frames_above_thresh}/{percent_frames_above_thresh:.2f}% , acc frame {acc_frame:.3f}'
            f'\t #vids {n_vids_above_thresh}/{percent_vid_above_thresh:.2f}%, acc vid {acc_vid_ps:.3f}')

    def filter_pseudo_img(self, ps_thresh_percent, epoch):
        # todo  filter frame-level pseudo labels to keep a specific percent of samples with the highest confidence scores
        #  then derive the video-level pseudo label
        #  if a video has at least one frame above threshold, it will be used in the self-training
        epoch_str = epoch if isinstance(epoch, str) else str(epoch + 1)
        logger = utils.logger  # global variable in utils.py
        n_class = list(self.pseudo_scores_dict.values())[0][3].shape[-1]  #  all_class_scores has shape (m_frames, n_class )
        ######################################## compute frame-level accuracy for each class
        n_correct_total = 0
        n_correct_class_total = np.zeros((n_class,))
        n_samples_class_total = np.zeros((n_class,))
        acc_class_total = np.zeros((n_class,))
        for vidname, items in self.pseudo_scores_dict.items():
            # vidname : ( gt label, pred label,  max_pred_score, predicted scores for all classes )
            gt_label_seq, pred_label_seq, max_pred_score_seq = items[0], items[1], items[2]
            n_frames = len(gt_label_seq)
            assert len(set(gt_label_seq)) == 1  # check if all the gt labels in a video are identical
            gt_label = int(gt_label_seq[0])

            n_samples_class_total[gt_label] += n_frames
            n_correct_frames = np.sum(gt_label_seq == pred_label_seq)
            n_correct_total += n_correct_frames
            n_correct_class_total[gt_label] += n_correct_frames
        n_frames_total = np.sum(n_samples_class_total)
        acc_total = float(n_correct_total) / n_frames_total
        for class_idx in range(n_class):
            acc_class_total[class_idx] = float(n_correct_class_total[class_idx]) / n_samples_class_total[class_idx]

            logger.debug(
                f'Epoch {epoch_str} class {class_idx}: #frames {n_samples_class_total[class_idx]}, acc {acc_class_total[ class_idx]:.3f}')
        logger.debug('Total:')
        logger.debug(f'Epoch {epoch_str} Total : #frames {n_frames_total}/100% , acc {acc_total:.3f}')

        if self.ps_filter_by == 'frame':
            raise Exception(f'{self.ps_filter_by} is not a valid ps_filter_by parameter!')
            #########################  Compute accuracy of frames above thresholds according to percent 0.9 ~ 0.1
            confidence_list = []
            for vidname, items in self.pseudo_scores_dict.items():
                confidence_list.extend(items[2].tolist())
            # confidence_list = [ np.mean(items[2])  for vidname, items in pred_scores_dict.items()]
            confidence_list = sorted(confidence_list, reverse=True)
            for ps_thresh_percent_ in np.arange(0.9, 0, -0.1):
                pos_ = min(len(confidence_list) - 1, int(len(confidence_list) * float(ps_thresh_percent_)))
                thresh = confidence_list[pos_]
                frame_thresholding(pseudo_scores_dict=self.pseudo_scores_dict, logger=logger, thresh=thresh, n_frames_total=n_frames_total, ps_thresh_percent_= ps_thresh_percent_)

            # todo update  self.video_infos , save ps label in self.vid_ps_from_img
            #  the complete list of videos is stored in self.pseudo_scores_dict
            pos_ = min(len(confidence_list) - 1, int(len(confidence_list) * float(ps_thresh_percent)))
            ps_thresh =  confidence_list[pos_]
            self.frame_thresholding_collect_vid_info(logger=logger, ps_thresh=ps_thresh, n_frames_total=n_frames_total,
                                                ps_thresh_percent=ps_thresh_percent, )

        elif self.ps_filter_by == 'frame_v2':
            # ps_thresh_percent is the percentage of videos,
            # frame threshold is determined to make sure [ps_thresh_percent] of videos remain
            vid_max_score_list = []
            for vidname, items in self.pseudo_scores_dict.items():
                vid_max_score_list.append(  np.max(items[2])  ) # list of the maximum frame-level scores in each video
            vid_max_score_list = sorted(vid_max_score_list) # sort in ascending order
            n_vids = len(vid_max_score_list)
            # todo thresh_list contains thresholds corresponding to  N, N-1, N-2, ..., 1 vid
            thresh_list = [0.0 ] + list(   ( np.array( vid_max_score_list[: n_vids-1] ) + np.array( vid_max_score_list[1: n_vids] )  ) / 2.0    )
            for ps_thresh_percent_ in np.arange(1.0, 0, -0.1):
                n_vids_remain = int(n_vids * ps_thresh_percent_)
                thresh = thresh_list[n_vids - n_vids_remain]
                frame_thresholding(pseudo_scores_dict=self.pseudo_scores_dict, logger=logger, thresh=thresh, n_frames_total=n_frames_total, ps_thresh_percent_=ps_thresh_percent_)

            n_vids_remain = int( n_vids * ps_thresh_percent)
            ps_thresh = thresh_list[ n_vids - n_vids_remain ]
            self.frame_thresholding_collect_vid_info(logger=logger, ps_thresh=ps_thresh, n_frames_total=n_frames_total,
                                                ps_thresh_percent=ps_thresh_percent, )

        elif self.ps_filter_by == 'vid':
            # derive the video-level confidence score as the average of confidence scores of frames, then do thresholding
            vid_ps_score_dict = dict()
            confidence_list = []
            for vidname, items in self.pseudo_scores_dict.items():
                gt_label_seq, pred_label_seq, max_pred_score_seq, pred_scores_all = items[0], items[1], items[2], items[3]
                # derive the video-level confidence score as the average of confidence scores of frames
                vid_confidence = np.mean( pred_scores_all, axis=0)
                vid_ps_label = np.argmax(vid_confidence )
                vid_max_confidence = vid_confidence[vid_ps_label]
                gt_label = gt_label_seq[0]
                vid_ps_score_dict.update({ vidname: [ gt_label, vid_ps_label, vid_max_confidence, vid_confidence ] })
                confidence_list.append( vid_max_confidence)
            confidence_list = sorted(confidence_list, reverse=True)
            for ps_thresh_percent_ in np.arange(0.9, 0, -0.1):
                pos_ = min( len(confidence_list)-1, int(len(confidence_list) * float(ps_thresh_percent_ ) ) )
                thresh = confidence_list[pos_]

                n_vids_correct = 0
                n_vids_above_thresh = 0
                for vidname, items in vid_ps_score_dict.items():
                    gt_label, vid_ps_label, vid_max_confidence, vid_confidence = items[0], items[1], items[2], items[3]
                    if vid_max_confidence >= thresh:
                        n_vids_above_thresh += 1
                        if gt_label == vid_ps_label:
                            n_vids_correct += 1
                percent_vid_above_thresh = float(n_vids_above_thresh) / len( vid_ps_score_dict ) * 100.0
                acc_vid_ps = np.NaN if n_vids_above_thresh == 0 else float(n_vids_correct) / n_vids_above_thresh
                logger.debug(f'\t #vids {n_vids_above_thresh}/{percent_vid_above_thresh:.2f}%, acc vid {acc_vid_ps:.3f}' )

            # todo update self.video_infos, save ps label in self.vid_ps_from_img
            #   the compete list of videos is stored in self.pseudo_scores_dict
            pos_ = min(len(confidence_list )-1, int(len(confidence_list) * float( ps_thresh_percent) ))
            ps_thresh = confidence_list[pos_]
            self.video_infos = []
            n_vids_correct = 0
            self.vid_ps_from_img = dict()
            for vidname, items in vid_ps_score_dict.items():
                gt_label, vid_ps_label, vid_max_confidence, vid_confidence = items[0], items[1], items[2], items[3]
                if vid_max_confidence >= ps_thresh:
                    # for vidpath_ in self.all_vid_paths:
                    #     if vidname + '.avi' in vidpath_:
                    #         vidpath = vidpath_
                    #         break
                    if self.if_rawframe:
                        vidpath = os.path.join(self.vid_path_prefix_dict[vidname], vidname)
                        self.video_infos.append(
                            {'frame_dir': vidpath, 'total_frames': self.n_frames_dict[vidname], 'label': gt_label})
                    else:
                        vidpath = os.path.join(self.vid_path_prefix_dict[vidname], vidname + self.vid_format)
                        self.video_infos.append({'filename': vidpath, 'label': gt_label})
                    self.vid_ps_from_img.update( { vidpath : vid_ps_label })
                    if gt_label == vid_ps_label:
                        n_vids_correct += 1
            self.num_target_train = len(self.video_infos)
            n_vids_above_thresh = self.num_target_train

            percent_vid_above_thresh = float(n_vids_above_thresh) / len(vid_ps_score_dict) * 100.0
            acc_vid_ps = np.NaN if n_vids_above_thresh == 0 else float(n_vids_correct) / n_vids_above_thresh
            logger.debug(f'Chosen threshold:   #vids {n_vids_above_thresh}/{percent_vid_above_thresh:.2f}%, acc vid {acc_vid_ps:.3f}'  )

        elif self.ps_filter_by == 'cbvid':
            # derive the video-level confidence score as the average of confidence scores of frames,
            # then perform class-balanced sampling
            vid_ps_score_dict = dict()
            confidence_class_dict = dict()
            for cls_idx in range(n_class):
                confidence_class_dict.update({ cls_idx: []}) #  a dictionary of list of vid-level confidence scores
            for vidname, items in self.pseudo_scores_dict.items():
                gt_label_seq, pred_label_seq, max_pred_score_seq, pred_scores_all = items[0], items[1], items[2], items[3]
                # derive the video-level confidence score as the average of confidence scores of frames
                vid_confidence = np.mean(pred_scores_all, axis=0)
                vid_ps_label = np.argmax(vid_confidence)
                vid_max_confidence = vid_confidence[vid_ps_label]
                gt_label = gt_label_seq[0]
                vid_ps_score_dict.update({vidname: [gt_label, vid_ps_label, vid_max_confidence, vid_confidence]})
                confidence_class_dict[vid_ps_label].append(vid_max_confidence)
            for cls_idx in range(n_class):
                confidence_class_dict[cls_idx] = sorted(confidence_class_dict[cls_idx], reverse=True)
            for ps_thresh_percent_ in np.arange(0.9, 0, -0.1):
                cls_thresh = get_cls_thresh(n_class = n_class, confidence_class_dict = confidence_class_dict, ps_thresh_percent_ = ps_thresh_percent_)
                n_vids_correct = 0
                n_vids_above_thresh = 0
                for vidname, items in vid_ps_score_dict.items():
                    gt_label, vid_ps_label, vid_max_confidence, vid_confidence = items[0], items[1], items[2], items[3]
                    thresh = cls_thresh[vid_ps_label] # class-dependent threshold
                    if vid_max_confidence >= thresh:
                        n_vids_above_thresh += 1
                        if gt_label == vid_ps_label:
                            n_vids_correct += 1
                percent_vid_above_thresh = float(n_vids_above_thresh) / len(vid_ps_score_dict) * 100.0
                acc_vid_ps = np.NaN if n_vids_above_thresh == 0 else float(n_vids_correct) / n_vids_above_thresh
                logger.debug(f'\t #vids {n_vids_above_thresh}/{percent_vid_above_thresh:.2f}%, acc vid {acc_vid_ps:.3f}')

            self.video_infos = []
            n_vids_correct = 0
            self.vid_ps_from_img = dict()
            cls_thresh = get_cls_thresh(n_class=n_class, confidence_class_dict=confidence_class_dict,
                                        ps_thresh_percent_=ps_thresh_percent)
            for vidname, items in vid_ps_score_dict.items():
                gt_label, vid_ps_label, vid_max_confidence, vid_confidence = items[0], items[1], items[2], items[3]
                ps_thresh = cls_thresh[vid_ps_label] # class-dependent threshold
                if vid_max_confidence >= ps_thresh:
                    # for vidpath_ in self.all_vid_paths:
                    #     if vidname + '.avi' in vidpath_:
                    #         vidpath = vidpath_
                    #         break
                    if self.if_rawframe:
                        vidpath = os.path.join(self.vid_path_prefix_dict[vidname], vidname)
                        self.video_infos.append(
                            {'frame_dir': vidpath, 'total_frames': self.n_frames_dict[vidname], 'label': gt_label})
                    else:
                        vidpath = os.path.join(self.vid_path_prefix_dict[vidname], vidname + self.vid_format)
                        self.video_infos.append({'filename': vidpath, 'label': gt_label})
                    self.vid_ps_from_img.update({vidpath: vid_ps_label})
                    if gt_label == vid_ps_label:
                        n_vids_correct += 1
            self.num_target_train = len(self.video_infos)
            n_vids_above_thresh = self.num_target_train

            percent_vid_above_thresh = float(n_vids_above_thresh) / len(vid_ps_score_dict) * 100.0
            acc_vid_ps = np.NaN if n_vids_above_thresh == 0 else float(n_vids_correct) / n_vids_above_thresh
            logger.debug(f'Chosen threshold:   #vids {n_vids_above_thresh}/{percent_vid_above_thresh:.2f}%, acc vid {acc_vid_ps:.3f}')


        elif self.ps_filter_by == 'cbframe':
            # vidname : ( gt label, pred label,  max_pred_score, predicted scores for all classes )
            confidence_class_dict = dict()
            for cls_idx in range(n_class):
                confidence_class_dict.update({ cls_idx: []}) #  a dictionary of list of frame-level confidence scores
            for vidname, items in self.pseudo_scores_dict.items():
                pred_label_seq, max_pred_score_seq = items[1], items[2]
                for frame_idx in range(len(pred_label_seq)):
                    confidence_class_dict[ int(pred_label_seq[frame_idx])].append( max_pred_score_seq[frame_idx])
            for cls_idx in range(n_class):
                confidence_class_dict[cls_idx] = sorted( confidence_class_dict[cls_idx], reverse=True )
            for ps_thresh_percent_ in np.arange( 0.9, 0, -0.1):
                cls_thresh = np.zeros((n_class,))
                # compute the threshold of frame-level confidence score for each class
                for cls_idx in range(n_class):
                    pos_ = min( len(confidence_class_dict[cls_idx]) -1, int( len(confidence_class_dict[cls_idx]) *float(ps_thresh_percent_)) )
                    thresh = confidence_class_dict[cls_idx][pos_]
                    cls_thresh[cls_idx] = thresh
                n_correct_frames = 0
                n_frames_above_thresh = 0
                n_vids_correct = 0
                n_vids_above_thresh = 0
                for vidname, items in self.pseudo_scores_dict.items():
                    gt_label_seq, pred_label_seq, max_pred_score_seq, pred_scores_all = items[0], items[1], items[2],  items[3]
                    thresh_seq =  np.array([  cls_thresh[ int(pred_label_) ]  for pred_label_ in pred_label_seq ])
                    mask_frames_above_thresh = np.greater_equal( max_pred_score_seq, thresh_seq )
                    n_frames_above_thresh_this_vid = np.sum(mask_frames_above_thresh)
                    n_frames_above_thresh += n_frames_above_thresh_this_vid
                    n_correct_frames += np.sum(np.logical_and(mask_frames_above_thresh, gt_label_seq == pred_label_seq))

                    # derive the video-level confidence score as the average of confidence scores of frames above threshold
                    if n_frames_above_thresh_this_vid > 0:
                        n_vids_above_thresh += 1
                        pred_scores_all_filtered = pred_scores_all[mask_frames_above_thresh,   :]  # (n_frames_above_thresh, 12)
                        vid_confidence = np.mean(pred_scores_all_filtered, axis=0)  # (n_class, )
                        vid_ps_label = np.argmax(vid_confidence)
                        gt_label = gt_label_seq[0]
                        if gt_label == vid_ps_label:
                            n_vids_correct += 1
                percent_frames_above_thresh = float(n_frames_above_thresh) / n_frames_total * 100.0
                acc_frame = np.NaN if n_frames_above_thresh == 0 else float(n_correct_frames) / n_frames_above_thresh
                percent_vid_above_thresh = float(n_vids_above_thresh) / len(self.pseudo_scores_dict) * 100.0
                acc_vid_ps = np.NaN if n_vids_above_thresh == 0 else float(n_vids_correct) / n_vids_above_thresh
                logger.debug(
                    f'Class balanced Frame Thresh  ({ps_thresh_percent_ * 100.0:.1f}%) : #frames {n_frames_above_thresh}/{percent_frames_above_thresh:.2f}% , acc frame {acc_frame:.3f}'
                    f'\t #vids {n_vids_above_thresh}/{percent_vid_above_thresh:.2f}%, acc vid {acc_vid_ps:.3f}')

            # todo update self.video_infos,  save ps_label in self.vid_ps_from_img
            #   the complete list of videos is stored in self.pseudo_scores_dict
            cls_thresh = np.zeros((n_class,))
            for cls_idx in range(n_class):
                pos_ = min(len(confidence_class_dict[cls_idx]) - 1, int(len(confidence_class_dict[cls_idx]) * float(ps_thresh_percent)))
                thresh = confidence_class_dict[cls_idx][pos_]
                cls_thresh[cls_idx] = thresh
            self.video_infos = []
            n_correct_frames = 0
            n_frames_above_thresh = 0
            n_vids_correct = 0
            self.vid_ps_from_img = dict()
            for vidname, items in self.pseudo_scores_dict.items():
                gt_label_seq, pred_label_seq, max_pred_score_seq, pred_scores_all = items[0], items[1], items[2], items[3]
                thresh_seq =  np.array([  cls_thresh[int(pred_label_)]  for pred_label_ in pred_label_seq ])
                mask_frames_above_thresh = np.greater_equal(max_pred_score_seq, thresh_seq)
                n_frames_above_thresh_this_vid = np.sum(mask_frames_above_thresh)
                n_frames_above_thresh += n_frames_above_thresh_this_vid
                n_correct_frames += np.sum(np.logical_and(mask_frames_above_thresh, gt_label_seq == pred_label_seq))
                if n_frames_above_thresh_this_vid > 0:
                    gt_label = int(gt_label_seq[0])
                    # for vidpath_ in self.all_vid_paths:
                    #     if vidname +'.avi' in vidpath_:
                    #         vidpath = vidpath_
                    #         break
                    if self.if_rawframe:
                        vidpath = os.path.join(self.vid_path_prefix_dict[vidname], vidname)
                        self.video_infos.append(
                            {'frame_dir': vidpath, 'total_frames': self.n_frames_dict[vidname], 'label': gt_label})
                    else:
                        vidpath = os.path.join(self.vid_path_prefix_dict[vidname], vidname + self.vid_format)
                        self.video_infos.append({'filename': vidpath, 'label': gt_label})
                    pred_scores_all_filtered = pred_scores_all[mask_frames_above_thresh,  :]  # (n_frames_above_thresh, 12)
                    vid_confidence = np.mean(pred_scores_all_filtered, axis=0)  # (n_class, )
                    vid_ps_label = np.argmax(vid_confidence)
                    self.vid_ps_from_img.update({vidpath: vid_ps_label})
                    if gt_label == vid_ps_label:
                        n_vids_correct += 1
            percent_frames_above_thresh = float(n_frames_above_thresh) / n_frames_total * 100.0
            acc_frame = np.NaN if n_frames_above_thresh == 0 else float(n_correct_frames) / n_frames_above_thresh
            self.num_target_train = len(self.video_infos)  # number of sampled videos
            n_vids_above_thresh = self.num_target_train

            percent_vid_above_thresh = float(n_vids_above_thresh) / len(self.pseudo_scores_dict) * 100.0
            acc_vid_ps = np.NaN if n_vids_above_thresh == 0 else float(n_vids_correct) / n_vids_above_thresh
            logger.debug(
                f'Chosen Class Balanced Frame Thresh  ({ps_thresh_percent * 100.0:.1f}%) : #frames {n_frames_above_thresh}/{percent_frames_above_thresh:.2f}% , acc frame {acc_frame:.3f}'
                f'\t #vids {n_vids_above_thresh}/{percent_vid_above_thresh:.2f}%, acc vid {acc_vid_ps:.3f}')


    def filter_pseudo_vid(self, ps_thresh_percent, epoch):
        """
        filter pseudo labels to keep a specific percent of samples with the highest confidence scores
        # todo update  self.video_infos
        #  todo the complete list of videos is stored in self.pseudo_scores_dict
        :param epoch:
        :return:
        """
        epoch_str = epoch if isinstance(epoch, str) else str( epoch+1)

        logger = utils.logger  # global variable in utils.py
        # check quality of pseudo labels

        n_correct_total = 0
        for vid_path, items in self.pseudo_scores_dict.items():
            # vidname : ( gt label, pred label,  max_pred_score, predicted scores for all classes )
            gt_label, pred_label, max_pred_score = items[0], items[1], items[2]
            if gt_label == pred_label:
                n_correct_total += 1
        acc_total = float(n_correct_total) / len(self.pseudo_scores_dict)
        logger.debug(f'Epoch {epoch_str} Total : #vids {len(self.pseudo_scores_dict)}/100% , acc {acc_total:.3f}')


        confidence_list = [items[2] for vidname, items in self.pseudo_scores_dict.items()]
        confidence_list = sorted(confidence_list, reverse=True)
        for ps_thresh_percent_ in np.arange(0.9, 0, -0.1):
            pos_ = min ( len(confidence_list) -1,  int( len(confidence_list ) * float(ps_thresh_percent_) ) )
            thresh =  confidence_list[ pos_ ]

            n_correct = 0
            n_vids_above_thresh = 0

            for vid_path, items in self.pseudo_scores_dict.items():
                gt_label, pred_label, max_pred_score = items[0], items[1], items[2]
                if max_pred_score >= thresh:
                    n_vids_above_thresh += 1
                    if gt_label == pred_label:
                        n_correct += 1

            percent_vids_above_thresh = float(n_vids_above_thresh) / len(self.pseudo_scores_dict) * 100.0
            acc = np.NaN if n_vids_above_thresh == 0 else float(n_correct) / n_vids_above_thresh
            logger.debug( f'Epoch {epoch_str} Thresh {thresh:.2f} ({ps_thresh_percent_ * 100.0:.1f}%) : #vids {n_vids_above_thresh}/{percent_vids_above_thresh:.2f}% , acc {acc:.3f}'  )


        # get the confidence threshold
        # confidence_list = [  items[2]  for vidname, items in self.pseudo_scores_dict.items() ]
        # confidence_list = sorted(confidence_list, reverse=True)

        pos_ = min(len(confidence_list) - 1, int(len(confidence_list) * float(ps_thresh_percent)))
        ps_thresh =  confidence_list[pos_]


        # filter the target training samples according to pseudo label confidence thresholding
        # target_train_list_filtered = []

        # todo update  self.video_infos
        #  the complete list of videos is stored in self.pseudo_scores_dict
        self.video_infos = []
        n_correct = 0
        for vid_path, items in self.pseudo_scores_dict.items():

            ps_confidence, pred_label, gt_label = items[2], items[1], items[0]
            if ps_confidence >= ps_thresh:
                # if self.data_dir not in vid_path:
                #     vid_path = vid_path.replace( self.data_dir,  )
                if self.if_rawframe:
                    vidname = vid_path.split('/')[-1].split('.')[0]  # todo here vid_path is without file extension
                    self.video_infos.append( {  'frame_dir': vid_path, 'total_frames': self.n_frames_dict[vidname], 'label': gt_label } )
                else:
                    # todo here vid_path is with file extension
                    self.video_infos.append( { 'filename': vid_path,  'label': gt_label } )
                if pred_label == int(gt_label):
                    n_correct += 1
        self.num_target_train = len(self.video_infos)
        acc = np.NaN if self.num_target_train == 0 else float(n_correct) / self.num_target_train
        logger.debug(
            f'pseudo confidence threshold {ps_thresh:.2f} ({ps_thresh_percent * 100.0:.1f}%) : {self.num_target_train} / { len(self.pseudo_scores_dict) } selected, acc {acc:.3f} ')



    def load_annotations(self):
        if self.if_rawframe:
            """Load annotation file to get video information."""
            if self.ann_file.endswith('.json'):
                return self.load_json_annotations()
            video_infos = []
            with open(self.ann_file, 'r') as fin:
                for line in fin:
                    line_split = line.strip().split()
                    video_info = {}
                    idx = 0
                    # idx for frame_dir
                    frame_dir = line_split[idx]
                    if self.data_prefix is not None:
                        frame_dir = osp.join(self.data_prefix, frame_dir)
                    video_info['frame_dir'] = frame_dir
                    idx += 1
                    if self.with_offset:
                        # idx for offset and total_frames
                        video_info['offset'] = int(line_split[idx])
                        video_info['total_frames'] = int(line_split[idx + 1])
                        idx += 2
                    else:
                        # idx for total_frames
                        video_info['total_frames'] = int(line_split[idx])
                        idx += 1
                    # idx for label[s]
                    label = [int(x) for x in line_split[idx:]]
                    assert label, f'missing label in line: {line}'
                    if self.multi_class:
                        assert self.num_classes is not None
                        video_info['label'] = label
                    else:
                        assert len(label) == 1
                        video_info['label'] = label[0]
                    video_infos.append(video_info)

            return video_infos
        else:
            """Load annotation file to get video information."""
            if self.ann_file.endswith('.json'):
                return self.load_json_annotations()

            video_infos = []
            with open(self.ann_file, 'r') as fin:
                for line in fin:
                    line_split = line.strip().split()
                    if self.multi_class:
                        assert self.num_classes is not None
                        filename, label = line_split[0], line_split[1:]
                        label = list(map(int, label))
                    else:
                        filename, label = line_split
                        label = int(label)
                    if self.data_prefix is not None:
                        filename = osp.join(self.data_prefix, filename)
                    video_infos.append(dict(filename=filename, label=label))
            return video_infos


    def prepare_train_frames(self, idx):
        if self.if_rawframe:
            """Prepare the frames for training given the index."""
            results = copy.deepcopy(self.video_infos[idx])
            results['filename_tmpl'] = self.filename_tmpl
            results['modality'] = self.modality
            results['start_index'] = self.start_index

            # prepare tensor in getitem
            if self.multi_class:
                onehot = torch.zeros(self.num_classes)
                onehot[results['label']] = 1.
                results['label'] = onehot

            return self.pipeline(results)
        else:
            """Prepare the frames for training given the index."""
            results = copy.deepcopy(self.video_infos[idx])
            results['modality'] = self.modality
            results['start_index'] = self.start_index

            # prepare tensor in getitem
            # If HVU, type(results['label']) is dict
            if self.multi_class and isinstance(results['label'], list):
                onehot = torch.zeros(self.num_classes)
                onehot[results['label']] = 1.
                results['label'] = onehot

            return self.pipeline(results)

    def prepare_test_frames(self, idx):
        if self.if_rawframe:
            """Prepare the frames for training given the index."""
            results = copy.deepcopy(self.video_infos[idx])
            results['filename_tmpl'] = self.filename_tmpl
            results['modality'] = self.modality
            results['start_index'] = self.start_index

            # prepare tensor in getitem
            if self.multi_class:
                onehot = torch.zeros(self.num_classes)
                onehot[results['label']] = 1.
                results['label'] = onehot

            return self.pipeline(results)
        else:
            """Prepare the frames for testing given the index."""
            results = copy.deepcopy(self.video_infos[idx])
            results['modality'] = self.modality
            results['start_index'] = self.start_index

            # prepare tensor in getitem
            # If HVU, type(results['label']) is dict
            if self.multi_class and isinstance(results['label'], list):
                onehot = torch.zeros(self.num_classes)
                onehot[results['label']] = 1.
                results['label'] = onehot

            return self.pipeline(results)

    def __getitem__(self, idx):
        """Get the sample for either training or testing given index."""
        if self.test_mode:
            # data_dict = self.prepare_test_frames(idx)
            return self.prepare_test_frames(idx)
        data_dict = self.prepare_train_frames(idx) #  here the sample idx is the idx in self.video_infos
        data_dict.update({ 'vid_idx': torch.tensor( [idx])})


        vid_path =  self.video_infos[idx]['frame_dir'] if self.if_rawframe else  self.video_infos[idx]['filename']

        if self.w_pseudo == 'vid': # todo here idx is the index in self.video_infos,  for target train dataset,  self.video_infos might only be a downsampled list
            data_dict.update({'ps_label' : self.pseudo_scores_dict[vid_path][1] })
        elif self.w_pseudo == 'img':
            data_dict.update({'ps_label': self.vid_ps_from_img[vid_path] })

        # return self.prepare_train_frames(idx)
        return data_dict