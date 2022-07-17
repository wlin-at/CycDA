
#  fuse the video-level pseudo label from step 2 (video model) and frame-level pseudo label from step 3 (image model)

# in step 2, the video-level pseudo label is   pred_scores_dict
#  #    vidname : ( gt label,     scalar
#                   pred label,   scalar
#                   max_pred_score,     scalar
#                   predicted scores for all classes,      (n_class, )
#                   n_frames  )

# in step 3,  the frame-level pseudo label  is  pred_scores_dict
# # each video has a tuple of        gt_label (n_frames, ),
#                                    pred_label (n_frames, ) ,
#                                    max_pred_score (n_frames, ),
#                                    all_class_scores  (n_frames, n_class )

import numpy as np
import copy as cp
def filter_pseudo_vid(pseudo_scores_dict, ps_thresh_percent,):
    """
    filter pseudo labels to keep a specific percent of samples with the highest confidence scores
    # todo update  self.video_infos
    #  todo the complete list of videos is stored in self.pseudo_scores_dict
    :param epoch:
    :return:
    """
    n_correct_total = 0
    for vid_path, items in pseudo_scores_dict.items():
        # vidname : ( gt label, pred label,  max_pred_score, predicted scores for all classes )
        gt_label, pred_label, max_pred_score = items[0], items[1], items[2]
        if gt_label == pred_label:
            n_correct_total += 1
    acc_total = float(n_correct_total) / len(pseudo_scores_dict)
    print(f'Total : #vids {len(pseudo_scores_dict)}/100% , acc {acc_total:.3f}')

    confidence_list = [items[2] for vidname, items in pseudo_scores_dict.items()]
    confidence_list = sorted(confidence_list, reverse=True)
    for ps_thresh_percent_ in np.arange(0.9, 0, -0.1):
        pos_ = min(len(confidence_list) - 1, int(len(confidence_list) * float(ps_thresh_percent_)))
        thresh = confidence_list[pos_]

        n_correct = 0
        n_vids_above_thresh = 0

        for vid_path, items in pseudo_scores_dict.items():
            gt_label, pred_label, max_pred_score = items[0], items[1], items[2]
            if max_pred_score >= thresh:
                n_vids_above_thresh += 1
                if gt_label == pred_label:
                    n_correct += 1

        percent_vids_above_thresh = float(n_vids_above_thresh) / len(pseudo_scores_dict) * 100.0
        acc = np.NaN if n_vids_above_thresh == 0 else float(n_correct) / n_vids_above_thresh
        print(
            f'Thresh {thresh:.2f} ({ps_thresh_percent_ * 100.0:.1f}%) : #vids {n_vids_above_thresh}/{percent_vids_above_thresh:.2f}% , acc {acc:.3f}')

    # get the confidence threshold
    # confidence_list = [  items[2]  for vidname, items in self.pseudo_scores_dict.items() ]
    # confidence_list = sorted(confidence_list, reverse=True)

    pos_ = min(len(confidence_list) - 1, int(len(confidence_list) * float(ps_thresh_percent)))
    ps_thresh = confidence_list[pos_]

    # filter the target training samples according to pseudo label confidence thresholding
    # target_train_list_filtered = []

    # todo update  self.video_infos
    #  the complete list of videos is stored in self.pseudo_scores_dict
    video_infos = []
    n_correct = 0

    new_pseudo_scores_dict = dict()
    list_vid_path_filtered = []
    for vid_path, items in pseudo_scores_dict.items():
        ps_confidence, pred_label, gt_label = items[2], items[1], items[0]
        if ps_confidence >= ps_thresh:
            # if self.data_dir not in vid_path:
            #     vid_path = vid_path.replace( self.data_dir,  )

            video_infos.append({'filename': vid_path, 'label': gt_label})
            new_pseudo_scores_dict.update({vid_path: items })

            if pred_label == int(gt_label):
                n_correct += 1
        else:
            new_pseudo_scores_dict.update({vid_path: ( items[0], None, None, None, items[4] ) })
            list_vid_path_filtered.append( vid_path)
    num_target_train = len(video_infos)
    acc = np.NaN if num_target_train == 0 else float(n_correct) / num_target_train
    print(
        f'pseudo confidence threshold {ps_thresh:.2f} ({ps_thresh_percent * 100.0:.1f}%) : {num_target_train} / {len(pseudo_scores_dict)} selected, acc {acc:.3f} ')
    return new_pseudo_scores_dict, list_vid_path_filtered




def fuse_pseudo_labels(ps_label_vid_step2_new, ps_label_frame_step3,  frame_thresh_within_vid,  list_vid_path_filtered  ):
    for vid_path in list_vid_path_filtered:
        vid_name = vid_path.split('/')[-1].split('.')[0]
        items = ps_label_frame_step3[vid_name]
        gt_label_seq, pred_label_seq, max_pred_score_seq, pred_scores_all = items[0], items[1], items[2], items[3]
        confidence_list = sorted( list(max_pred_score_seq), reverse=True )
        pos_ = min( len(confidence_list)-1, int(len(confidence_list) * float(frame_thresh_within_vid)) )
        ps_thresh = confidence_list[pos_]

        mask_frames_above_thresh = max_pred_score_seq >= ps_thresh
        pred_scores_all_filtered = pred_scores_all[mask_frames_above_thresh, :]  # (n_frames_above_thresh, 12)
        vid_confidence = np.mean(pred_scores_all_filtered, axis=0)  # (n_class, )
        vid_ps_label = np.argmax(vid_confidence)
        vid_confidence_max = vid_confidence[vid_ps_label]

        # vidname : ( gt label, pred label,  max_pred_score, predicted scores for all classes )
        data = list( ps_label_vid_step2_new[vid_path] )
        data[1], data[2], data[3] =  vid_ps_label, vid_confidence_max, vid_confidence
        ps_label_vid_step2_new[vid_path] = tuple(data)
    return ps_label_vid_step2_new


def frame_to_vid_ps( ps_label_frame_step3,  ps_thresh_percent, all_vid_paths, ps_label_vid_step2  ):
    vid_max_score_list = []
    for vidname, items in ps_label_frame_step3.items():
        vid_max_score_list.append(np.max(items[2]))  # list of the maximum frame-level scores in each video
    vid_max_score_list = sorted(vid_max_score_list)  # sort in ascending order
    n_vids = len(vid_max_score_list)
    # todo thresh_list contains thresholds corresponding to  N, N-1, N-2, ..., 1 vid
    thresh_list = [0.0] + list(
        (np.array(vid_max_score_list[: n_vids - 1]) + np.array(vid_max_score_list[1: n_vids])) / 2.0)
    # for ps_thresh_percent_ in np.arange(1.0, 0, -0.1):
    #     n_vids_remain = int(n_vids * ps_thresh_percent_)
    #     thresh = thresh_list[n_vids - n_vids_remain]
    #     frame_thresholding(pseudo_scores_dict=self.pseudo_scores_dict, logger=logger, thresh=thresh,
    #                        n_frames_total=n_frames_total, ps_thresh_percent_=ps_thresh_percent_)

    n_vids_remain = int(n_vids * ps_thresh_percent)
    ps_thresh = thresh_list[n_vids - n_vids_remain]

    vid_counter = 0
    for vidname, items in ps_label_frame_step3.items():
        gt_label_seq, pred_label_seq, max_pred_score_seq, pred_scores_all = items[0], items[1], items[2], items[3]
        mask_frames_above_thresh = max_pred_score_seq >= ps_thresh
        n_frames_above_thresh_this_vid = np.sum(mask_frames_above_thresh)
        # n_frames_above_thresh += n_frames_above_thresh_this_vid
        # n_correct_frames += np.sum(np.logical_and(mask_frames_above_thresh, gt_label_seq == pred_label_seq))


        if n_frames_above_thresh_this_vid > 0:
            vid_counter +=1
            gt_label = int(gt_label_seq[0])
            pred_scores_all_filtered = pred_scores_all[mask_frames_above_thresh, :]  # (n_frames_above_thresh, 12)
            vid_confidence = np.mean(pred_scores_all_filtered, axis=0)  # (n_class, )
            vid_ps_label = np.argmax(vid_confidence)
            vid_confidence_max = vid_confidence[vid_ps_label]
            # todo avoid case of  vid_1  and vid_10
            for vidpath_ in all_vid_paths:
                if vidname + '.avi' in vidpath_:
                    vidpath = vidpath_
                    break
            data = list(ps_label_vid_step2[vidpath])
            # vidname : ( gt label, pred label,  max_pred_score, predicted scores for all classes, n_frames  )
            data[1], data[2], data[3] = vid_ps_label, vid_confidence_max, vid_confidence
            ps_label_vid_step2[vidpath] = tuple(data)
    print( f'{vid_counter} videos updated!')
    return ps_label_vid_step2




if __name__ == '__main__':
    ps_label_vid_step2 = '/media/data_8T/UCF-HMDB/ablation_incepi3d/img2vid/e2h/split1/step1_grl_tsn_img_model_64d/v4/step2_vid_model/0.7/epoch_10_test3clip_vid_ps.npy'
    ps_label_vid_step2 = np.load(ps_label_vid_step2, allow_pickle=True).item()
    all_vid_paths = list(ps_label_vid_step2.keys())





    ps_thresh_percent_vid = 0.8
    filter_pseudo_vid(ps_label_vid_step2, ps_thresh_percent=ps_thresh_percent_vid)

    ps_label_frame_step3 = '/media/data_8T/UCF-HMDB/ablation_incepi3d/img2vid/e2h/split1/step1_grl_tsn_img_model_64d/v4/step2_vid_model/0.7/step3_s_and_t_img_model/v5/20220207_205623_frame_ps.npy'
    ps_label_frame_step3 = np.load(ps_label_frame_step3, allow_pickle=True).item()
    all_vid_names = list(ps_label_frame_step3.keys())


    # frame_thresh_within_vid = 1.0

    # ps_label_vid_step2_new = fuse_pseudo_labels(ps_label_vid_step2_new, ps_label_frame_step3, frame_thresh_within_vid, list_vid_path_filtered)


    # ps_thresh_percent_frame = 0.9
    for ps_thresh_percent_frame in np.arange(0.9, 0, -0.1 ):
        print( f'ps_thresh_percent_frame {ps_thresh_percent_frame}')
        ps_label_vid_step2_new = frame_to_vid_ps( ps_label_frame_step3,  ps_thresh_percent_frame, all_vid_paths, ps_label_vid_step2  )
        filter_pseudo_vid(ps_label_vid_step2_new, ps_thresh_percent=ps_thresh_percent_vid)





