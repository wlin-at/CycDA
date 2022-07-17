
"""
Clean frame-level pseudo labels:
thresholding
temporal averaging ,  all frames in the same video should share the same pseudo label


"""

import numpy as np

frame_ps_file = '/media/data_8T/UCF-HMDB/ucf_hmdb_img_train/ucf_train_hmdb_val_resnet/20220103_160039_frame_ps.npy'
pseudo_scores_dict = np.load(frame_ps_file, allow_pickle=True).item()
ps_filter_by = 'frame'

n_class = list(pseudo_scores_dict.values())[0][3].shape[-1]  #  all_class_scores has shape (m_frames, n_class )

######################################## compute frame-level accuracy for each class
n_correct_total = 0
n_correct_class_total = np.zeros((n_class,))
n_samples_class_total = np.zeros((n_class,))
acc_class_total = np.zeros((n_class,))
for vidname, items in pseudo_scores_dict.items():
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
    print(f'class {class_idx}: #frames {n_samples_class_total[class_idx]}, acc {acc_class_total[class_idx]:.3f}')
print('Total:')
print(f'Total : #frames {n_frames_total}/100% , acc {acc_total:.3f}')

if ps_filter_by == 'frame':
    #########################  Compute accuracy of frames above thresholds according to percent 0.9 ~ 0.1
    confidence_list = []
    for vidname, items in pseudo_scores_dict.items():
        confidence_list.extend(items[2].tolist())
    # confidence_list = [ np.mean(items[2])  for vidname, items in pred_scores_dict.items()]
    confidence_list = sorted(confidence_list, reverse=True)
    for ps_thresh_percent_ in np.arange(0.9, 0, -0.1):
        pos_ = min(len(confidence_list) - 1, int(len(confidence_list) * float(ps_thresh_percent_)))
        thresh = confidence_list[pos_]

        n_correct_frames = 0
        n_frames_above_thresh = 0

        n_vid_correct = 0
        n_vids_above_thresh = 0

        n_frames_kept_total = 0
        n_correct_frames_after_update = 0

        for vidname, items in pseudo_scores_dict.items():
            gt_label_seq, pred_label_seq, max_pred_score_seq, pred_scores_all = items[0], items[1], items[2], items[3]
            n_frames = len(gt_label_seq)
            mask_frames_above_thresh = max_pred_score_seq >= thresh
            n_frames_above_thresh_this_vid = np.sum(mask_frames_above_thresh)
            n_frames_above_thresh += n_frames_above_thresh_this_vid
            # the frames that are above threshold and are correct
            n_correct_frames += np.sum(np.logical_and(mask_frames_above_thresh, gt_label_seq == pred_label_seq))

            # drive the video-level confidence score as the average of confidence scores of frames above threshold
            if n_frames_above_thresh_this_vid > 0:
                n_vids_above_thresh += 1
                n_frames_kept_total += n_frames
                pred_scores_all_filtered = pred_scores_all[mask_frames_above_thresh, :]  # (n_frames_above_thresh, 12)
                vid_confidence = np.mean(pred_scores_all_filtered, axis=0)  # (n_class, )
                vid_ps_label = np.argmax(vid_confidence)
                gt_label = gt_label_seq[0]
                if gt_label == vid_ps_label:
                    n_vid_correct += 1
                    # todo  after expanding the pseudo labels, all the frames in the video will be correct
                    n_correct_frames_after_update += n_frames

        percent_frames_above_thresh = float(n_frames_above_thresh) / n_frames_total * 100.0
        acc_frame = np.NaN if n_frames_above_thresh == 0 else float(n_correct_frames) / n_frames_above_thresh
        percent_vid_above_thresh = float(n_vids_above_thresh) / len(pseudo_scores_dict) * 100.0
        acc_vid_ps = np.NaN if n_vids_above_thresh == 0 else float(n_vid_correct) / n_vids_above_thresh

        acc_frame_after_update = np.NaN if n_frames_above_thresh == 0 else float(n_correct_frames_after_update) / n_frames_kept_total * 100.0
        print(
            f'Frame Thresh {thresh:.2f} ({ps_thresh_percent_ * 100.0:.1f}%) : #frames {n_frames_above_thresh}/{percent_frames_above_thresh:.2f}% , acc frame {acc_frame:.3f}'
            f'\t #vids {n_vids_above_thresh}/{percent_vid_above_thresh:.2f}%, acc vid {acc_vid_ps:.3f}'
            f'\t acc frame after update #correct/#frames={n_correct_frames_after_update}/{n_frames_kept_total}={acc_frame_after_update:.3f}')

pass




    # # todo update  self.video_infos , save ps label in self.vid_ps_from_img
    # #  the complete list of videos is stored in self.pseudo_scores_dict
    # pos_ = min(len(confidence_list) - 1, int(len(confidence_list) * float(ps_thresh_percent)))
    # ps_thresh = confidence_list[pos_]
    # video_infos = []
    #
    # n_correct_frames = 0
    # n_frames_above_thresh = 0
    #
    # n_vid_correct = 0
    # # n_vids_above_thresh = 0
    #
    # vid_ps_from_img = dict()
    #
    # for vidname, items in pseudo_scores_dict.items():
    #     gt_label_seq, pred_label_seq, max_pred_score_seq, pred_scores_all = items[0], items[1], items[2], items[3]
    #     mask_frames_above_thresh = max_pred_score_seq >= ps_thresh
    #     n_frames_above_thresh_this_vid = np.sum(mask_frames_above_thresh)
    #     n_frames_above_thresh += n_frames_above_thresh_this_vid
    #     n_correct_frames += np.sum(np.logical_and(mask_frames_above_thresh, gt_label_seq == pred_label_seq))
    #     if n_frames_above_thresh_this_vid > 0:
    #         gt_label = int(gt_label_seq[0])
    #         # if self.data_dir not in vid_path:
    #         #     vid_path = vid_path.replace( self.data_dir,  )
    #         # occurrence = 0
    #         # vidpath_list = []
    #         for vidpath_ in all_vid_paths:
    #             if vidname + '.avi' in vidpath_:
    #                 vidpath = vidpath_
    #                 break
    #                 # vidpath_list.append(vidpath_)
    #                 # occurrence += 1
    #
    #         # if occurrence > 1:
    #         #     print(vidname)
    #         #     print( vidpath_list)
    #         self.video_infos.append({'filename': vidpath, 'label': gt_label})
    #         pred_scores_all_filtered = pred_scores_all[mask_frames_above_thresh, :]  # (n_frames_above_thresh, 12)
    #         vid_confidence = np.mean(pred_scores_all_filtered, axis=0)  # (n_class, )
    #         vid_ps_label = np.argmax(vid_confidence)
    #         self.vid_ps_from_img.update({vidpath: vid_ps_label})
    #         if gt_label == vid_ps_label:
    #             n_vid_correct += 1
    #
    # percent_frames_above_thresh = float(n_frames_above_thresh) / n_frames_total * 100.0
    # acc_frame = np.NaN if n_frames_above_thresh == 0 else float(n_correct_frames) / n_frames_above_thresh
    # self.num_target_train = len(self.video_infos)  # number of sampled videos
    # n_vids_above_thresh = self.num_target_train
    #
    # percent_vid_above_thresh = float(n_vids_above_thresh) / len(self.pseudo_scores_dict) * 100.0
    # acc_vid_ps = np.NaN if n_vids_above_thresh == 0 else float(n_vid_correct) / n_vids_above_thresh
    # logger.debug(
    #     f'Chosen Frame Thresh {ps_thresh:.2f} ({ps_thresh_percent * 100.0:.1f}%) : #frames {n_frames_above_thresh}/{percent_frames_above_thresh:.2f}% , acc frame {acc_frame:.3f}'
    #     f'\t #vids {n_vids_above_thresh}/{percent_vid_above_thresh:.2f}%, acc vid {acc_vid_ps:.3f}')