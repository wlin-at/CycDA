
import os.path as osp

import os
import numpy as np
from utils_img_model import get_class_dict
# from ..utils_img_model import


def read_ucf_category_group(category_group_list=None):
    groupid_to_class = dict()
    class_to_groupid = dict()
    for line in open(category_group_list):
        class_name, category_group = line.strip('\n').split(' ')
        category_group = int(category_group)
        groupid_to_class.update({category_group: class_name})
        class_to_groupid.update({class_name: category_group})
    return groupid_to_class, class_to_groupid


def frame_thresholding(pseudo_scores_dict = None,  thresh = None, n_frames_total = None, ps_thresh_percent_ = None, ):
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
    print(
        f'Frame Thresh {thresh:.2f} ({ps_thresh_percent_ * 100.0:.1f}%) : #frames {n_frames_above_thresh}/{percent_frames_above_thresh:.2f}% , acc frame {acc_frame:.3f}'
        f'\t #vids {n_vids_above_thresh}/{percent_vid_above_thresh:.2f}%, acc vid {acc_vid_ps:.3f}')

def process_img_ps( img_ps_dict, ps_thresh_percent = None,  id_to_class_dict = None ):
    n_correct_total = 0
    n_correct_class_total = np.zeros((n_class,))
    n_samples_class_total = np.zeros((n_class,))
    acc_class_total = np.zeros((n_class,))
    for vidname, items in img_ps_dict.items():
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

        print(
            f'class {class_idx} {id_to_class_dict[class_idx]}: #frames {n_samples_class_total[class_idx]}, acc {acc_class_total[class_idx]:.3f}')
    print('Total:')
    print(f'Total : #frames {n_frames_total}/100% , acc {acc_total:.3f}')



    vid_max_score_list = []
    for vidname, items in img_ps_dict.items():
        vid_max_score_list.append(np.max(items[2]))
    vid_max_score_list = sorted(vid_max_score_list)
    n_vids = len(vid_max_score_list)
    thresh_list = [0.0] + list(
        (np.array(vid_max_score_list[: n_vids - 1]) + np.array(vid_max_score_list[1: n_vids])) / 2.0)
    if False:
        for ps_thresh_percent_ in np.arange(1.0, 0, -0.1):
            n_vids_remain = int(n_vids * ps_thresh_percent_)
            thresh = thresh_list[n_vids - n_vids_remain]
            frame_thresholding(pseudo_scores_dict= img_ps_dict,  thresh=thresh,
                               n_frames_total=n_frames_total, ps_thresh_percent_=ps_thresh_percent_)

    n_vids_remain = int(n_vids * ps_thresh_percent)
    ps_thresh = thresh_list[n_vids - n_vids_remain]


    n_correct_frames = 0
    n_frames_above_thresh = 0
    n_vid_correct = 0
    n_vids_above_thresh = 0

    n_correct_vid_class_total = np.zeros((n_class,))
    n_samples_vid_class_total = np.zeros((n_class, ))
    acc_vid_class_total = np.zeros((n_class, ))
    n_vids_above_thresh_class = np.zeros((n_class, ))
    n_vids_total_class = np.zeros((n_class, ))

    for vidname, items in img_ps_dict.items():
        gt_label_seq, pred_label_seq, max_pred_score_seq, pred_scores_all = items[0], items[1], items[2], items[3]
        mask_frames_above_thresh = max_pred_score_seq >= ps_thresh
        n_frames_above_thresh_this_vid = np.sum(mask_frames_above_thresh)
        n_frames_above_thresh += n_frames_above_thresh_this_vid
        n_correct_frames += np.sum(np.logical_and(mask_frames_above_thresh, gt_label_seq == pred_label_seq))

        n_vids_total_class[ int(gt_label_seq[0])] += 1
        if n_frames_above_thresh_this_vid > 0:
            # gt_label = int(gt_label_seq)
            n_vids_above_thresh += 1

            gt_label = int(gt_label_seq[0])
            n_samples_vid_class_total[gt_label] += 1
            n_vids_above_thresh_class[gt_label] += 1

            pred_scores_all_filtered = pred_scores_all[mask_frames_above_thresh, :]  # (n_frames_above_thresh, 12)
            vid_confidence = np.mean(pred_scores_all_filtered, axis=0)  # (n_class, )
            vid_ps_label = np.argmax(vid_confidence)
            if gt_label == vid_ps_label:
                n_vid_correct += 1
                n_correct_vid_class_total[gt_label] += 1



    for class_idx in range(n_class):
        acc_vid_class_total[class_idx] = float(n_correct_vid_class_total[class_idx]) / n_samples_vid_class_total[class_idx]
        n_vids_above_thresh_class_percent = float(n_vids_above_thresh_class[class_idx]) / n_vids_total_class[class_idx]
        print( f'class {class_idx} {id_to_class_dict[class_idx]} #vids {n_samples_vid_class_total[class_idx]}/{ n_vids_above_thresh_class_percent*100.0:.1f}%, acc vid {acc_vid_class_total[class_idx]:.3f}' )

    percent_vid_above_thresh = float(n_vids_above_thresh) / len(img_ps_dict) * 100.0
    acc_vid_ps = np.NaN if n_vids_above_thresh == 0 else float(n_vid_correct) / n_vids_above_thresh
    print(f'Total all all #vids {n_vids_above_thresh}/{percent_vid_above_thresh:.3f}%, acc vid {acc_vid_ps:.3f}')

    return acc_vid_class_total


def process_vid_ps( vid_ps_dict , ps_thresh_percent, id_to_class_dict = None ):


    # vid_ps_score_dict = dict()
    # confidence_list = []

    confidence_list = [items[2] for _, items in vid_ps_dict.items()]
    confidence_list = sorted(confidence_list, reverse=True)
    # for ps_thresh_percent_ in np.arange(0.9, 0, -0.1):
    #     pos_ = min ( len(confidence_list) -1,  int( len(confidence_list ) * float(ps_thresh_percent_) ) )
    #     thresh =  confidence_list[ pos_ ]
    pos_ = min(len(confidence_list) - 1, int(len(confidence_list) * float(ps_thresh_percent)))
    ps_thresh = confidence_list[pos_]
    n_correct_vid_total = 0
    n_correct_vid_class = np.zeros((n_class, ))
    n_vid_above_thresh_class = np.zeros((n_class, ))
    n_vid_total_class = np.zeros((n_class, ))

    n_vids_above_thresh = 0

    acc_vid_class = np.zeros((n_class, ))


    for vidpath, items in vid_ps_dict.items():
        ps_confidence, pred_label, gt_label = items[2], items[1], items[0]
        n_vid_total_class[int(gt_label)] += 1
        if ps_confidence >= ps_thresh:
            n_vids_above_thresh +=1
            n_vid_above_thresh_class[gt_label] += 1
            if pred_label == int(gt_label):
                n_correct_vid_total += 1
                n_correct_vid_class[gt_label] += 1
    for class_id in range(n_class):
        percent_vid_above_thresh_class = float(n_vid_above_thresh_class[class_id]) / n_vid_total_class[class_id]
        acc_vid_class[class_id] = float(n_correct_vid_class[class_id]) / n_vid_total_class[class_id]
        print(f'class {class_id} {id_to_class_dict[class_id]} #vids {n_vid_above_thresh_class[class_id]}/{percent_vid_above_thresh_class * 100.0:.1f}%, acc vid {acc_vid_class[class_id]:.3f}')

    percent_vid_above_thresh = float(n_vids_above_thresh) / len(vid_ps_dict)
    acc_vid_ps = float(n_correct_vid_total) / len(vid_ps_dict)
    print(f'Total all all #vids {n_vids_above_thresh}/{percent_vid_above_thresh * 100.0:.3f}%, acc vid {acc_vid_ps:.3f}')

    return acc_vid_class

if __name__ == '__main__':
    main_dir = '/media/data_8T/UCF-HMDB/ablation_incepi3d/img2vid/bu2ucf/split1'


    step1_img_ps_file = osp.join(main_dir, 'step1_grl_tsn_img_model_128d', '20220209_121259_epoch20_frame_ps.npy'  )  # 0.42
    # imgage pseudo label   {vidname:  gt_label_seq,  pred_label_seq,  max_pred_score_seq,  all_class_scores (n_frames, n_class ) }

    step1_img_ps = np.load(step1_img_ps_file, allow_pickle=True).item()

    step2_vid_ps_file = osp.join(main_dir, 'step1_grl_tsn_img_model_128d/step2_vid_model', 'epoch_20_test1clip_vid_ps.npy' )  # 0.553
    #   {vidname:   ( gt label, pred label,  max_pred_score, predicted scores for all classes )  }

    step3_img_ps_file = osp.join(main_dir, 'step1_grl_tsn_img_model_128d/step2_vid_model/step3_contrast_img_model/v1', '20220210_101236_epoch30_frame_ps.npy')  #  0.558
    step4_vid_ps_file = osp.join(main_dir, 'step1_grl_tsn_img_model_128d/step2_vid_model/step3_contrast_img_model/v1/step4_vid_model', 'epoch_20_test1clip_vid_ps.npy' )  #  0.609

    n_class = list(step1_img_ps.values())[0][3].shape[-1]  # all_class_scores has shape (m_frames, n_class )

    ps_thresh_percent = 1.0
    class_to_id_dict, id_to_class_dict = get_class_dict(osp.join('/media/data_8T/UCF-HMDB/UCF-HMDB_all', 'UCF_mapping.txt'))

    def group_analysis(acc_vid_class):
        groupid_to_class, class_to_groupid = read_ucf_category_group( osp.join(  '/media/data_8T/UCF-HMDB/UCF-HMDB_all', 'UCF_category_group.txt'))
        group_dict = dict()
        for groupid in range(0, 5):
            group_dict.update({groupid: []})
        for class_name, group_id_ in class_to_groupid.items():
            group_dict[group_id_].append( class_to_id_dict[class_name] )
        for group_id in group_dict:
            group_average = np.average(acc_vid_class[ group_dict[group_id]])
            print(f'group {group_id} average {group_average:.3f}')

    print('\n')
    print('Processing step1 img ps  ...')

    acc_vid_class_step1 = process_img_ps(img_ps_dict = step1_img_ps, ps_thresh_percent=  ps_thresh_percent, id_to_class_dict=id_to_class_dict)
    print('Processing step1 img ps  ...')
    group_analysis(acc_vid_class_step1)

    print('\n')
    print('Processing step2 vid ps ...')
    step2_vid_ps = np.load(step2_vid_ps_file, allow_pickle=True).item()
    acc_vid_class_step2 = process_vid_ps(vid_ps_dict = step2_vid_ps, ps_thresh_percent = ps_thresh_percent, id_to_class_dict =id_to_class_dict)
    print('Processing step2 vid ps ...')
    group_analysis(acc_vid_class_step2)

    print('\n')
    print('Processing step3 img ps  ...')
    step3_img_ps = np.load(step3_img_ps_file, allow_pickle=True).item()
    acc_vid_class_step3 = process_img_ps(img_ps_dict=step3_img_ps, ps_thresh_percent=ps_thresh_percent, id_to_class_dict=id_to_class_dict)
    print('Processing step3 ...')
    group_analysis(acc_vid_class_step3)

    print('\n')
    print('Processing step4 vid ps ...')
    step4_vid_ps = np.load(step4_vid_ps_file, allow_pickle=True).item()
    acc_vid_class_step4 = process_vid_ps(vid_ps_dict=step4_vid_ps, ps_thresh_percent=ps_thresh_percent, id_to_class_dict=id_to_class_dict)
    print('Processing step4 ...')
    group_analysis(acc_vid_class_step4)

    # print('\n')
    # print('Processing step3 self train vid ps ...')
    # step3_self_train_ps_file = osp.join(main_dir, 'step1_grl_tsn_img_model_64d/v4/step2_vid_model/20epochs_0.7/step3_vid_model', 'epoch_20_test3clip_vid_ps.npy'  )
    # step3_self_train_vid_ps = np.load(step3_self_train_ps_file, allow_pickle=True).item()
    # process_vid_ps(vid_ps_dict=step3_self_train_vid_ps, ps_thresh_percent=ps_thresh_percent, id_to_class_dict=e2h_id_to_class_dict)
    #
    #
    # print('\n')
    # print('Processing step 4 self train vid ps ...')
    # step4_self_train_ps_file = osp.join(main_dir, 'step1_grl_tsn_img_model_64d/v4/step2_vid_model/20epochs_0.7/step3_vid_model', 'epoch_20_test3clip_vid_ps.npy'  )
    # step3_self_train_vid_ps = np.load(step3_self_train_ps_file, allow_pickle=True).item()
    # process_vid_ps(vid_ps_dict=step3_self_train_vid_ps, ps_thresh_percent=ps_thresh_percent, id_to_class_dict=e2h_id_to_class_dict)


    t =1

