import numpy as np
import matplotlib.pyplot as plt
import os
import os.path as osp
import time
from mmaction.utils import  get_root_logger
from scipy.special import softmax
import cv2
# # A = torch.tensor( np.arange( 6 )).reshape((2,3,1))
#
#
# A = np.arange( 6 ).reshape((2,3,1))
# print( A [0,:,0])
#
# B = A.reshape( (-1,) + A.shape[2:])
#
# C = np.expand_dims(B, 0)
# C= C.reshape(  (-1, 3, 1),    )
# print(C [0, :, 0])


def save_clip_prediction(outputs, out, dataset, cfg):
    """

    :param outputs:  a list of  array (n_clips, n_class)
    :param out:
    :param dataset:
    :return:
    """
    n_clips, num_class = outputs[0].shape[0], outputs[0].shape[1]
    result_dir, result_filename = os.path.split(out)
    pred_scores_dict = dict()
    max_pred_scores_list = []
    for vid_idx in range(len(outputs)):
        outputs[vid_idx] = softmax(outputs[vid_idx], axis=1)
        pred_label_vec = np.argmax( outputs[vid_idx], axis=1 )
        max_predict_score_vec = np.amax(outputs[vid_idx], axis=1)
        if dataset.if_rawframe:
            vid_filename = dataset.video_infos[vid_idx]['frame_dir']  # todo without file extension
        else:
            vid_filename = dataset.video_infos[vid_idx]['filename']  # todo with file extension
        if cfg.data_dir != cfg.new_data_dir:
            vid_filename = vid_filename.replace(cfg.data_dir,  cfg.new_data_dir )
        vid_label = dataset.video_infos[vid_idx]['label']
        pred_scores_dict.update({vid_filename: (vid_label, pred_label_vec, max_predict_score_vec, outputs[vid_idx])})
    # np.save(out.split('.')[0] + '_clip_ps.npy', pred_scores_dict)
    np.save(osp.join(result_dir, result_filename.split('.')[0] + '_clip_ps.npy'), pred_scores_dict)


    final_results_write = open(os.path.join(result_dir, f'pseudo_gt_target_train_final_results_clip.txt'), 'w+')
    n_correct_clips_total = 0
    n_correct_clips_class_total = np.zeros((num_class,))
    n_vid_samples_class_total = np.zeros((num_class,))
    acc_class_total = np.zeros((num_class,))
    for vid_idx, items in pred_scores_dict.items():
        # vidname : ( gt label, pred label,  max_pred_score, predicted scores for all classes )
        gt_label, pred_label_vec, max_pred_score_vec = items[0], items[1], items[2]
        n_vid_samples_class_total[gt_label] += 1
        n_correct_clips = np.sum( pred_label_vec == gt_label )
        n_correct_clips_total += n_correct_clips
        n_correct_clips_class_total[gt_label] += n_correct_clips
    acc_total = float(n_correct_clips_total) / (len(pred_scores_dict) * n_clips)

    for class_idx in range(num_class):
        acc_class_total[class_idx] = float(n_correct_clips_class_total[class_idx]) / (n_vid_samples_class_total[class_idx] * n_clips)
        final_results_write.write(
            f'class {class_idx + 1}: #clips {n_vid_samples_class_total[class_idx] * n_clips}, acc {acc_class_total[class_idx]:.3f}\n')
    final_results_write.write('Total:\n')
    final_results_write.write(f'Total : #clips {len(pred_scores_dict) * n_clips}/100% , acc {acc_total:.3f}\n')
    # compute accuracy of pseudo labels w.r.t thresholding
    final_results_write.write('\n')


    # todo   the confidence score of a video is the sum of max confidence score of clips in this video
    #   confidence_list is a list of  video-wise confidence score
    confidence_list = [ np.sum(items[2])  for vidname, items in pred_scores_dict.items()]
    confidence_list = sorted(confidence_list, reverse=True)
    for ps_thresh_percent_ in np.arange(0.9, 0, -0.1):
        pos_ = min(len(confidence_list) - 1, int(len(confidence_list) * float(ps_thresh_percent_)))
        thresh = confidence_list[pos_]

        n_correct_clips = 0
        n_clips_above_thresh = 0

        for vid_path, items in pred_scores_dict.items():
            gt_label, pred_label_vec, max_pred_score_vec = items[0], items[1], items[2]

            # todo the confidence of a vid is the sum of max_confidence scores of clips in this video
            if np.sum(max_pred_score_vec) >= thresh:
                n_clips_above_thresh += n_clips
                n_correct_clips +=  np.sum( pred_label_vec == gt_label )
        percent_clips_above_thresh = float(n_clips_above_thresh) / (len(pred_scores_dict) * n_clips )* 100.0
        acc = np.NaN if n_clips_above_thresh == 0 else float(n_correct_clips) / n_clips_above_thresh
        final_results_write.write(
            f'Thresh {thresh:.2f} ({ps_thresh_percent_ * 100.0:.1f}%) : #clips {n_clips_above_thresh}/{percent_clips_above_thresh:.2f}% , acc {acc:.3f}\n')
    final_results_write.write('\n')
    final_results_write.close()

def save_features( outputs, vid_features,  out, dataset, cfg ):
    num_class = len(outputs[0])  # outputs is a list of  a list of (n_class, )
    feat_dim = vid_features[0].shape[-1]  # features is a list of  (1, feat_dim, )
    result_dir, _ = os.path.split(out)
    # target_train_vid_info = np.load(cfg.DA_config.target_train_vid_info, allow_pickle=True).item()
    feat_all = np.empty( (0, feat_dim ))
    # gt_labels = np.empty((0, ))
    gt_labels = list()
    pred_labels_all = list()
    # pred_labels_all = np.empty((0, ))

    for vid_idx in range(len(outputs)):
        outputs[vid_idx] = softmax(outputs[vid_idx], axis=0)  # (n_class)
        pred_label = np.argmax(outputs[vid_idx])
        vid_label = dataset.video_infos[vid_idx]['label']
        feat_all = np.concatenate([ feat_all, vid_features[vid_idx] ], axis=0) # (*, feat_dim)
        gt_labels.append( vid_label )
        # gt_labels = np.concatenate([ gt_labels,  vid_label  ] )
        pred_labels_all.append(  pred_label)
        # pred_labels_all = np.concatenate([ pred_labels_all, pred_label])

    feat_dict = {'feat': feat_all, 'gt': np.array(gt_labels), 'pred_label': np.array(pred_labels_all) }
    np.save(osp.join( result_dir, 'feat_dict.npy'), feat_dict )
    return feat_dict



def save_vid_prediction(outputs, out, dataset, cfg):

    """
    save prediction results,    compute accuracy of pseudo labels, and accuracy of pseudo labels w.r.t thresholding
    :param outputs:    a list of video-level prediction scores (n_class, )
    :param out:   the input given after the flag --out
    :param dataset:
    :return:
    """

    num_class = len(outputs[0])
    result_dir, result_filename = os.path.split( out)
    pred_scores_dict = dict()
    max_pred_score_list = []
    # target_train_vid_info = np.load( '/media/data_8T/UCF-HMDB/datalist_new/ucf_train_vid_info.npy', allow_pickle= True ).item()
    target_train_vid_info = np.load( cfg.DA_config.target_train_vid_info, allow_pickle= True ).item()

    for vid_idx in range(len(outputs)):
        outputs[vid_idx] = softmax( outputs[vid_idx], axis=0)
        pred_label = np.argmax(outputs[vid_idx])
        max_predict_score = outputs[vid_idx][pred_label]
        max_pred_score_list.append(max_predict_score)
        vid_filename = dataset.video_infos[vid_idx]['filename']  # todo  filename is the video path
        # n_frames = int(cv2.VideoCapture(vid_filename).get(cv2.CAP_PROP_FRAME_COUNT))
        vidname = vid_filename.split('/')[-1].split('.')[0]
        n_frames = target_train_vid_info[vidname][0]
        if cfg.data_dir != cfg.new_data_dir:   #  todo new_data_dir is used for the path to be saved in the dictionary
            vid_filename = vid_filename.replace(cfg.data_dir,  cfg.new_data_dir )


        vid_label = dataset.video_infos[vid_idx]['label']
        # vidname : ( gt label, pred label,  max_pred_score, predicted scores for all classes )
        pred_scores_dict.update({vid_filename: (vid_label, pred_label, max_predict_score, outputs[vid_idx], n_frames)})

    # save prediction results
    # np.save(out.split('.')[0] + '_vid_ps.npy', pred_scores_dict)
    file_str = result_filename.split('.')[0]
    np.save(osp.join(result_dir, file_str + '_vid_ps.npy' ), pred_scores_dict)
    # # plt the max pred scores distribution
    # fig = plt.figure()
    # plt.hist(max_pred_score_list, 100)
    # plt.title('max pred scores distr')
    # plt.xlabel('pred score')
    # plt.show()
    # fig.savefig(os.path.join(result_dir, 'max_pred_scores_distr'))



    # compute accuracy of pseudo labels
    final_results_write = open(os.path.join(result_dir, f'{file_str}_pseudo_gt_target_train.txt'), 'w+')
    n_correct_total = 0
    n_correct_class_total = np.zeros((num_class,))
    n_samples_class_total = np.zeros((num_class,))
    acc_class_total = np.zeros((num_class,))

    n_frames_total = 0
    n_correct_frames = 0
    for vid_idx, items in pred_scores_dict.items():
        # vidname : ( gt label, pred label,  max_pred_score, predicted scores for all classes )
        gt_label, pred_label, max_pred_score, n_frames = items[0], items[1], items[2], items[4]
        n_samples_class_total[gt_label] += 1
        n_frames_total += n_frames
        if gt_label == pred_label:
            n_correct_total += 1
            n_correct_class_total[gt_label] += 1
            n_correct_frames += n_frames
    acc_vid_total = float(n_correct_total) / len(pred_scores_dict)
    acc_frame_total = float(n_correct_frames) / n_frames_total
    for class_idx in range(num_class):
        acc_class_total[class_idx] = float(n_correct_class_total[class_idx]) / n_samples_class_total[class_idx]
        final_results_write.write(
            f'class {class_idx + 1}: #vids {n_samples_class_total[class_idx]}, acc {acc_class_total[class_idx]:.3f}\n')
    final_results_write.write('Total:\n')
    final_results_write.write(f'Total : #vids {len(pred_scores_dict)}/100% , acc vid {acc_vid_total:.3f}  #frames {n_frames_total}/100% , acc frame {acc_frame_total:.3f} \n')
    # compute accuracy of pseudo labels w.r.t thresholding
    final_results_write.write('\n')



    confidence_list = [items[2] for vidname, items in pred_scores_dict.items()]
    confidence_list = sorted(confidence_list, reverse=True)
    for ps_thresh_percent_ in np.arange(0.9, 0, -0.1):
        pos_ = min(len(confidence_list) - 1, int(len(confidence_list) * float(ps_thresh_percent_)))
        thresh = confidence_list[pos_]

        n_correct_vids = 0
        n_vids_above_thresh = 0

        n_correct_frames = 0
        n_frames_above_thresh = 0
        n_frames_total = 0
        for vid_path, items in pred_scores_dict.items():
            gt_label, pred_label, max_pred_score, n_frames = items[0], items[1], items[2], items[4]
            n_frames_total += n_frames
            if max_pred_score >= thresh:
                n_vids_above_thresh += 1
                n_frames_above_thresh += n_frames
                if gt_label == pred_label:
                    n_correct_vids += 1
                    n_correct_frames += n_frames

        percent_vids_above_thresh = float(n_vids_above_thresh) / len(pred_scores_dict) * 100.0
        acc_vid = np.NaN if n_vids_above_thresh == 0 else float(n_correct_vids) / n_vids_above_thresh

        percent_frames_above_thresh = float(n_frames_above_thresh ) / n_frames_total * 100.0
        acc_frame = np.NaN if n_frames_above_thresh == 0 else float(n_correct_frames) / n_frames_above_thresh
        final_results_write.write(
            f'Thresh {thresh:.2f} ({ps_thresh_percent_ * 100.0:.1f}%) : #vids {n_vids_above_thresh}/{percent_vids_above_thresh:.2f}% , acc vid {acc_vid:.4f} #frames {n_frames_above_thresh}/{percent_frames_above_thresh:.2f}% , acc frame {acc_frame:.4f} \n')




    # # for thresh in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
    # for thresh in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,12]:
    #     n_correct = 0
    #     n_vids_above_thresh = 0
    #
    #     for vid_idx, items in pred_scores_dict.items():
    #         gt_label, pred_label, max_pred_score = items[0], items[1], items[2]
    #         if max_pred_score >= thresh:
    #             n_vids_above_thresh += 1
    #             if gt_label == pred_label:
    #                 n_correct += 1
    #
    #     percent_vids_above_thresh = float(n_vids_above_thresh) / len(pred_scores_dict) * 100.0
    #     acc = np.NaN if n_vids_above_thresh == 0 else float(n_correct) / n_vids_above_thresh
    #     final_results_write.write(
    #         f'Thresh {thresh} : #vids {n_vids_above_thresh}/{percent_vids_above_thresh:.2f}% , acc {acc:.3f}\n')




    final_results_write.write('\n')
    final_results_write.close()


def initialize_logger(cfg, timestamp):
    global logger
    # timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(cfg.work_dir, f'{timestamp}.log')
    logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)

    return logger