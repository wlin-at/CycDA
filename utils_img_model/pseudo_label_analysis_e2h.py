
import os.path as osp

import os
import numpy as np
from utils_img_model import get_class_dict
# from ..utils_img_model import
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
def plt_grouped_bar(labels,  list_of_list, legend_list = None, bar_width = 0.35, title =None, y_label = None):
    labels = list(labels)
    x = np.arange(len(labels))  # the label locations
    # the width of the bars
    n_bars = len(list_of_list)

    fig, ax = plt.subplots(figsize=(10, 5))
    shift_list = list(np.arange( - (n_bars -1) / 2.0, (n_bars -1) / 2.0 + 0.5, 1.0  ))
    rects_list = []
    for bar_id in range(n_bars):
        rects = ax.bar(x  + bar_width * shift_list[bar_id] ,  list_of_list[bar_id], bar_width, label=legend_list[bar_id])
        rects_list.append(rects)
        # rects2 = ax.bar(x + width / 2, list_of_list[bar_id], width, label='Women')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.set_xticks(x, )
    ax.set_xticklabels(labels)

    ax.legend()

    # for bar_id in range(n_bars):
    #     ax.bar_label( rects_list[bar_id], padding =3)
    # # ax.bar_label(rects1, padding=3)
    # # ax.bar_label(rects2, padding=3)

    # fig.tight_layout()

    plt.show()
    return fig

def plot_cf_mat(cf_matrix, id_to_class_dict = None, title = None , vmin = 0, vmax = None):
    # group_names = ['True Neg','False Pos','False Neg','True Pos','True Pos','True Pos','True Pos','True Pos','True Pos']
    fig, ax = plt.subplots(figsize=(10, 10))
    n_class = cf_matrix.shape[0]
    group_counts = ["{0:0.0f}".format(value) for value in
                    cf_matrix.flatten()]

    group_percentages = ["{0:.2%}".format(value) for value in
                         cf_matrix.flatten() / np.sum(cf_matrix)]

    labels = [f"{v1}\n{v2}\n" for v1, v2 in
              zip(group_counts, group_percentages)]

    labels = np.asarray(labels).reshape(n_class, n_class)

    ax = sns.heatmap(cf_matrix, annot=labels, fmt='', cmap='Blues', vmin= vmin, vmax=vmax)

    # ax.set_title('Seaborn Confusion Matrix with labels\n\n')
    ax.set_title(title)
    ax.set_xlabel('\nPredicted Category')
    ax.set_ylabel('Actual Category ')

    ## Ticket labels - List must be in alphabetical order
    class_labels = []
    for class_id in range(n_class):
        class_labels.append( id_to_class_dict[class_id] )

    ax.xaxis.set_ticklabels(class_labels)
    ax.yaxis.set_ticklabels(class_labels)

    ## Display the visualization of the Confusion Matrix.
    plt.show()

    return fig


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

    # gt_concat = []
    # pred_concat = []

    for vidname, items in img_ps_dict.items():
        gt_label_seq, pred_label_seq, max_pred_score_seq, pred_scores_all = items[0], items[1], items[2], items[3]
        # gt_concat = np.concatenate([gt_concat, gt_label])
        # pred_concat = np.concatenate([pred_concat, pred_label])

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

    # confusion_mat = confusion_matrix(gt_concat, pred_label)

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
    # n_vids = len(vid_ps_dict)
    # gt_concat = np.zeros((n_vids, ))
    # pred_concat = np.zeros((n_vids, ))
    gt_concat = []
    pred_concat = []

    for vidpath, items in vid_ps_dict.items():
        ps_confidence, pred_label, gt_label = items[2], items[1], items[0]
        gt_concat.append( int(gt_label))
        pred_concat.append( int(pred_label))
        # gt_concat = np.concatenate( [gt_concat, gt_label] )
        # pred_concat = np.concatenate([pred_concat, pred_label])

        n_vid_total_class[int(gt_label)] += 1
        if ps_confidence >= ps_thresh:
            n_vids_above_thresh +=1
            n_vid_above_thresh_class[gt_label] += 1
            if pred_label == int(gt_label):
                n_correct_vid_total += 1
                n_correct_vid_class[gt_label] += 1

    # confusion_mat = confusion_matrix( np.array(gt_concat).astype(int), np.array(pred_label).astype(int) )
    confusion_mat = confusion_matrix(gt_concat, pred_concat)
    acc_vid_class_array = np.zeros((n_class, ))
    for class_id in range(n_class):
        percent_vid_above_thresh_class = float(n_vid_above_thresh_class[class_id]) / n_vid_total_class[class_id]
        acc_vid_class = float(n_correct_vid_class[class_id]) / n_vid_total_class[class_id]
        acc_vid_class_array[class_id] = acc_vid_class
        print(f'class {class_id} {id_to_class_dict[class_id]} #vids {n_vid_above_thresh_class[class_id]}/{percent_vid_above_thresh_class * 100.0:.1f}%, acc vid {acc_vid_class:.3f}')
    percent_vid_above_thresh = float(n_vids_above_thresh) / len(vid_ps_dict)
    acc_vid_ps = float(n_correct_vid_total) / len(vid_ps_dict)
    print(f'Total all all #vids {n_vids_above_thresh}/{percent_vid_above_thresh * 100.0:.3f}%, acc vid {acc_vid_ps:.3f}')
    return acc_vid_class_array, confusion_mat



if __name__ == '__main__':
    main_dir = '/media/data_8T/UCF-HMDB/ablation_incepi3d/img2vid/e2h/split1'


    step1_img_ps_file = osp.join(main_dir, 'step1_grl_tsn_img_model_64d/v4', '20220206_183442_frame_ps.npy'  )  # 0.42
    # imgage pseudo label   {vidname:  gt_label_seq,  pred_label_seq,  max_pred_score_seq,  all_class_scores (n_frames, n_class ) }

    step1_img_ps = np.load(step1_img_ps_file, allow_pickle=True).item()

    step2_vid_ps_file = osp.join(main_dir, 'step1_grl_tsn_img_model_64d/v4/step2_vid_model/20epochs_0.7', 'epoch_20_test3clip_vid_ps.npy' )  # 0.553
    #   {vidname:   ( gt label, pred label,  max_pred_score, predicted scores for all classes )  }

    step3_img_ps_file = osp.join(main_dir, 'step1_grl_tsn_img_model_64d/v4/step2_vid_model/20epochs_0.7/step3_contrast_img_model/v5', '20220210_164005_frame_ps.npy')  #  0.558
    step4_vid_ps_file = osp.join(main_dir, 'step1_grl_tsn_img_model_64d/v4/step2_vid_model/20epochs_0.7/step3_contrast_img_model/v5/step4_vid_model', 'epoch_20_test3clip_vid_ps.npy' )  #  0.609

    n_class = list(step1_img_ps.values())[0][3].shape[-1]  # all_class_scores has shape (m_frames, n_class )

    ps_thresh_percent = 1.0
    e2h_class_to_id_dict, e2h_id_to_class_dict = get_class_dict(osp.join('/media/data_8T/UCF-HMDB/E2H', 'action_list.txt'))
    fig_dir = '/home/eicg/Documents/Lin/Understanding Long-term Activity/eccv22_img2vid/figs'

    print('\n')
    print('Processing step1 img ps  ...')

    acc_vid_class_step1 =  process_img_ps( img_ps_dict = step1_img_ps,  ps_thresh_percent=  ps_thresh_percent , id_to_class_dict=e2h_id_to_class_dict)

    print('\n')
    print('Processing step2 vid ps ...')
    step2_vid_ps = np.load(step2_vid_ps_file, allow_pickle=True).item()
    acc_vid_class_step2, confusion_matrix_step2 = process_vid_ps( vid_ps_dict = step2_vid_ps , ps_thresh_percent = ps_thresh_percent, id_to_class_dict =e2h_id_to_class_dict)
    fig_cf_step2 = plot_cf_mat(confusion_matrix_step2, id_to_class_dict=e2h_id_to_class_dict, title='CycDA step 2',vmin=5, vmax=64)
    fig_cf_step2.savefig(osp.join(fig_dir, 'cf_stage2.svg' ))

    print('\n')
    print('Processing step3 img ps  ...')
    step3_img_ps = np.load(step3_img_ps_file, allow_pickle=True).item()
    acc_vid_class_step3 = process_img_ps(img_ps_dict=step3_img_ps, ps_thresh_percent=ps_thresh_percent, id_to_class_dict=e2h_id_to_class_dict)

    print('\n')
    print('Processing step4 vid ps ...')
    step4_vid_ps = np.load(step4_vid_ps_file, allow_pickle=True).item()
    acc_vid_class_step4, confusion_matrix_step4 = process_vid_ps(vid_ps_dict=step4_vid_ps, ps_thresh_percent=ps_thresh_percent, id_to_class_dict=e2h_id_to_class_dict)
    fig_cf_step4 = plot_cf_mat( confusion_matrix_step4, id_to_class_dict=e2h_id_to_class_dict, title='CycDA step 4', vmin=5, vmax=64)
    fig_cf_step4.savefig(osp.join(fig_dir, 'cf_stage4.svg'))

    cf_mat_diff = confusion_matrix_step4 - confusion_matrix_step2
    fig_cf_mat_diff = plot_cf_mat(cf_mat_diff, id_to_class_dict=e2h_id_to_class_dict, title='cf_mat_diff')


    print('\n')
    print('Processing step3 self train vid ps ...')
    step3_self_train_ps_file = osp.join(main_dir, 'step1_grl_tsn_img_model_64d/v4/step2_vid_model/20epochs_0.7/step3_vid_model', 'epoch_20_test3clip_vid_ps.npy'  )
    step3_self_train_vid_ps = np.load(step3_self_train_ps_file, allow_pickle=True).item()
    acc_vid_class_step3_self_train, _ = process_vid_ps(vid_ps_dict=step3_self_train_vid_ps, ps_thresh_percent=ps_thresh_percent, id_to_class_dict=e2h_id_to_class_dict)


    print('\n')
    print('Processing step 4 self train vid ps ...')
    step4_self_train_ps_file = osp.join(main_dir, 'step1_grl_tsn_img_model_64d/v4/step2_vid_model/20epochs_0.7/step3_vid_model/step4_vid_model/0.9_20epochs', 'epoch_20_test3clip_vid_ps.npy'  )
    step4_self_train_vid_ps = np.load(step4_self_train_ps_file, allow_pickle=True).item()
    acc_vid_class_step4_self_train, _ = process_vid_ps(vid_ps_dict=step4_self_train_vid_ps, ps_thresh_percent=ps_thresh_percent, id_to_class_dict=e2h_id_to_class_dict)

    labels = list(e2h_class_to_id_dict.keys())
    acc_vid_class_step2 = list(acc_vid_class_step2)
    acc_vid_class_step4 = list(acc_vid_class_step4)
    acc_vid_class_step4_self_train = list(acc_vid_class_step4_self_train)

    list_combined = []
    for class_id in range(len(labels)):
        list_combined.append( ( labels[class_id], acc_vid_class_step2[class_id], acc_vid_class_step4[class_id], acc_vid_class_step4_self_train[class_id] ) )
    list_combined = sorted(list_combined, key=lambda x: x[1])

    zipped = zip(labels, acc_vid_class_step2, acc_vid_class_step4, acc_vid_class_step4_self_train)
    zipped = list(zipped)
    zipped = sorted(zipped, key=lambda x:x[1])
    labels, acc_vid_class_step2, acc_vid_class_step4, acc_vid_class_step4_self_train = zip(*zipped)


    # list_of_list = [  acc_vid_class_step2, acc_vid_class_step4, acc_vid_class_step4_self_train ]
    # legend_list = ['step2', 'step4 ours', 'step4 self train', ]

    list_of_list = [  acc_vid_class_step2, acc_vid_class_step4 ]
    legend_list = ['step2', 'step4 ours' ]

    fig = plt_grouped_bar(labels,  list_of_list, legend_list=legend_list, bar_width=0.25,
                    title = 'performance', y_label = 'acc')

    t = 1
    fig.savefig( osp.join( fig_dir, 'ps_label_analysis_v2.svg') )

