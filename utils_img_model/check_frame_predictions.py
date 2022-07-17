
import numpy as np
from utils_img_model.utils_img_cls import make_dir
import matplotlib.pyplot as plt
import os.path as op

pseudo_gt_dict = '/media/data_8T/UCF-HMDB/ucf_hmdb_img_train/ucf_train_hmdb_val/20211215_232454_frame_ps.npy'
save_dir = '/media/data_8T/UCF-HMDB/ucf_hmdb_img_train/ucf_train_hmdb_val/frame_prediction_plots'

make_dir(save_dir)

# vidname : ( gt label, pred label,  max_pred_score, predicted scores for all classes )
pseudo_scores_dict = np.load(pseudo_gt_dict, allow_pickle=True).item()
hmdb_action_list = ['climb', 'fencing','golf', 'kick_ball', 'pullup', 'punch', 'pushup','ride_bike', 'ride_horse', 'shoot_ball', 'shoot_bow', 'walk' ]

for vidname, items in pseudo_scores_dict.items():
    gt_label_seq, pred_label_seq, max_pred_score_seq, pred_scores_all = items[0], items[1], items[2], items[3] #  pred_scores_all (n_frames, n_class )
    gt_label = int(gt_label_seq[0])
    cls_dir =  op.join( save_dir, f'{hmdb_action_list[gt_label]}')
    make_dir( cls_dir )

    n_frames, n_class = pred_scores_all.shape
    cls_idx_list = []
    fig = plt.figure(figsize=(25,10))
    # plt.set_cmap('jet')
    cmap_jet = plt.get_cmap( 'tab20', n_class)
    colors = cmap_jet(np.arange( cmap_jet.N))
    for cls_idx in range(n_class):
        plt.plot( pred_scores_all[:, cls_idx ], color=colors[cls_idx] )
        cls_idx_list.append(f'{cls_idx} {hmdb_action_list[cls_idx]}')

    plt.xlabel('frame')
    plt.ylabel('confidence')
    plt.legend(cls_idx_list, loc='upper right')
    plt.title(f'{vidname} gt {gt_label} {hmdb_action_list[gt_label]}')
    plt.grid(True)
    fig.savefig(op.join( cls_dir, f'{vidname}.png' ))
