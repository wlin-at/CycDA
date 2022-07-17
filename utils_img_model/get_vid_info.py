import glob
from utils_img_model.MyImageFolder import load_mapping, load_vid_info
import getpass
import os.path as osp
import numpy as np
if getpass.getuser() == 'eicg':
    env_id = 0
elif getpass.getuser() == 'lin':
    env_id = 1
else:
    raise Exception("Unknown username!")

dataset_id = 0  #  we collect vid info for the target dataset
main_dir = [
    '/media/data_8T/UCF-HMDB',
    '/data/lin/UCF-HMDB'][env_id]
target_train_dir = [ osp.join( main_dir, 'UCF/UCF-101_imgs/train' ),
           osp.join(main_dir,'HMDB/hmdb51_org_imgs/train')][dataset_id]


target_train_datalist = osp.join(main_dir, 'datalist_new',  ['list_ucf101_train_hmdb_ucf-feature.txt',
                                                              'list_hmdb51_train_hmdb_ucf-feature.txt'][dataset_id])
mapping_file = osp.join(main_dir, 'datalist_new', 'mapping_hmdb.txt')

# vid_ps_file = '/home/eicg/action_recognition_codes/data_lin_on_krios/UCF-HMDB/work_dirs/ResNet3d_tmp_transformer_uh_DA_clip4x16_w_ps_filterbyframe_img0.3_targetonly/ps_labels/best_top1_acc_epoch_50_vid_ps.npy'


mapping_dict = load_mapping(mapping_file=mapping_file)


#  vidname: (n_frames, class_id)
target_train_vid_info_dict = load_vid_info(datalist= target_train_datalist, img_dir= target_train_dir, mapping_dict= mapping_dict, )

np.save( osp.join(main_dir,  'datalist_new', ['ucf_train_vid_info.npy',
                                              'hmdb_train_vid_info.npy'][dataset_id]), target_train_vid_info_dict)


# vid_ps = np.load