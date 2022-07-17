from __future__ import print_function
from __future__ import division

import torchvision
import os.path as osp
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
import sys
sys.path.append("..")  # last level
import argparse

print("PyTorch Version: ",torch.__version__)
print("Torchvision Version: ",torchvision.__version__)
# from .,utils_img_cls import train_model, initialize_model
# from ..utils_img_model.utils_img_cls import train_model, initialize_model
from utils_img_model.MyImageFolder import load_mapping
from utils_img_model.train_img_cls import run



parser = argparse.ArgumentParser()
parser.add_argument('--debug', default=False, action='store_true' )
parser.add_argument('--eval_set_id', type=int, default=2)
parser.add_argument('--split_nr', type=int, default=1)
parser.add_argument('--img_format', default='.jpg')
parser.add_argument('--main_dir', default='/media/data_8T/UCF-HMDB')
parser.add_argument('--dataloader_iter', default='max', choices=['min', 'max'],
                    help='use target loader to determine the number of iterations in each epoch, set to max when the target loader has more samples than source loader.')
parser.add_argument('--ps_main_dir', default='/data/lin/UCF-HMDB/UCF-HMDB_all/UCF', help='Path prefix for data with pseudo labels ')
parser.add_argument('--num_epochs', default=20)
parser.add_argument('--val_frequency', default=20)
parser.add_argument('--compute_pseudo_labels', default='per_epoch', choices=[False, True, 'per_epoch']) # when val_frequency == num_epochs, only one inference will be done
parser.add_argument('--return_model', default='last', choices=['best', 'last'])
parser.add_argument('--model_name', default="resnet_grl_tsn", choices=['resnet_grl_tsn', 'resnet_contrast_tsn'])
parser.add_argument('--n_frames_per_vid', default=5, type=int, )
parser.add_argument('--beta_high', default=1.0, type=float)
parser.add_argument('--batch_size', default=32,)
parser.add_argument('--if_dim_reduct', default=True, action='store_true')
parser.add_argument('--reduced_dim', default=128, help='feature dimension')
parser.add_argument('--clsfr_mlp_dim', default=128, help='classier FC dimension')
parser.add_argument('--d_clsfr_mlp_dim', default=128, help='domain discriminator FC dimension')
parser.add_argument('--num_workers', default=12)
parser.add_argument('--lr', default=0.001)
parser.add_argument('--freeze_child_nr', default=6, help='number of blocks to freeze in the backbone')

args = parser.parse_args()

# env_id = get_env_id()
# main_dir = ['/media/data_8T/UCF-HMDB',  '/data/lin/UCF-HMDB'][env_id]


# todo ################################
main_dir = args.main_dir
debug = args.debug
dummy_str = '_dummy' if debug else ''

eval_set_id = args.eval_set_id
split_nr = args.split_nr
img_format = args.img_format
#  todo source has 23804 images , target  9537x5=47685 frames main load
dataload_iter = args.dataload_iter
ps_main_dir = args.ps_main_dir

num_epochs = args.num_epochs
val_frequency = args.val_frequency
compute_pseudo_labels = args.compute_pseudo_labels
return_model = args.return_model

model_name = args.model_name
n_frames_per_vid = args.n_frames_per_vid
beta_high = args.beta_high

batch_size = args.batch_size
if_dim_reduct = args.if_dim_reduct
reduced_dim = args.reduced_dim
clsfr_mlp_dim = args.clsfr_mlp_dim
d_clsfr_mlp_dim = args.d_clsfr_mlp_dim


num_workers = args.num_workers
lr = args.lr

# todo Flag for feature extracting. When False, we finetune the whole model,
#   when True we only update the reshaped layer params
# feature_extract = [False, True][0]
freeze_child_nr = args.freeze_child_nr
# todo ################################


# todo ########################################
source_dataset = ['Stanford40', 'EAD', 'BU101'][eval_set_id]
eval_setting = ['S2U', 'E2H', 'BU2UCF'][eval_set_id]
target_dataset = ['ucf', 'hmdb', 'ucf_all'][eval_set_id]
num_classes = [12,  13, 101][eval_set_id]

mapping_file = [ osp.join(main_dir, eval_setting,'action_list.txt'),
                 osp.join(main_dir, eval_setting,'action_list.txt'),
                 osp.join(main_dir, 'UCF-HMDB_all', 'UCF_mapping.txt' )][eval_set_id]

mapping_dict = load_mapping(mapping_file=mapping_file)


source_train_data_list = osp.join( main_dir, eval_setting, f'list_{eval_setting}_img_new{dummy_str}.txt' )
target_train_vidlist = [osp.join( main_dir, eval_setting, f'list_{target_dataset}_all.txt'),
                        osp.join( main_dir, eval_setting, f'list_{target_dataset}_all.txt'),
                      osp.join( main_dir, eval_setting, f'list_{target_dataset}_train_split{split_nr}{dummy_str}.txt')][eval_set_id]


# ps_main_dir = osp.join( '/data/lin/UCF-HMDB', ['UCF/UCF-101',
#                                                'HMDB/hmdb51_org',
#                                                'UCF-HMDB_all/UCF'][eval_set_id]         )


target_train_dir = [osp.join( main_dir, eval_setting, f'{target_dataset}_imgs', 'all'),
                    osp.join( main_dir, eval_setting, f'{target_dataset}_imgs', 'all'),
                    osp.join( main_dir, 'UCF-HMDB_all/ucf_all_imgs/all' )][eval_set_id]
target_train_data_list = [osp.join( main_dir, eval_setting, f'{target_dataset}_imgs', 'list_frames_all_allframes.txt') ,
                        osp.join( main_dir, eval_setting, f'{target_dataset}_imgs', 'list_frames_all_allframes.txt') ,
                        osp.join( main_dir, eval_setting,  f'list_frames_train_split{split_nr}_allframes{dummy_str}.txt')][eval_set_id]

val_data_list = target_train_data_list

target_train_vid_level_pseudo_dict = None
# todo ################################




run(source_dataset = source_dataset, target_dataset = target_dataset,
    target_train_vidlist = target_train_vidlist, target_train_dir=target_train_dir, source_train_img_list= source_train_data_list, val_img_list=val_data_list,
    target_train_img_list= target_train_data_list, target_train_vid_level_pseudo_dict= target_train_vid_level_pseudo_dict,
    mapping_dict = mapping_dict, main_dir = main_dir,
    model_name = model_name, n_frames_per_vid = n_frames_per_vid, num_classes = num_classes,
    batch_size = batch_size, num_epochs= num_epochs, lr= lr, dataload_iter =dataload_iter, beta_high = beta_high, val_frequency = val_frequency, num_workers = num_workers,
    if_dim_reduct = if_dim_reduct, reduced_dim =reduced_dim, clsfr_mlp_dim=clsfr_mlp_dim, d_clsfr_mlp_dim =d_clsfr_mlp_dim, freeze_child_nr =freeze_child_nr,
    return_model = return_model,
    compute_pseudo_labels = compute_pseudo_labels,
    ps_main_dir=ps_main_dir,
    img_format=img_format)

