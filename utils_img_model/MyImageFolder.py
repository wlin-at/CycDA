
import torchvision
from typing import Any, Callable, cast, Dict, List, Optional, Tuple
from torchvision import datasets, transforms

import torch.utils.data as data

from PIL import Image
import os
import os.path as osp
import glob
import random
import numpy as np
import getpass
ucf_mapping_dict = {
    0: ['RockClimbingIndoor', 'RopeClimbing'],
    1: ['Fencing'],
    2: ['GolfSwing'],
3: ['SoccerPenalty'],
4: ['PullUps'],
5: ['Punch', 'BoxingPunchingBag', 'BoxingSpeedBag' ],
6: ['PushUps'],
7: ['Biking'],
8: ['HorseRiding'],
9: ['Basketball'],
10: ['Archery'],
11: ['WalkingWithDog'],
}


class MyImageFolder(datasets.ImageFolder):

    def __init__(self, root, transform ):
        super().__init__(root, transform)

    def __getitem__(self, index: int) -> Tuple[Any, Any, Any]:
        """
        todo this customized dataset is used for computing pseudo labels for frames in target training set.
        todo  the paths of the images are needed to organize pseudo labels in different videos

        Args:
            index (int): Index

        Returns:
            tuple: (path, sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return path, sample, target


def default_loader(path):
    return Image.open(path).convert('RGB')


def default_flist_reader(flist):
    """
    flist format: impath label\nimpath label\n ...(same to caffe's filelist)
    """
    imlist = []
    with open(flist, 'r') as rf:
        for line in rf.readlines():
            impath, imlabel = line.strip().split()
            imlist.append((impath, int(imlabel)))

    return imlist


# for each video, uniformly divide it into 5 segments
# in each segment, randomly sample a frame
def xxx(prob_sampling = None,):
    if prob_sampling:
        pass

def get_correct_path(path,):
    dir1 = '/media/data_8T'
    dir2 = '/data/lin'
    if getpass.getuser() == 'eicg':
        if (dir1 not in path) and (dir2 in path):
            path= path.replace(dir2, dir1)
    elif getpass.getuser() == 'lin':
        if (dir2 not in path) and (dir1 in path):
            path = path.replace(dir1, dir2)
    else:
        raise Exception("Unknown username!")
    return path

def load_frame_list( list_file, data_prefix = None  ):
    frame_list = []
    for line in open(list_file):
        items = line.strip('\n').split(' ')
        if data_prefix is None:
            frame_list.append( (items[0], int(items[1])))
        else:
            frame_list.append(( osp.join( data_prefix, items[0]), int(items[1])))
    return frame_list

def generate_target_frame_list(vid_info_dict, mapping_dict,
                               ps_thresh_percent = None,
                               img_dir=None, img_format='.png', target_train_vid_level_pseudo_dict = None,
                               ps_main_dir = '/data/lin/UCF-HMDB/HMDB/hmdb51_org', vid_format = '.avi',

                               tsn_sampling = False, n_segments = None,
                               logger = None):
    """
    generate frame_list  that contains all the frames of the target videos
    each element is a tuple of 3 :  imgpath, label, ps_label

    :param target_train_vid_level_pseudo_dict   {vidpath: (vid_label, pred_label, max_predict_score, outputs[vid_idx], n_frames)}
     todo   notice that in target_train_vid_level_pseudo_dict,  the vidpath starts with /data/lin
    :return:
    """
    target_train_vid_level_pseudo_dict = np.load(target_train_vid_level_pseudo_dict, allow_pickle=True).item()



    # determine the confidence score threshold according to the percent
    confidence_list = [ items[2]  for vidpath, items in target_train_vid_level_pseudo_dict.items() ]
    confidence_list = sorted(confidence_list, reverse=True)
    pos_ = min(len(confidence_list) - 1, int(len(confidence_list) * float(ps_thresh_percent)))
    thresh = confidence_list[pos_]

    frame_list = []
    n_correct_vids = 0
    n_vids_above_thresh = 0
    n_correct_frames = 0
    n_frames_above_thresh = 0
    n_frames_total = 0
    for vidname, (vid_len, class_id) in vid_info_dict.items():
        class_name = mapping_dict[class_id]
        if ps_main_dir == '/data/lin/UCF-HMDB/UCF/UCF-101':
            class_name_list = ucf_mapping_dict[class_id]
            for class_name_ in class_name_list:
                if osp.join(  ps_main_dir, class_name_, vidname+vid_format ) in target_train_vid_level_pseudo_dict:
                    class_name_for_ps = class_name_
                    break
        else:
            class_name_for_ps = class_name
        vidpath_in_ps_dict = osp.join(ps_main_dir, class_name_for_ps, vidname + vid_format)
        # ps_vid_label = target_train_vid_level_pseudo_dict[vidpath_in_ps_dict][1]  # vid-level pseudo label
        gt_vid_label, ps_vid_label, max_predict_score, pred_scores_all_class, n_frames = target_train_vid_level_pseudo_dict[vidpath_in_ps_dict]


        n_frames_total += n_frames
        # todo !!!!!! if a vid has confidence score above thresh,
        #  !!!!!!!!!! add all frames (either TSN style sampling or all frames ) in this vid for training
        if max_predict_score >= thresh:
            n_vids_above_thresh += 1
            n_frames_above_thresh += n_frames
            if tsn_sampling:  # todo  for video that has confidence score above thresh, we perform TSN sampling for frames in this video
                # sample in TSN style, uniformly sample each video into segments, then randomly sample a frame from each segment
                seg_len = int(np.floor(float(vid_len) / n_segments))
                seg_len_list = [seg_len] * n_segments
                for idx in range(vid_len - seg_len * n_segments):
                    seg_len_list[idx] += 1
                selected_frames = []
                start = 0
                for seg_len in seg_len_list:
                    end = start + seg_len -1
                    selected_frames.append( random.randint( start,  end )  )  # randint -  both borders included
                    start = end +1
                for frame_id in selected_frames:
                    frame_list.append( (osp.join(img_dir, class_name, vidname, f'{frame_id:08d}{img_format}'), class_id, ps_vid_label ) )
            else:
                # add all frames in this video for training
                for frame_id in range(vid_len):
                    frame_list.append((osp.join(img_dir, class_name, vidname, f'{frame_id:08d}{img_format}'), class_id, ps_vid_label ))
            if gt_vid_label == ps_vid_label:
                n_correct_vids += 1
                n_correct_frames += n_frames
    percent_vids_above_thresh = float(n_vids_above_thresh) / len(target_train_vid_level_pseudo_dict) * 100.0
    acc_vid = np.NaN if n_vids_above_thresh == 0 else float(n_correct_vids) / n_vids_above_thresh

    percent_frames_above_thresh = float(n_frames_above_thresh) / n_frames_total * 100.0
    acc_frame = np.NaN if n_frames_above_thresh == 0 else float(n_correct_frames) / n_frames_above_thresh

    to_print = f'Thresh {thresh:.2f} ({ps_thresh_percent * 100.0:.1f}%) : #vids {n_vids_above_thresh}/{percent_vids_above_thresh:.2f}% , acc vid {acc_vid:.3f} #frames {n_frames_above_thresh}/{percent_frames_above_thresh:.2f}% , acc frame {acc_frame:.3f}'
    if logger is not None:
        logger.debug(to_print)
    else:
        print(to_print)
    return frame_list

def generate_tsn_frames( vid_info_dict, mapping_dict, n_segments = 5,
        img_dir = None ,  img_format = '.png', n_digits =8,  prob_sampling = False, sample_first = 'low',   source_only_pseudo_dict = None  ):
    """
    
    :param img_dir:  e.g. /media/data_8T/UCF-HMDB/HMDB/hmdb51_org_imgs/train
    :return:
    """
    if prob_sampling:
        pseudo_scores_dict = np.load( source_only_pseudo_dict, allow_pickle=True ).item() # vidname : ( gt label, pred label,  max_pred_score, predicted scores for all classes )   # (n_frames, n_class)

    frame_list = []
    # img_name_format = f'0{n_digits}d'
    for vidname, (vid_len, class_id) in vid_info_dict.items():
        if prob_sampling:
            gt_label_seq, _, _, pred_scores_all = pseudo_scores_dict[vidname]
            # derive the video-level confidence score as the average of confidence scores of frames
            vid_confidence = np.mean( pred_scores_all, axis= 0  ) # (n_class, )
            vid_ps_label = np.argmax(vid_confidence)
            if sample_first == 'low':
                # sample_prob_vid =  np.reciprocal(pred_scores_all[:,  vid_ps_label] )  # (n_frames,  )   reciprocal of confidence scores, todo  frames with low confidence scores have high probability in sampling
                sample_prob_vid =  1- pred_scores_all[:,  vid_ps_label]   # (n_frames,  ) todo  frames with low confidence scores have high probability in sampling
            elif sample_first == 'high':
                sample_prob_vid = pred_scores_all[:, vid_ps_label]  # (n_frames,  ) # todo frames with high confidence scores have high probability in sampling
        class_name = mapping_dict[class_id]
        # seg_len = int(np.ceil( float(vid_len) / n_segments  ))
        seg_len = int(np.floor( float(vid_len) / n_segments  ))
        # seg_len_list = [seg_len] * (n_segments -1) + [ vid_len- seg_len*(n_segments-1) ]
        seg_len_list = [seg_len] * n_segments
        for idx in range( vid_len - seg_len * n_segments ):
            seg_len_list[idx] += 1
        selected_frames = []
        start = 1 if 'NEC-Drone' in img_dir else 0  # todo NEC-Drone starts from 1
        for seg_len in seg_len_list:
            end = start + seg_len -1  #   (0, 5)  (6, 11),  (12, 17)
            if prob_sampling:
                sample_prob_seg = sample_prob_vid[start : end+1 ]
                sample_prob_seg = sample_prob_seg / np.sum(sample_prob_seg)
                selected_frames.append( np.random.choice( np.arange(start, end+1 ), p= sample_prob_seg, replace=False )  ) # sample without replacement
            else:
                selected_frames.append( random.randint( start,  end ) )  # todo random.randint returns a random number, both borders are included
            start = end +1

        for frame_id in selected_frames:
            if 'NEC-Drone' in img_dir:
                frame_list.append((osp.join(img_dir, class_name, vidname, f'{frame_id:05d}{img_format}'), class_id))
            else:
                frame_list.append( ( osp.join(img_dir, class_name, vidname, f'{frame_id:08d}{img_format}'), class_id ) )
    return frame_list

def uniform_sample_frames( vid_info_dict, mapping_dict, n_segments = 5,
        img_dir = None ,  img_format = '.png'   ):
    frame_list = []
    for vidname, (vid_len, class_id) in vid_info_dict.items():
        class_name = mapping_dict[class_id]
        # seg_len = int(np.ceil( float(vid_len) / n_segments  ))
        seg_len = int(np.floor(float(vid_len) / n_segments))
        # seg_len_list = [seg_len] * (n_segments -1) + [ vid_len- seg_len*(n_segments-1) ]
        seg_len_list = [seg_len] * n_segments
        for idx in range( vid_len - seg_len * n_segments ):
            seg_len_list[idx] += 1
        selected_frames = []
        start = 1 if 'NEC-Drone' in img_dir else 0  # todo NEC-Drone starts from 1
        for seg_len in seg_len_list:
            end = start + seg_len - 1  # (0, 5)  (6, 11),  (12, 17)
            selected_frames.append(  int( (start + end) / 2.0  ))
            start = end + 1

        for frame_id in selected_frames:
            if 'NEC-Drone' in img_dir:
                frame_list.append((osp.join(img_dir, class_name, vidname, f'{frame_id:05d}{img_format}'), class_id))
            else:
                frame_list.append((osp.join(img_dir, class_name, vidname, f'{frame_id:08d}{img_format}'), class_id))
    return frame_list

def load_vid_info( datalist, img_dir = None, mapping_dict = None, img_format = '.png' ):
    """  load  n_frames and class_id
    :param img_dir:  e.g. /media/data_8T/UCF-HMDB/HMDB/hmdb51_org_imgs/train
    :param datalist:   e.g.   /media/data_8T/UCF-HMDB/datalists_UCF-HMDB/list_hmdb51_train_hmdb_ucf-feature.txt
    :return:
    """
    # load the #frames for each video
    # todo  the number of frames in the datalist could be incorrect
    vid_info_dict = dict()
    for line in open(datalist):
        items = line.strip('\n').split(' ')
        vidname = items[0].split('/')[-1].split('.')[0]
        # n_frames = int(items[1]) # todo the n_frames could be incorrect
        if len(items) == 3:
            class_id = int(items[2])
        elif len(items) == 2:
            class_id = int(items[1])
        class_name = mapping_dict[class_id]
        # n_frames = len(glob.glob( osp.join( img_dir, class_name,  vidname, f'*{img_format}' )  ))
        # todo  glob.glob does not work if there is special character, e.g. '[' or ']' contained in the path
        #   here we use  glob.glob1("/Users/dir", '*.txt')
        n_frames = len(glob.glob1( osp.join( img_dir, class_name,  vidname), '*'   ))
        assert n_frames > 0
        vid_info_dict.update({ vidname: (n_frames, class_id) })
    return vid_info_dict

def load_mapping( mapping_file):
    mapping_dict = dict()
    for line in open(mapping_file):
        items = line.strip('\n').split(' ')
        mapping_dict.update({ int(items[0]) : items[1] })
    return mapping_dict

class ImageFilelist(data.Dataset):
    def __init__(self, imlist,
                 transform=None, target_transform=None,
                 loader=default_loader,  w_ps = False, return_index = False,):
        # self.root = root
        # self.imlist = flist_reader(flist)
        self.imlist = imlist

        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        self.w_ps = w_ps
        self.return_index = return_index

    def __getitem__(self, index):
        if self.w_ps:
            # todo if with pseudo label, there are 3 items in the list
            impath, target, ps_target = self.imlist[index]
        else:
            impath, target = self.imlist[index]
        # img = self.loader(os.path.join(self.root, impath))
        img = self.loader( impath)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        if self.w_ps:
            if self.return_index:
                return img, target, ps_target, index
            else:
                return img, target, ps_target
        else:
            if self.return_index:
                return img, target, index
            else:
                return img, target

    def __len__(self):
        return len(self.imlist)

class ImageFilelist_w_path(data.Dataset):
    def __init__(self, imlist,
                 transform=None, target_transform=None,
                 loader=default_loader,  w_ps = False):
        # self.root = root
        # self.imlist = flist_reader(flist)
        self.imlist = imlist

        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        self.w_ps = w_ps

    def __getitem__(self, index):
        if self.w_ps:
            # todo if with pseudo label, there are 3 items in the list
            impath, target, ps_target = self.imlist[index]
        else:
            impath, target = self.imlist[index]
        # img = self.loader(os.path.join(self.root, impath))
        img = self.loader( impath)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        if self.w_ps:
            return impath, img, target, ps_target
        else:
            return impath, img, target

    def __len__(self):
        return len(self.imlist)

# if __name__ == '__main__':
#     # generate_tsn_frames(img_dir= '/media/data_8T/UCF-HMDB/HMDB/hmdb51_org_imgs/train' )
#     vid_info_dict = load_vid_info(datalist= '/media/data_8T/UCF-HMDB/datalists_UCF-HMDB/list_hmdb51_train_hmdb_ucf-feature.txt')
#     mapping_dict = load_mapping( mapping_file= '/media/data_8T/UCF-HMDB/datalist_new/mapping_hmdb.txt')
#     frame_list = generate_tsn_frames(vid_info_dict, mapping_dict, img_dir= '/media/data_8T/UCF-HMDB/HMDB/hmdb51_org_imgs/train' )
#     pass