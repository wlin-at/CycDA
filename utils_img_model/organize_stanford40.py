import glob
import os
import os.path as osp

def make_dir(dir_):
    if not os.path.exists(dir_):
        os.makedirs(dir_)

if __name__ == '__main__':
    data_dir = '/media/data_8T/UCF-HMDB/Stanford40/JPEGImages'
    img_list = glob.glob(osp.join( data_dir, '*' ))
    for img_path in img_list:
        img_name = img_path.split('/')[-1].split('.')[0]
        action_name, img_nr = img_name.split('_')
        make_dir(osp.join(data_dir ))
