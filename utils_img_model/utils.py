import os.path as osp
from PIL import Image
import os
import glob

def resize_and_save(imgpath, new_imgpath, base_width):
    img = Image.open(imgpath)
    wpercent = (base_width / float(img.size[0]))
    hsize = int(float(img.size[1]) * float(wpercent))
    img = img.resize((base_width, hsize), Image.ANTIALIAS)
    try:
        img.save(new_imgpath)
    except OSError:
        img = img.convert('RGB')
        img.save(new_imgpath)
def get_class_dict(file_ ):
    class_to_id_dict = dict()
    id_to_class_dict = dict()
    for line in open(file_):
        items = line.strip('\n').split(' ')
        class_id, class_name = int(items[0]), items[1]
        class_to_id_dict.update({ class_name: class_id })
        id_to_class_dict.update({ class_id: class_name })
    return class_to_id_dict, id_to_class_dict
def make_dir(_dir):
    if not os.path.exists(_dir):
        os.makedirs(_dir)