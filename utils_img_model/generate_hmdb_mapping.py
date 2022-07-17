
import glob
import os.path as osp
main_dir = '/media/data_8T/UCF-HMDB/UCF-HMDB_all'
dataset_name = 'HMDB'

data_dir = osp.join(main_dir, dataset_name)
folder_list = glob.glob( osp.join(data_dir, '*/'   )  )

class_list = []

for folder_path in folder_list:
    class_list.append( folder_path.split('/')[-2] )
class_list = sorted(class_list)
f_write = open(osp.join( main_dir, f'{dataset_name}_mapping.txt' ), 'w+')

for class_id, class_name in enumerate(class_list):
    f_write.write( f'{class_id} {class_name}\n' )

f_write.close()

pass