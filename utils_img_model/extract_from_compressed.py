import os
import glob
import os.path as osp
data_dir = '/media/data_8T/UCF-HMDB/UCF-HMDB_unused_actions/HMDB'
file_ext = '.rar'

file_list = glob.glob(  osp.join( data_dir,  f'*{file_ext}' )  )
for file in file_list:
    os.system(f'cd {data_dir}')
    os.system( f'unar {file}' )