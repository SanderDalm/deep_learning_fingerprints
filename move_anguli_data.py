from glob import glob
from os import rename, remove
import shutil
from tqdm import tqdm

shutil.copytree('/media/sander/Data/fingerprints', '/home/sander/data/deep_learning_fingerprints/anguli')
source_path = '/home/sander/data/deep_learning_fingerprints/anguli/batch1/Fingerprints/fp_1/'
file_list = glob(source_path+'*')
len(file_list)

for name in tqdm(file_list):
    #print(name)
    filename = name.split('/')[-1]
    number = filename.split('.')[0]
    new_filename = str(int(number)+100000)+'.jpg'
    new_name = source_path.replace('batch1', 'final')+new_filename
    #print(new_name)
    shutil.copy(name, new_name)


filelist = glob('/home/sander/data/deep_learning_fingerprints/anguli/*/*/*/*')
len(filelist)
for name in tqdm(filelist):
   remove(name)