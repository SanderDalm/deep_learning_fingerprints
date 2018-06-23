from glob import glob
from os import rename, remove
import shutil
from tqdm import tqdm

#shutil.copytree('/media/sander/Data/fingerprints', '/home/sander/data/deep_learning_fingerprints/anguli')
source_path = '/home/sander/data/deep_learning_fingerprints/anguli/batch1/Impression_1/fp_1/' #Impression_2 Meta Info
file_list = glob(source_path+'*')
len(file_list)

for name in tqdm(file_list):
    filename = name.split('/')[-1]
    number = filename.split('.')[0]
    new_filename = str(int(number)+200000)+'.jpg' #'.txt'
    new_name = source_path.replace('batch3', 'final')+new_filename
    #print(name)
    #print(new_name)
    shutil.copy(name, new_name)


# filelist = glob('/home/sander/data/deep_learning_fingerprints/anguli/*/*/*/*')
# len(filelist)
# for name in tqdm(filelist):
#    remove(name)



filelist = glob('/home/sander/data/deep_learning_fingerprints/anguli/final/Meta Info/fp_1/*')
len(filelist)