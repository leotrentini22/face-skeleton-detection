import os
import pandas as pd
import numpy as np
from open_pifpaf_utils import *


#-------------------------------------------------------------------------------------------------------------------------------
# AffWild2
print("processing AffWild2 skeletons------------------------------------------------------------")

au_ids  = ['1', '2' ,'4', '6', '7', '10', '12', '15', '23', '24', '25', '26']


list_path_prefix, skeleton_path, img_path_vita, label_root, train_path, val_path, test_path = set_your_paths()

train_list = os.listdir(os.path.join(label_root, train_path))
val_list = os.listdir(os.path.join(label_root, val_path))
test_list = os.listdir(os.path.join(label_root, test_path))



# Validation

for val_txt in val_list:
    with open(os.path.join(os.path.join(label_root, val_path), val_txt), 'r') as f:
        lines = f.readlines()
    lines = lines[1:]
    for j, line in enumerate(lines):
        line = line.rstrip('\n').split(',')
        line = np.array(line).astype(np.int32)
        if -1 in line:
            continue
        actual_img_path = os.path.join(os.path.join(img_path_vita,os.path.basename(os.path.normpath(val_txt.split('.')[0]))), str(j+1).zfill(5)+'.jpg')
        skeleton_output_dir = os.path.join(skeleton_path,val_path)
        calculate_skeleton(actual_img_path, skeleton_output_dir)

