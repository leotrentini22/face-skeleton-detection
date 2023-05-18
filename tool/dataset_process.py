import os
import pandas as pd
import numpy as np
from open_pifpaf_utils import *

au_ids  = ['1', '2' ,'4', '6', '7', '10', '12', '15', '23', '24', '25', '26']
total_AUs = au_ids

new_dataset_train_img_list = []
new_dataset_val_img_list = []
new_dataset_test_img_list = []

new_dataset_train_label_list = []
new_dataset_val_label_list = []
new_dataset_test_label_list = []


#-------------------------------------------------------------------------------------------------------------------------------
# AffWild2
print("processing AffWild2------------------------------------------------------------")

au_ids  = ['1', '2' ,'4', '6', '7', '10', '12', '15', '23', '24', '25', '26']

new_dataset_train_img_list = []
new_dataset_val_img_list = []
new_dataset_test_img_list = []

new_dataset_train_label_list = []
new_dataset_val_label_list = []
new_dataset_test_label_list = []



list_path_prefix, skeleton_path, img_path_vita, label_root, train_path, val_path, test_path = set_your_paths()



train_list = os.listdir(os.path.join(label_root, train_path)) 

train_labels = os.path.join(list_path_prefix, 'AffWild2_train_label.txt')
with open(train_labels, 'w') as  f:
    i = 0

val_list = os.listdir(os.path.join(label_root, val_path))

val_labels = os.path.join(list_path_prefix, 'AffWild2_val_label.txt')
with open(val_labels, 'w') as  f:
    i = 0

test_list = os.listdir(os.path.join(label_root, test_path)) 

test_labels = os.path.join(list_path_prefix, 'AffWild2_test_label.txt') 
with open(test_labels, 'w') as  f:    
    i = 0


train_img_path = os.path.join(list_path_prefix, 'AffWild2_train_img_path.txt')
with open(train_img_path, 'w') as f:
    i = 0
val_img_path = os.path.join(list_path_prefix, 'AffWild2_val_img_path.txt')
with open(val_img_path, 'w') as f:
    i = 0
test_img_path = os.path.join(list_path_prefix, 'AffWild2_test_img_path.txt') 
with open(test_img_path, 'w') as f:    
    i = 0




# Train

au_labels = []
au_img_path = []
for train_txt in train_list:  
    with open(os.path.join(os.path.join(label_root, train_path), train_txt), 'r') as f: 
        lines = f.readlines()  
    lines = lines[1:] 
    for j, line in enumerate(lines): 
        line = line.rstrip('\n').split(',')
        line = np.array(line).astype(np.int32)
        if -1 in line: 
            continue
        au_labels.append(line.reshape(1, -1)) 
        actual_img_path = os.path.join(os.path.join(img_path_vita,os.path.basename(os.path.normpath(train_txt.split('.')[0]))), str(j+1).zfill(5)+'.jpg')
        au_img_path.append(actual_img_path)  



au_labels = np.concatenate(au_labels, axis=0)
AffWild2_train_image_label = np.zeros((au_labels.shape[0], len(total_AUs))) - 1
for i, au in enumerate(au_ids):
    au = str(au)
    index = total_AUs.index(au)
    AffWild2_train_image_label[:, index] = au_labels[:, i]

with open(train_img_path, 'a+') as f:
    for line in au_img_path:
        f.write(os.path.join('AffWild2',line+'\n'))
        new_dataset_train_img_list.append(os.path.join('AffWild2',line+'\n'))

np.savetxt(train_labels, AffWild2_train_image_label ,fmt='%d', delimiter=' ')
new_dataset_train_label_list.append(AffWild2_train_image_label)




# Validation

au_labels = []
au_img_path = []
for val_txt in val_list:
    with open(os.path.join(os.path.join(label_root, val_path), val_txt), 'r') as f:
        lines = f.readlines()
    lines = lines[1:]
    for j, line in enumerate(lines):
        line = line.rstrip('\n').split(',')
        line = np.array(line).astype(np.int32)
        if -1 in line:
            continue
        au_labels.append(line.reshape(1, -1))
        actual_img_path = os.path.join(os.path.join(img_path_vita,os.path.basename(os.path.normpath(val_txt.split('.')[0]))), str(j+1).zfill(5)+'.jpg')
        au_img_path.append(actual_img_path)  


au_labels = np.concatenate(au_labels, axis=0)
AffWild2_val_image_label = np.zeros((au_labels.shape[0], len(total_AUs))) - 1
for i, au in enumerate(au_ids):
    au = str(au)
    index = total_AUs.index(au)
    AffWild2_val_image_label[:, index] = au_labels[:, i]

with open(val_img_path, 'a+') as f:
    for line in au_img_path:
        f.write(os.path.join('AffWild2',line+'\n'))
        new_dataset_val_img_list.append(os.path.join('AffWild2',line+'\n'))

np.savetxt(val_labels, AffWild2_val_image_label ,fmt='%d', delimiter=' ')
new_dataset_val_label_list.append(AffWild2_val_image_label)





# Test

au_labels = []
au_img_path = []             
for test_txt in test_list:   
    with open(os.path.join(os.path.join(label_root, test_path), test_txt), 'r') as f:
        lines = f.readlines()
    lines = lines[1:]
    for j, line in enumerate(lines):
        line = line.rstrip('\n').split(',')
        line = np.array(line).astype(np.int32)
        if -1 in line:
            continue
        au_labels.append(line.reshape(1, -1))
        actual_img_path = os.path.join(os.path.join(img_path_vita,os.path.basename(os.path.normpath(test_txt.split('.')[0]))), str(j+1).zfill(5)+'.jpg')
        au_img_path.append(actual_img_path) 



au_labels = np.concatenate(au_labels, axis=0)
AffWild2_test_image_label = np.zeros((au_labels.shape[0], len(total_AUs))) - 1
for i, au in enumerate(au_ids):
    au = str(au)
    index = total_AUs.index(au)
    AffWild2_test_image_label[:, index] = au_labels[:, i]

with open(test_img_path, 'a+') as f:
    for line in au_img_path:
        f.write(os.path.join('AffWild2',line+'\n'))    #
        new_dataset_test_img_list.append(os.path.join('AffWild2',line+'\n'))     

np.savetxt(test_labels, AffWild2_test_image_label ,fmt='%d', delimiter=' ')
new_dataset_test_label_list.append(AffWild2_test_image_label)




new_dataset_train_label_list = np.concatenate(new_dataset_train_label_list, axis=0)
new_dataset_val_label_list = np.concatenate(new_dataset_val_label_list, axis=0)
new_dataset_test_label_list = np.concatenate(new_dataset_test_label_list, axis=0)


sub_list = [0,1,2,4,7,8,11]

for i in range(new_dataset_train_label_list.shape[0]):
    for j in range(27, 12): 
        sub_au_label = new_dataset_train_label_list[i, j]
        if sub_au_label >0:
            main_au_index = sub_list[ (j - 27) // 2]
            new_dataset_train_label_list[i, main_au_index] = 1


for i in range(new_dataset_val_label_list.shape[0]):
    for j in range(27, 12):
        sub_au_label = new_dataset_val_label_list[i, j]
        if sub_au_label >0:
            main_au_index = sub_list[ (j - 27) // 2]
            new_dataset_val_label_list[i, main_au_index] = 1

for i in range(new_dataset_test_label_list.shape[0]):
    for j in range(27, 12):
        sub_au_label = new_dataset_test_label_list[i, j]
        if sub_au_label >0:
            main_au_index = sub_list[ (j - 27) // 2]
            new_dataset_test_label_list[i, main_au_index] = 1

np.savetxt(os.path.join(list_path_prefix, 'AffWild2_train_label.txt'), new_dataset_train_label_list ,fmt='%d', delimiter=' ')
np.savetxt(os.path.join(list_path_prefix, 'AffWild2_train_label.txt'), new_dataset_val_label_list ,fmt='%d', delimiter=' ')
np.savetxt(os.path.join(list_path_prefix, 'AffWild2_train_label.txt'), new_dataset_test_label_list ,fmt='%d', delimiter=' ')

with open(os.path.join(list_path_prefix, 'AffWild2_train_img_path.txt') , 'w+') as f:
    for line in new_dataset_train_img_list:
        f.write(line)

with open(os.path.join(list_path_prefix, 'AffWild2_val_img_path.txt') , 'w+') as f:
    for line in new_dataset_val_img_list:
        f.write(line)

with open(os.path.join(list_path_prefix, 'AffWild2_test_img_path.txt') , 'w+') as f:
    for line in new_dataset_test_img_list:
        f.write(line)

