import os
import pandas as pd
import numpy as np
from open_pifpaf_utils import *

#AUs = ['1', '2' ,'4', '5', '6', '7', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '22' ,'23', '24', '25', '26', '27', '32', '38', '39']
#mcro_AUs = ['L1', 'R1', 'L2', 'R2', 'L4', 'R4', 'L6', 'R6', 'L10', 'R10', 'L12', 'R12', 'L14', 'R14']
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


# da togliere se vogliamo trainare su tutti i dataset
new_dataset_train_img_list = []
new_dataset_val_img_list = []
new_dataset_test_img_list = []

new_dataset_train_label_list = []
new_dataset_val_label_list = []
new_dataset_test_label_list = []
#fino a qui


list_path_prefix = '/home/trentini/face-skeleton-detection/data/AffWild2/list/' #'/work/vita/datasets/Aff-Wild2/Third_ABAW_Annotations/AU_Detection_Challenge/' #'Datasets/hybrid_dataset/AffWild2/list'
img_path_vita = '/work/vita/datasets/Aff-Wild2/cropped_aligned/'
skeleton_path = '/home/trentini/face-skeleton-detection/data/AffWild2/skeletons/'

#data/AffWild2/list
#/work/vita/datasets/Aff-Wild2/Third_ABAW_Annotations/AU_Detection_Challenge/

train_path ='Train_Set' 
val_path = 'Validation_Set' 
test_path = 'Validation_Set'   #there is no test set in our dataset

label_root = '/work/vita/datasets/Aff-Wild2/Third_ABAW_Annotations/AU_Detection_Challenge/'

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
for train_txt in train_list:  #scorre le cartelle in train list
    with open(os.path.join(os.path.join(label_root, train_path), train_txt), 'r') as f: #apre la cartella
        lines = f.readlines()  #legge contenuto
    lines = lines[1:] #toglie prima linea
    for j, line in enumerate(lines):   #converte contenuto linee
        line = line.rstrip('\n').split(',')
        line = np.array(line).astype(np.int32)
        if -1 in line:  #se c'e' -1, skippa linea
            continue
        au_labels.append(line.reshape(1, -1)) #appende la linea rishapata
        actual_img_path = os.path.join(os.path.join(img_path_vita,os.path.basename(os.path.normpath(train_txt.split('.')[0]))), str(j+1).zfill(5)+'.jpg')
        au_img_path.append(actual_img_path)  #appende il path dell'immagine, "split" splitta il path dove ci sono i punti e poi prende solo cio che viene prima del punto (quindi toglie ".txt")
        skeleton_output_dir = os.path.join(skeleton_path,Train_Set)
        calculate_skeleton(actual_image_path, skeleton_output_dir)


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
        au_img_path.append(actual_img_path)  #appende il path dell'immagine, "split" splitta il path dove ci sono i punti e poi prende solo cio che viene prima del punto (quindi toglie ".txt")
        skeleton_output_dir = os.path.join(skeleton_path,Val_Set)
        calculate_skeleton(actual_image_path, skeleton_output_dir)


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
au_img_path = []             # for loop not clear  ->   test_text is a number? Or it's just a loop over the folders
for test_txt in test_list:   # test list = directories in /work/vita/datasets/Aff-Wild2/Third_ABAW_Annotations/AU_Detection_Challenge/Validation_Set
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
        au_img_path.append(actual_img_path)  #appende il path dell'immagine, "split" splitta il path dove ci sono i punti e poi prende solo cio che viene prima del punto (quindi toglie ".txt")
        skeleton_output_dir = os.path.join(skeleton_path,Test_Set)
        calculate_skeleton(actual_image_path, skeleton_output_dir)


au_labels = np.concatenate(au_labels, axis=0)
AffWild2_test_image_label = np.zeros((au_labels.shape[0], len(total_AUs))) - 1
for i, au in enumerate(au_ids):
    au = str(au)
    index = total_AUs.index(au)
    AffWild2_test_image_label[:, index] = au_labels[:, i]

with open(test_img_path, 'a+') as f:
    for line in au_img_path:
        f.write(os.path.join('AffWild2',line+'\n'))    #
        new_dataset_test_img_list.append(os.path.join('AffWild2',line+'\n'))     # what does it do here?

np.savetxt(test_labels, AffWild2_test_image_label ,fmt='%d', delimiter=' ')
new_dataset_test_label_list.append(AffWild2_test_image_label)




new_dataset_train_label_list = np.concatenate(new_dataset_train_label_list, axis=0)
new_dataset_val_label_list = np.concatenate(new_dataset_val_label_list, axis=0)
new_dataset_test_label_list = np.concatenate(new_dataset_test_label_list, axis=0)


sub_list = [0,1,2,4,7,8,11]

for i in range(new_dataset_train_label_list.shape[0]):
    for j in range(27, 12):  #qua "12" dipende dal numero di AUs che abbiamo  (anche sotto)
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

np.savetxt('/home/trentini/ME-GraphAU/OpenGraphAU/data/AffWild2/list/AffWild2_train_label.txt', new_dataset_train_label_list ,fmt='%d', delimiter=' ')
np.savetxt('/home/trentini/ME-GraphAU/OpenGraphAU/data/AffWild2/list/AffWild2_val_label.txt', new_dataset_val_label_list ,fmt='%d', delimiter=' ')
np.savetxt('/home/trentini/ME-GraphAU/OpenGraphAU/data/AffWild2/list/AffWild2_test_label.txt', new_dataset_test_label_list ,fmt='%d', delimiter=' ')

with open('/home/trentini/ME-GraphAU/OpenGraphAU/data/AffWild2/list/AffWild2_train_img_path.txt', 'w+') as f:
    for line in new_dataset_train_img_list:
        f.write(line)

with open('/home/trentini/ME-GraphAU/OpenGraphAU/data/AffWild2/list/AffWild2_val_img_path.txt', 'w+') as f:
    for line in new_dataset_val_img_list:
        f.write(line)

with open('/home/trentini/ME-GraphAU/OpenGraphAU/data/AffWild2/list/AffWild2_test_img_path.txt', 'w+') as f:
    for line in new_dataset_test_img_list:
        f.write(line)

