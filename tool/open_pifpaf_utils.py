import numpy as np
import os

import openpifpaf
from PIL import Image
import json

def calculate_skeleton(image_path, output_dir):
    checkpoint = 'shufflenetv2k30-wholebody'

    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Load image
    print(image_path, flush=True)
    with open(image_path, 'rb') as f:
        img = Image.open(f).convert('RGB')

    # Create predictor object
    predictor = openpifpaf.Predictor(checkpoint=checkpoint)

    # Run prediction on image
    pred = predictor.image(image_path)
    
    directory_name = os.path.basename(os.path.dirname(image_path))
    file_name = os.path.splitext(os.path.basename(image_path))[0]
    new_file_name = f"{directory_name}.{file_name}.predictions.json"

    # json output
    json_out_name = os.path.join(output_dir, new_file_name)

    with open(json_out_name, 'w') as f:
        json.dump([ann.json_data() for ann in pred[0]], f)




#### FUNCTION TO SET YOUR PATHS

def set_your_paths():
    # CHANGE HERE PATHS 

    # list_path_prefix -> path of the folder where the list of paths of the images and the list of AUs will be stored
    # skeleton_path -> path of the folder where the skeletons json files will be stored 

    list_path_prefix = '/home/trentini/face-skeleton-detection/data/AffWild2/list/' 
    skeleton_path = '/home/trentini/face-skeleton-detection/data/AffWild2/skeletons/'


    # CHANGE HERE THE PATH OF THE DATASET
    # this will depend on where you store the dataser
    # img_path_vita -> folder in which are stored the images
    # label_root -> folder in which are stored the labels (i.e. the AUs)

    img_path_vita = '/work/vita/datasets/Aff-Wild2/cropped_aligned/'
    label_root = '/work/vita/datasets/Aff-Wild2/Third_ABAW_Annotations/AU_Detection_Challenge/'


    # names of the folders in label_root where there train_set, val_set and test_set are divided
    
    train_path ='Train_Set' 
    val_path = 'Validation_Set' 
    test_path = 'Validation_Set'   #there is no test set in our dataset

    return list_path_prefix, skeleton_path, img_path_vita, label_root, train_path, val_path, test_path