import numpy as np
import os

import openpifpaf
from PIL import Image
import json
from dataset import * 
from utils import *


def calculate_skeleton(image_path, output_dir):
    checkpoint = 'shufflenetv2k30-wholebody'

    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Generate the JSON output file path
    json_output = os.path.join(output_dir, os.path.splitext(os.path.basename(image_path))[0] + '.json')

    # Load image
    print(type(image_path))
    with open(image_path, 'rb') as f:
        img = Image.open(f).convert('RGB')

    # Create predictor object
    predictor = openpifpaf.Predictor(checkpoint=checkpoint)

    # Run prediction on image
    pred = predictor.image(image_path)
    print(type(pred))
    
    # json output
    json_out_name = os.path.join(output_dir, os.path.splitext(os.path.basename(image_path))[0] + '.predictions.json')
    with open(json_out_name, 'w') as f:
        json.dump([ann.json_data() for ann in pred[0]], f)

actual_image_path = '/work/vita/datasets/Aff-Wild2/cropped_aligned/430/00262.jpg'
skeleton_output_dir = '/home/trentini/face-skeleton-detection'
calculate_skeleton(actual_image_path, skeleton_output_dir)