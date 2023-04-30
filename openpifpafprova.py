import numpy as np
import os

import openpifpaf
from PIL import Image
import json
from dataset import * 
from utils import *



print("entra qui")

def out_name(arg, in_name, default_extension):
    """Determine an output name from args, input name and extension.
    arg can be:
    - none: return none (e.g. show image but don't store it)
    - True: activate this output and determine a default name
    - string:
        - not a directory: use this as the output file name
        - is a directory: use directory name and input name to form an output
    """
    if arg is None:
        return None

    if arg is True:
        return in_name + default_extension

    if os.path.isdir(arg):
        return os.path.join(
            arg,
            os.path.basename(in_name)
        ) + default_extension

    return arg

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
    pred = predictor.images([image_path])
    # json output
    json_out_name = os.path.join(output_dir, os.path.splitext(os.path.basename(image_path))[0] + '.predictions.json')
    with open(json_out_name, 'w') as f:
        json.dump([ann.json_data() for ann in pred], f)

actual_image_path = '/work/vita/datasets/Aff-Wild2/cropped_aligned/430/00262.jpg'
skeleton_output_dir = '/home/trentini/face-skeleton-detection'
calculate_skeleton(actual_image_path, skeleton_output_dir)