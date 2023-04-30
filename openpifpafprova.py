import numpy as np
import os

import openpifpaf
from PIL import Image
import json
from dataset import * 
from utils import *

import imageio

print("entra qui")

def calculate_skeleton(image_path, output_dir):
    checkpoint = 'shufflenetv2k30-wholebody'

    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Generate the JSON output file path
    json_output = os.path.join(output_dir, os.path.splitext(os.path.basename(image_path))[0] + '.json')

    pifpaf = openpifpaf.PifPaf(checkpoint=checkpoint)

    # Load the input image
    image = PIL.Image.open(image_path)

    # Generate JSON output for the image
    predictions, _ = pifpaf.process_image(image)
    openpifpaf.io.CocoWriter().write(json_output, image_path, predictions, {}, {})

actual_image_path = '/work/vita/datasets/Aff-Wild2/cropped_aligned/430/00262.jpg'
skeleton_output_dir = '/home/trentini/face-skeleton-detection'
calculate_skeleton(actual_image_path, skeleton_output_dir)