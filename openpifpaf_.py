import numpy as np
import os

import openpifpaf
from PIL import Image
import json
from dataset import * 
from utils import *

import subprocess

image_paths = ['/work/vita/datasets/Aff-Wild2/cropped_aligned/430/00261.jpg', 
               '/work/vita/datasets/Aff-Wild2/cropped_aligned/430/00262.jpg',
               '/work/vita/datasets/Aff-Wild2/cropped_aligned/430/00263.jpg']
checkpoint = 'shufflenetv2k30-wholebody'
output_dir = '/home/trentini/'

for image_path in image_paths:
    json_output = output_dir + image_path.split('/')[-1].split('.')[0] + '.json'
    command = f'srun python -m openpifpaf.predict {image_path} --checkpoint={checkpoint} --json-output {json_output}'
    subprocess.run(command.split())