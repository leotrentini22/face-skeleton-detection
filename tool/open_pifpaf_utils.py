import numpy as np
import os

import openpifpaf
from PIL import Image
import json
from dataset import * 
from utils import *

import subprocess

def calculate_skeleton(image_path, output_dir):
    checkpoint = 'shufflenetv2k30-wholebody'
    json_output = output_dir + image_path.split('/')[-1].split('.')[0] + '.json'
    command = f'srun python -m openpifpaf.predict {image_path} --checkpoint={checkpoint} --json-output {json_output}'
    subprocess.run(command.split())