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

# def calculate_skeleton(image_path, output_dir):
#     checkpoint = 'shufflenetv2k30-wholebody'

#     # Create the output directory if it doesn't exist
#     os.makedirs(output_dir, exist_ok=True)

#     # Generate the JSON output file path
#     json_output = os.path.join(output_dir, os.path.splitext(os.path.basename(image_path))[0] + '.json')

#     # Load image
#     with open(image_path, 'rb') as f:
#         img = Image.open(f).convert('RGB')

#     # Create predictor object
#     predictor = openpifpaf.Predictor(checkpoint=checkpoint)

#     # Run prediction on image
#     pred = predictor.image(image_path)
    
#     # json output
#     json_out_name = os.path.join(output_dir, os.path.splitext(os.path.basename(image_path))[0] + '.predictions.json')
#     with open(json_out_name, 'w') as f:
#         json.dump([ann.json_data() for ann in pred[0]], f)