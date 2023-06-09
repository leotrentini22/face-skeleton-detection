
<div id="top"></div>

<br />
<div align="center">
<h1 align="center">Face Skeleton Detection</h1>

</div>

<details>
  <summary>Table of Contents</summary>
  <ol>
    <li><a href="#General-Information">General Information</a></li>
    <li><a href="#Requirements">Requirements</a></li>
    <li><a href="#Install-OpenPifPaf">Install OpenPifPaf</a></li>
    <li><a href="#Data">Data</a></li>
    <li><a href="#Structure">Structure</a></li>
    <li><a href="#Usage">Usage</a></li>
  </ol>
</details>

## General Information

The repository contains the code and report for the Face Skeleton Detection, an implementation of [OpenPifPaf](https://openpifpaf.github.io/intro.html) on our dataset AffWild2. The aim is to extract keypoints (skeletons) from facial images. This repository is part of a broader project that aims to adapt a general action recognition algorithm to a more specific face action units recognition task.

## Requirements
- Python 3
- PyTorch

## Install OpenPifPaf

Make sure there is no folder named openpifpaf in your current directory. To install OpenPifPaf, run this command:
   ```sh
   pip3 install openpifpaf
   ```

## Data

The Dataset we used:
  * [AffWild2](https://ibug.doc.ic.ac.uk/resources/aff-wild2/)

We provide tools for prepareing data in ```tool/```.

1. Download raw data files
2. Modify the file ```tool/openpifpaf_utils.py```, by setting your own personal paths (e.g. the path where you store the dataset, or the path where you would like to store the lists of AUs. More details are provided in this specific file)
3. From this folder, run:
   ```sh
   cd tool/
   python dataset_process.py
   python calculate_AU_class_weights.py
   ```

## Structure

- The folder `tool` contains all the methods used to list all the images, all the face action units and to extract skeletons with keypoints from the dataset
- `utils.py` and `dataset.py` contains useful functions for our implementation
- The folder `data` will contain all the paths of the images of the dataset, all the Face Action Units and all the skeletons.

## Usage

1. Clone the repo
   ```sh
   git clone https://github.com/leotrentini22/face-skeleton-detection.git
   ```
2. Follow the instructions contained in the [Data](#data) section
3. From this folder run in the terminal:
   ```sh
   cd tool/
   python skeleton_process_train.py
   python skeleton_process_validation.py
   python skeleton_process_test.py
   ```
This command will produce as output for each image in the dataset a json file, containing all the information about keypoints (please read [OpenPifPaf](https://openpifpaf.github.io/intro.html) readme for more information about the json output)

<p align="right">(<a href="#top">Back to top</a>)</p>
