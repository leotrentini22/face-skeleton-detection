
<div id="top"></div>

<br />
<div align="center">
<h1 align="center">Face Skeleton Detection</h1>

</div>

<details>
  <summary>Table of Contents</summary>
  <ol>
    <li><a href="#General-Information">General Information</a></li>
    <li><a href="#Data">Data</a></li>
    <li><a href="#Structure">Structure</a></li>
    <li><a href="#Usage">Usage</a></li>
  </ol>
</details>

## General Information

The repository contains the code and report for the Face Skeleton Detection, an implementation of [OpenPifPaf](https://openpifpaf.github.io/intro.html) on our dataset AffWild2

## Data

The data we use come from the [AffWild2](https://www.aicrowd.com/challenges/epfl-machine-learning-higgs/dataset_files) dataset. Before running, please be sure to download the data and change the paths inside the scripts.

To prepare the data, run this command:
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
